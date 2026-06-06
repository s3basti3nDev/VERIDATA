"""Wrapper around databench-eval for loading data and scoring predictions.

API assumptions (verified against databench-eval >= 4.0):
- load_qa(name, split) → HuggingFace Dataset (normalised to list[dict] on load)
  Keys: question, answer, type, dataset, columns_used, sample_answer
- load_table(dataset_id) → pd.DataFrame
- Evaluator(compare=fn).eval(responses) → float accuracy.
  compare(value, truth, semantic) is called per answer; semantic is the
  "type" field from the QA row (boolean | number | category |
  list[number] | list[category]).

Normalized comparator
---------------------
The default Evaluator uses exact string matching, which underestimates
accuracy for two answer types:
- number:          "42.0" ≠ "42", "1234.57" ≠ "1234.5699…"
- list[category]:  "a, b" ≠ "b, a"

normalized_compare fixes both while delegating all other types to a
case-insensitive strip match.  This comparator is passed to every
Evaluator instance so that BaselineEvaluator.score() returns the
corrected accuracy that becomes the SER reference baseline.
"""

from typing import Optional

import pandas as pd
from databench_eval import Evaluator
from databench_eval.utils import load_qa, load_table


# ---------------------------------------------------------------------------
# Normalized comparator
# ---------------------------------------------------------------------------

def _compare_number(value: str, truth: str) -> bool:
    """Relative tolerance 1e-4; falls back to absolute when truth == 0."""
    v = float(str(value).strip().replace(",", "."))
    t = float(str(truth).strip().replace(",", "."))
    if t == 0.0:
        return abs(v - t) < 1e-4
    return abs(v - t) / abs(t) < 1e-4


def _compare_list_category(value: str, truth: str) -> bool:
    """Order-insensitive set comparison, case-insensitive, strip-normalised."""
    def to_set(s: str) -> set[str]:
        return {item.strip().lower() for item in s.split(",") if item.strip()}
    return to_set(value) == to_set(truth)


def normalized_compare(value: str, truth: str, semantic: str) -> bool:
    """Normalized comparator for DataBench answer types.

    Handles the two types where exact string matching underestimates accuracy:
    - ``number``        → relative tolerance 1e-4
    - ``list[category]`` → unordered set comparison, case-insensitive

    All other types (boolean, category, list[number]) use case-insensitive
    exact match after stripping whitespace.
    """
    try:
        if semantic == "number":
            return _compare_number(value, truth)
        if semantic == "list[category]":
            return _compare_list_category(value, truth)
        return str(value).strip().lower() == str(truth).strip().lower()
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """Loads DataBench data and computes normalized accuracy scores."""

    def __init__(self, name: str = "semeval", split: str = "dev") -> None:
        self._name = name
        self._split = split
        self._evaluator = Evaluator(compare=normalized_compare)
        self._qa: Optional[list[dict]] = None
        self._table_cache: dict[str, pd.DataFrame] = {}

    def load_sample(self, n: int) -> list[dict]:
        """Return the first ``n`` QA entries in dataset order (deterministic)."""
        if self._qa is None:
            raw = load_qa(name=self._name, split=self._split)
            # load_qa returns a HuggingFace Dataset; normalise to list[dict].
            self._qa = list(raw) if not isinstance(raw, list) else raw
        return self._qa[:n]

    def load_table_for(self, row: dict) -> pd.DataFrame:
        """Load and cache the DataFrame for a QA row."""
        table_id: str = row["dataset"]
        if table_id not in self._table_cache:
            self._table_cache[table_id] = load_table(table_id)
        return self._table_cache[table_id]

    def score(self, responses: list[str]) -> float:
        """Return normalized accuracy via databench-eval's Evaluator.

        Uses ``normalized_compare`` for ``number`` (rel. tol. 1e-4) and
        ``list[category]`` (set-based, case-insensitive).  This is the
        corrected accuracy that serves as the SER reference baseline.

        ``responses`` must be ordered to match entries from ``load_sample``
        (first N entries of the dataset split, in order).
        """
        return self._evaluator.eval(responses)
