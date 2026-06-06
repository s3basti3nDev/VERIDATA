"""Wrapper around databench-eval for loading data and scoring predictions.

API notes (databench-eval >= 4.0):
- load_qa(name, split) → HuggingFace Dataset (normalised to list[dict] on load).
  Keys: question, answer, type, dataset, columns_used, sample_answer.
- load_table(dataset_id) → pd.DataFrame.

Why we do NOT use Evaluator().eval()
-------------------------------------
Evaluator.eval() compares responses against its own internal QA copy whose
ordering may differ from load_qa() — producing severe alignment mismatch
(observed: accuracy=0.0022 on a run that was manually ~82%).

score(responses, sample) computes accuracy locally against the sample rows
that were actually used to generate the responses, guaranteeing alignment.
The verdict is also stored as a ``correct`` field in the JSONL for per-record
auditability.

Normalized comparator — design decisions
-----------------------------------------
DataBench answer types: boolean | number | category | list[number] | list[category]

number:
  Relative tolerance 1e-4.  Absolute fallback when truth == 0.
  "42.0" == "42", "1234.57" == "1234.5699…"

list[category] and list[number]:
  The LLM frequently returns Python list literals (['a','b']) while the
  ground truth is CSV ("a, b").  The previous set-based comparator failed to
  strip brackets/quotes, producing false negatives for any multi-element list
  where the sets differed after the malformed split.

  Fix: _parse_list normalises both sides identically:
    1. Strip surrounding [ ].
    2. Split on ','.
    3. Per item: strip whitespace, strip matching '' or "" quotes, lowercase.

  Ordering: DataBench evaluates list answers in order (position matters).
  _compare_list_category and _compare_list_number use list equality, not set.

boolean / category:
  Case-insensitive exact match after strip.
"""

from typing import Optional

import pandas as pd
from databench_eval.utils import load_qa, load_table


# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------

def _parse_list(s: str) -> list[str]:
    """Normalise a list answer to stripped, lowercased strings.

    Accepts both Python list-literal format (['a','b']) and CSV (a, b).
    Strips surrounding [ ], then for each comma-separated token strips
    whitespace and matching surrounding quotes.
    """
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    items: list[str] = []
    for raw in s.split(","):
        item = raw.strip()
        if len(item) >= 2 and item[0] in ("'", '"') and item[-1] == item[0]:
            item = item[1:-1].strip()
        if item:
            items.append(item.lower())
    return items


# ---------------------------------------------------------------------------
# Per-type comparators — public for direct testing
# ---------------------------------------------------------------------------

def _compare_number(value: str, truth: str) -> bool:
    """Relative tolerance 1e-4; absolute fallback when truth == 0."""
    v = float(str(value).strip().replace(",", "."))
    t = float(str(truth).strip().replace(",", "."))
    if t == 0.0:
        return abs(v - t) < 1e-4
    return abs(v - t) / abs(t) < 1e-4


def _compare_list_category(value: str, truth: str) -> bool:
    """Ordered list comparison for list[category].

    Normalises both sides via _parse_list (handles list-literal and CSV).
    Ordered per DataBench spec: position matters.
    """
    return _parse_list(value) == _parse_list(truth)


def _compare_list_number(value: str, truth: str) -> bool:
    """Ordered list comparison for list[number] with per-element tolerance 1e-4."""
    v_items = _parse_list(value)
    t_items = _parse_list(truth)
    if len(v_items) != len(t_items):
        return False
    try:
        return all(_compare_number(v, t) for v, t in zip(v_items, t_items))
    except (ValueError, TypeError):
        return False


def normalized_compare(value: str, truth: str, semantic: str) -> bool:
    """Normalized comparator for all DataBench answer types.

    - ``number``         → relative tolerance 1e-4
    - ``list[category]`` → ordered, case-insensitive, handles list-literal & CSV
    - ``list[number]``   → ordered, per-element tolerance 1e-4
    - ``boolean``        → case-insensitive exact match
    - ``category``       → case-insensitive exact match
    """
    try:
        if semantic == "number":
            return _compare_number(value, truth)
        if semantic == "list[category]":
            return _compare_list_category(value, truth)
        if semantic == "list[number]":
            return _compare_list_number(value, truth)
        return str(value).strip().lower() == str(truth).strip().lower()
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Evaluator wrapper
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """Loads DataBench data and computes locally-aligned accuracy scores."""

    def __init__(self, name: str = "semeval", split: str = "dev") -> None:
        self._name = name
        self._split = split
        self._qa: Optional[list[dict]] = None
        self._table_cache: dict[str, pd.DataFrame] = {}

    def load_sample(self, n: int) -> list[dict]:
        """Return the first ``n`` QA entries in dataset order (deterministic)."""
        if self._qa is None:
            raw = load_qa(name=self._name, split=self._split)
            self._qa = list(raw) if not isinstance(raw, list) else raw
        return self._qa[:n]

    def load_table_for(self, row: dict) -> pd.DataFrame:
        """Load and cache the DataFrame for a QA row."""
        table_id: str = row["dataset"]
        if table_id not in self._table_cache:
            self._table_cache[table_id] = load_table(table_id)
        return self._table_cache[table_id]

    def score(self, responses: list[str], sample: list[dict]) -> float:
        """Return accuracy computed locally from aligned (response, QA-row) pairs.

        Each ``responses[i]`` is compared against ``sample[i]`` using
        ``normalized_compare``.  Raises ``ValueError`` on length mismatch.
        """
        if len(responses) != len(sample):
            raise ValueError(
                f"Alignment mismatch: {len(responses)} responses vs "
                f"{len(sample)} sample rows."
            )
        if not sample:
            return 0.0
        n_correct = sum(
            normalized_compare(
                resp,
                str(row.get("answer", "")),
                str(row.get("type", "")),
            )
            for resp, row in zip(responses, sample)
        )
        return n_correct / len(sample)
