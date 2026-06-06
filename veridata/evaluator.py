"""Wrapper around databench-eval for loading data and scoring predictions.

API notes (databench-eval >= 4.0):
- load_qa(name, split) → HuggingFace Dataset (normalised to list[dict] on load).
  Keys: question, answer, type, dataset, columns_used, sample_answer.
- load_table(dataset_id) → pd.DataFrame.
- Evaluator().eval(responses) was removed from this module.

Why we do NOT use Evaluator().eval()
-------------------------------------
Evaluator.eval() computes accuracy against its *own internal copy* of the full
1822-question dataset.  When we pass N < 1822 responses, it compares
responses[i] against internal_qa[i], but the internal ordering may differ from
the ordering returned by load_qa() — causing a severe alignment mismatch
(observed: 0.0022 accuracy on a run where manual inspection showed ~82%).

Fix: score() accepts both responses and the sample used to generate them, then
calls normalized_compare(response, ground_truth, type) for each aligned pair.
The verdict is also written to the JSONL as a ``correct`` field so every run
is auditable line by line.

Normalized comparator
---------------------
- number:          relative tolerance 1e-4 (absolute fallback when truth == 0)
- list[category]:  unordered set comparison, case-insensitive
- all others:      case-insensitive exact match after strip
"""

from typing import Optional

import pandas as pd
from databench_eval.utils import load_qa, load_table


# ---------------------------------------------------------------------------
# Normalized comparator — public so run_baseline.py can use it per-record
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

    - ``number``         → relative tolerance 1e-4
    - ``list[category]`` → unordered set, case-insensitive
    - all others         → case-insensitive exact match after strip
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
            # load_qa returns a HuggingFace Dataset; normalise to list[dict].
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

        Each ``responses[i]`` is compared against ``sample[i]["answer"]`` using
        ``normalized_compare``.  This guarantees alignment and is identical to
        the ``correct`` field written to the JSONL — so the displayed accuracy
        is always consistent with the per-record audit trail.

        Raises ``ValueError`` if lengths differ (alignment guard).
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
