"""Wrapper around databench-eval for loading data and scoring predictions.

API assumptions (verified against databench-eval >= 4.0):
- load_qa(name, split) → list[dict] with keys: question, answer, type, dataset, ...
- load_table(dataset_id) → pd.DataFrame
- Evaluator().eval(responses) → float accuracy, comparing responses[i] against
  internal ground truth[i]. Passing fewer responses than the full dataset is
  supported — only the first len(responses) entries are evaluated.
"""

from typing import Optional

import pandas as pd
from databench_eval import Evaluator
from databench_eval.utils import load_qa, load_table


class BaselineEvaluator:
    """Loads DataBench data and computes official accuracy scores."""

    def __init__(self, name: str = "semeval", split: str = "dev") -> None:
        self._name = name
        self._split = split
        self._evaluator = Evaluator()
        self._qa: Optional[list[dict]] = None
        self._table_cache: dict[str, pd.DataFrame] = {}

    def load_sample(self, n: int) -> list[dict]:
        """Return the first ``n`` QA entries in dataset order (deterministic)."""
        if self._qa is None:
            raw = load_qa(name=self._name, split=self._split)
            # load_qa may return a HuggingFace Dataset; normalise to list[dict]
            # so that row["dataset"] works regardless of the underlying type.
            self._qa = list(raw) if not isinstance(raw, list) else raw
        return self._qa[:n]

    def load_table_for(self, row: dict) -> pd.DataFrame:
        """Load and cache the DataFrame for a QA row."""
        table_id: str = row["dataset"]
        if table_id not in self._table_cache:
            self._table_cache[table_id] = load_table(table_id)
        return self._table_cache[table_id]

    def score(self, responses: list[str]) -> float:
        """Return official accuracy via databench-eval's Evaluator.

        ``responses`` must be ordered to match the entries returned by
        ``load_sample`` (first N entries of the dataset, in order).
        """
        return self._evaluator.eval(responses)
