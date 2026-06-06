"""Verification layer for VERIDATA Week 3.

run_invariants() applies all four invariants to an agent result and returns a
VerificationResult with abstention decision, confidence score, and the full
audit trace.

Abstention policy
-----------------
Only invariants with demonstrated discriminating power trigger abstention.
``numeric_outliers`` is computed and included in the trace but does NOT trigger
abstention: on real-world DataBench tables (follower counts, revenues, sports
scores) it fires at the same rate on clean and perturbed data — delta ≈ 0.
An injected outlier is statistically indistinguishable from a real one, so
keeping it in the abstention decision degrades precision without improving recall.

Abstaining invariants and their target perturbation class:
    duplicate_rows        → row_duplication  (biases sum/mean/count)
    dtype_mismatch        → locale_format    (numeric col becomes string)
    unexplained_constant  → hallucinated constant (free-floating coefficient)

Confidence: 1.0 − max(severity of ABSTAINING fired invariants). 1.0 if none.
Trace: ALL four invariants are returned so the audit log is complete.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import Config
from .invariants import (
    InvariantResult,
    duplicate_rows,
    dtype_mismatch,
    numeric_outliers,
    unexplained_constant,
)


# Invariants that participate in the abstention decision.
# numeric_outliers is excluded: fires equally on clean and perturbed data
# (injected outlier ≡ real outlier, statistically).
_ABSTAINING_INVARIANTS: frozenset[str] = frozenset({
    "duplicate_rows",
    "dtype_mismatch",
    "unexplained_constant",
})


@dataclass
class VerificationResult:
    abstained: bool          # True if an ABSTAINING invariant fired
    confidence: float        # 1.0 − max(severity of abstaining fired); 1.0 if none
    invariants: list[InvariantResult]   # full trace — all four invariants


def run_invariants(
    df: pd.DataFrame,
    code: str,
    result_str: str,
    question: str,
    config: Config,
) -> VerificationResult:
    """Run all invariants and return the verification decision + audit trace.

    Args:
        df:         the DataFrame the agent operated on.
        code:       generated Python code string.
        result_str: the agent's string answer.
        question:   the original question text.
        config:     loaded Config (invariant thresholds in config.invariants).
    """
    inv_cfg = config.invariants

    results: list[InvariantResult] = [
        duplicate_rows(
            df, code, result_str, question,
            threshold=inv_cfg.duplicate_row_threshold,
        ),
        numeric_outliers(
            df, code, result_str, question,
            k=inv_cfg.outlier_iqr_factor,
        ),
        dtype_mismatch(df, code, result_str, question),
        unexplained_constant(
            df, code, result_str, question,
            trivials=inv_cfg.trivial_constants,
        ),
    ]

    abstaining_fired = [r for r in results if r.fired and r.name in _ABSTAINING_INVARIANTS]
    abstained = len(abstaining_fired) > 0
    max_severity = max((r.severity for r in abstaining_fired), default=0.0)
    confidence = 1.0 - max_severity

    return VerificationResult(
        abstained=abstained,
        confidence=confidence,
        invariants=results,
    )


def verification_to_dict(vr: VerificationResult) -> dict:
    """Serialize VerificationResult for JSONL storage."""
    return {
        "abstained": vr.abstained,
        "confidence": vr.confidence,
        "invariants": [asdict(r) for r in vr.invariants],
    }
