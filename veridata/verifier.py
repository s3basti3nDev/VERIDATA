"""Verification layer for VERIDATA Week 3.

run_invariants() applies all four invariants to an agent result and returns a
VerificationResult with abstention decision, confidence score, and the full
audit trace.

Design principles
-----------------
- Abstention policy: conservative — any fired invariant → abstained=True.
  A single data-quality violation is enough to distrust the result.
- Confidence: 1.0 − max(severity of fired invariants). Stays 1.0 if none fire.
- Trace: ALL invariants are returned (not just fired ones) so the audit log
  shows what was checked, not only what failed.
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


@dataclass
class VerificationResult:
    abstained: bool          # True if at least one invariant fired
    confidence: float        # 1.0 − max(severity of fired); 1.0 if none fired
    invariants: list[InvariantResult]   # full trace — all invariants, fired or not


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

    fired = [r for r in results if r.fired]
    abstained = len(fired) > 0
    max_severity = max((r.severity for r in fired), default=0.0)
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
