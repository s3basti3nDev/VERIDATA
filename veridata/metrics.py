"""SER, precision, and coverage metrics for VERIDATA runs.

Reads lists of JSONL records (dicts) with at minimum the ``correct`` field.
``abstained`` and ``confidence`` are optional (None-safe) — ready for Week 3.

Week 2 invariant: abstained=None everywhere → SER = 1 − precision.
"""

from typing import Any


def compute(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute precision, SER, and coverage from a list of JSONL records.

    - precision = mean(correct)
    - coverage  = mean(abstained != True)   # fraction that gave an answer
    - SER       = mean(not correct AND abstained != True)

    In Week 2 (abstained=None everywhere): SER = 1 - precision.
    """
    n = len(records)
    if n == 0:
        return {"n": 0, "precision": 0.0, "SER": 0.0, "coverage": 0.0}

    n_correct = sum(1 for r in records if r.get("correct") is True)
    n_answered = sum(1 for r in records if r.get("abstained") is not True)
    n_silent_error = sum(
        1 for r in records
        if r.get("correct") is not True and r.get("abstained") is not True
    )

    return {
        "n": n,
        "precision": n_correct / n,
        "SER": n_silent_error / n,
        "coverage": n_answered / n,
    }


def compare(
    clean: list[dict[str, Any]],
    perturbed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare clean vs perturbed run metrics.

    Returns a dict with precision/SER/coverage for both runs,
    plus delta_precision (perturbed - clean, negative = degradation).
    """
    mc = compute(clean)
    mp = compute(perturbed)
    return {
        "n_clean": mc["n"],
        "n_perturbed": mp["n"],
        "precision_clean": mc["precision"],
        "precision_perturbed": mp["precision"],
        "delta_precision": mp["precision"] - mc["precision"],
        "SER_clean": mc["SER"],
        "SER_perturbed": mp["SER"],
        "coverage_clean": mc["coverage"],
        "coverage_perturbed": mp["coverage"],
    }
