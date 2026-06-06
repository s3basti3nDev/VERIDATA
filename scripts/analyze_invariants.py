#!/usr/bin/env python3
"""Per-invariant discrimination table — VERIDATA Week 3 headline result.

For each invariant, measures:
  - fire_rate_perturbed : fraction of sensitive-perturbed records where invariant fired
  - fire_rate_clean     : fraction of clean records where invariant fired
  - delta               : fire_rate_perturbed − fire_rate_clean (discriminating power)
  - precision           : P(wrong | invariant fired on perturbed-sensitive)
  - recall              : P(fired | wrong AND sensitive-perturbed)
  - abstains            : whether the invariant participates in the abstention policy

The ``delta`` column is the key signal: a high delta means the invariant reliably
distinguishes perturbed-corrupted data from clean data without knowing the answer.

Usage
-----
    # Compare perturbed vs clean verified runs
    python scripts/analyze_invariants.py \\
        --perturbed runs/verified_row_duplication_....jsonl \\
        --clean     runs/verified_clean_....jsonl

    # Perturbed run only (no false-positive rate)
    python scripts/analyze_invariants.py \\
        --perturbed runs/verified_row_duplication_....jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Mirror the abstention policy from verifier.py
_ABSTAINING_INVARIANTS: frozenset[str] = frozenset({
    "duplicate_rows",
    "dtype_mismatch",
    "unexplained_constant",
})

_ALL_INVARIANTS = [
    "duplicate_rows",
    "numeric_outliers",
    "dtype_mismatch",
    "unexplained_constant",
]

_NA = float("nan")


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _inv_fired(record: dict, name: str) -> bool:
    for inv in record.get("invariants", []):
        if inv.get("name") == name and inv.get("fired"):
            return True
    return False


def _safe_mean(values: Iterable[bool]) -> float:
    lst = list(values)
    return sum(lst) / len(lst) if lst else _NA


def _analyze_invariant(
    perturbed: list[dict],
    clean: list[dict],
    inv_name: str,
) -> dict:
    # Only count sensitive questions for perturbed metrics
    sensitive = [r for r in perturbed if r.get("expected_sensitive", True)]

    fire_pert = _safe_mean(_inv_fired(r, inv_name) for r in sensitive)
    fire_clean = _safe_mean(_inv_fired(r, inv_name) for r in clean) if clean else _NA

    delta = (fire_pert - fire_clean) if not (math.isnan(fire_pert) or math.isnan(fire_clean)) else _NA

    # Precision: P(wrong | fired on sensitive-perturbed)
    fires = [r for r in sensitive if _inv_fired(r, inv_name)]
    precision = _safe_mean(not r.get("correct", True) for r in fires)

    # Recall: P(fired | wrong AND sensitive-perturbed)
    errors = [r for r in sensitive if not r.get("correct", True)]
    recall = _safe_mean(_inv_fired(r, inv_name) for r in errors)

    return {
        "invariant": inv_name,
        "fire_rate_perturbed": fire_pert,
        "fire_rate_clean": fire_clean,
        "delta": delta,
        "precision": precision,
        "recall": recall,
        "n_sensitive": len(sensitive),
        "n_clean": len(clean),
        "n_fires_perturbed": len(fires),
        "n_errors_perturbed": len(errors),
        "abstains": inv_name in _ABSTAINING_INVARIANTS,
    }


def _fmt(v: float, fmt: str = ".3f") -> str:
    return "N/A" if math.isnan(v) else format(v, fmt)


def _print_table(rows: list[dict], perturbation: str) -> None:
    print()
    print(f"Perturbation: {perturbation}")
    print(f"{'Invariant':<26} {'fire_pert':>9} {'fire_clean':>10} {'delta':>7}  "
          f"{'prec':>6}  {'recall':>6}  {'abstains':>8}")
    print("-" * 82)
    for r in rows:
        mark = "Y" if r["abstains"] else "N (trace)"
        print(
            f"  {r['invariant']:<24} {_fmt(r['fire_rate_perturbed']):>9} "
            f"{_fmt(r['fire_rate_clean']):>10} {_fmt(r['delta']):>7}  "
            f"{_fmt(r['precision']):>6}  {_fmt(r['recall']):>6}  {mark:>9}"
        )
    print()
    print(f"  n_sensitive={rows[0]['n_sensitive']}  n_clean={rows[0]['n_clean']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-invariant discrimination table (VERIDATA headline result)"
    )
    parser.add_argument(
        "--perturbed", type=Path, required=True,
        help="Verified JSONL run on perturbed tables",
    )
    parser.add_argument(
        "--clean", type=Path, default=None,
        help="Verified JSONL run on clean tables (same manifest, --clean flag). "
             "Required for fire_rate_clean and delta columns.",
    )
    parser.add_argument(
        "--json", dest="json_out", action="store_true",
        help="Also write results as JSON to stdout (after the table)",
    )
    args = parser.parse_args()

    perturbed_records = _load_jsonl(args.perturbed)
    clean_records = _load_jsonl(args.clean) if args.clean else []

    if not perturbed_records:
        print(f"ERROR: empty perturbed file: {args.perturbed}", file=sys.stderr)
        sys.exit(1)

    perturbation = perturbed_records[0].get("perturbation", "unknown")

    rows = [
        _analyze_invariant(perturbed_records, clean_records, name)
        for name in _ALL_INVARIANTS
    ]

    _print_table(rows, perturbation)

    if args.json_out:
        print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
