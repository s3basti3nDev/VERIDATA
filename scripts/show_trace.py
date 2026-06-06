#!/usr/bin/env python3
"""Display the invariant audit trace for a specific question in a verified run.

Usage
-----
    python scripts/show_trace.py --run runs/verified_....jsonl --question_idx 5

Shows: question, ground truth, answer, abstained/confidence, generated code,
and the full invariant table (all invariants, fired or not).
"""

import argparse
import json
import sys
from pathlib import Path

# Mirrors verifier._ABSTAINING_INVARIANTS — trace-only invariants do not
# trigger abstention (no discriminating power on real DataBench tables).
_TRACE_ONLY = frozenset({"numeric_outliers"})


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _fmt_bool(v: bool) -> str:
    return "YES" if v else "no"


def main() -> None:
    parser = argparse.ArgumentParser(description="Display invariant trace for one question")
    parser.add_argument("--run", type=Path, required=True, help="Path to verified JSONL run file")
    parser.add_argument("--question_idx", type=int, required=True, help="0-based question index")
    args = parser.parse_args()

    records = _load_jsonl(args.run)
    matches = [r for r in records if r.get("question_idx") == args.question_idx]

    if not matches:
        print(f"No record found for question_idx={args.question_idx} in {args.run}")
        sys.exit(1)

    rec = matches[0]

    print("=" * 72)
    print(f"  Question [{rec.get('question_idx')}]  dataset={rec.get('dataset', '?')}")
    print("=" * 72)
    print(f"  Question    : {rec['question']}")
    print(f"  Ground truth: {rec['ground_truth']}")
    print(f"  Answer      : {rec['answer']}")
    print(f"  Correct     : {_fmt_bool(rec.get('correct', False))}")
    print(f"  Abstained   : {_fmt_bool(rec.get('abstained', False))}"
          f"  |  Confidence: {rec.get('confidence', 1.0):.3f}")
    print()
    print("  Generated code:")
    for line in rec.get("generated_code", "").splitlines():
        print(f"    {line}")
    print()

    invariants = rec.get("invariants", [])
    if not invariants:
        print("  (no invariant trace — run was not produced by run_verified.py)")
        return

    print("  Invariant trace:  (* = trace only, does not trigger abstention)")
    print(f"  {'Name':<28} {'Fired':<12} {'Sev':>5}  Detail")
    print("  " + "-" * 70)
    for inv in invariants:
        trace_only = inv["name"] in _TRACE_ONLY
        marker = "*" if trace_only else " "
        if inv["fired"]:
            fired_str = "(trace)" if trace_only else "FIRED"
        else:
            fired_str = "ok"
        sev_str = f"{inv['severity']:.2f}"
        detail = inv["detail"]
        if len(detail) > 50:
            detail = detail[:47] + "…"
        print(f"  {marker}{inv['name']:<27} {fired_str:<12} {sev_str:>5}  {detail}")

    print("=" * 72)


if __name__ == "__main__":
    main()
