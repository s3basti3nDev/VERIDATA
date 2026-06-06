#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate K verified runs: mean ± σ + Wilson confidence interval.

Two uncertainty types — never conflated:
  (a) σ_K  = run-to-run variability (model noise); indicative when K=3 (3 points)
  (b) Wilson 95% CI = sampling robustness on the n-question set (pooled across K)

Two modes:

  AGGREGATE — use existing JSONL files (zero API cost):
    python scripts/repeat_runs.py \\
      --perturbed A.jsonl B.jsonl C.jsonl \\
      --clean     Ac.jsonl Bc.jsonl Cc.jsonl

  LAUNCH — run K new repetitions via run_verified.py:
    python scripts/repeat_runs.py --manifest runs/perturbed/<id>/manifest.jsonl --k 3

Integrity gate (mandatory before recomputing 'abstained'):
  Verifies that veridata/invariants.py was NOT modified after the provided runs.
  If it was, the stored 'fired' values may not reflect current logic → refuses to
  proceed and requests a relaunch.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean as _mean, stdev as _stdev
from typing import Sequence

# Force UTF-8 output on Windows so Unicode characters print cleanly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from veridata.verifier import _ABSTAINING_INVARIANTS

_ALL_INVARIANTS = [
    "duplicate_rows",
    "numeric_outliers",
    "dtype_mismatch",
    "unexplained_constant",
]
_NA = float("nan")


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for proportion k/n."""
    if n == 0:
        return (_NA, _NA)
    z2 = z * z
    n_adj = n + z2
    p_adj = (k + z2 / 2) / n_adj
    margin = z * math.sqrt(max(0.0, p_adj * (1 - p_adj) / n_adj))
    return (max(0.0, p_adj - margin), min(1.0, p_adj + margin))


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Integrity check — invariants.py unchanged since runs were produced
# ---------------------------------------------------------------------------

def check_invariant_integrity(all_runs: list[list[dict]]) -> tuple[bool, str]:
    """Return (ok, message).

    Checks that veridata/invariants.py was NOT committed after the earliest
    run timestamp across all provided records. If it was, the stored 'fired'
    values may not match current logic and recomputing 'abstained' is INVALID.
    """
    ts_values: list[str] = [
        rec["ts"] for run in all_runs for rec in run if "ts" in rec
    ]
    if not ts_values:
        return True, "no 'ts' field in records — cannot verify (assuming OK)"

    ts_values.sort()
    earliest_str = ts_values[0]
    try:
        earliest_unix = datetime.fromisoformat(
            earliest_str.replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return True, f"cannot parse ts '{earliest_str}' — skipping integrity check"

    git_result = subprocess.run(
        ["git", "log", "-1", "--format=%ct %H %s", "--", "veridata/invariants.py"],
        capture_output=True, text=True, cwd=_PROJECT_ROOT,
    )
    if not git_result.stdout.strip():
        return True, "veridata/invariants.py has no git history — assuming OK"

    parts = git_result.stdout.strip().split(" ", 2)
    inv_unix = int(parts[0])
    inv_hash = parts[1][:8]
    inv_msg = parts[2] if len(parts) > 2 else ""
    inv_iso = datetime.fromtimestamp(inv_unix, tz=timezone.utc).isoformat()

    if inv_unix > earliest_unix:
        return False, (
            f"INTEGRITY FAILURE: veridata/invariants.py was modified AFTER the "
            f"earliest run.\n"
            f"  invariants.py last commit : {inv_hash} '{inv_msg}' ({inv_iso})\n"
            f"  Earliest run ts           : {earliest_str}\n"
            f"  The stored 'fired' values may not reflect current detection logic.\n"
            f"  Recomputing 'abstained' from these traces is INVALID.\n"
            f"  → Relaunch: python scripts/run_verified.py --manifest <manifest>"
        )

    return True, (
        f"OK — invariants.py last commit {inv_hash} '{inv_msg}' ({inv_iso})\n"
        f"         before earliest run {earliest_str}"
    )


# ---------------------------------------------------------------------------
# Distinctness check — are K runs truly independent?
# ---------------------------------------------------------------------------

def check_distinctness(perturbed_runs: list[list[dict]]) -> list[str]:
    """Compare generated_code across all pairs of runs.

    Returns warning strings if ≥ 90% of codes are identical across any pair
    (which would mean σ reflects rounding noise, not genuine model variance).
    """
    warnings: list[str] = []
    for i in range(len(perturbed_runs)):
        for j in range(i + 1, len(perturbed_runs)):
            ri = {r["question_idx"]: r.get("generated_code", "") for r in perturbed_runs[i]}
            rj = {r["question_idx"]: r.get("generated_code", "") for r in perturbed_runs[j]}
            common = set(ri) & set(rj)
            if not common:
                warnings.append(
                    f"  Run {i+1} vs {j+1}: no common question_idx — cannot compare"
                )
                continue
            n_identical = sum(1 for q in common if ri[q] == rj[q] and ri[q] != "")
            pct = n_identical / len(common)
            msg = (
                f"  Run {i+1} vs {j+1}: {n_identical}/{len(common)} identical "
                f"codes ({pct:.0%})"
            )
            if pct >= 0.90:
                warnings.append(
                    msg + " — ⚠ σ may NOT reflect true model variance (nearly deterministic)"
                )
            else:
                warnings.append(msg + " — OK")
    return warnings


# ---------------------------------------------------------------------------
# Recompute abstained under current policy
# ---------------------------------------------------------------------------

def recompute_record(rec: dict) -> dict:
    """Return copy of rec with 'abstained' and 'confidence' recomputed from trace."""
    invs = rec.get("invariants", [])
    abstaining_fired = [
        inv for inv in invs
        if inv["name"] in _ABSTAINING_INVARIANTS and inv.get("fired", False)
    ]
    abstained = bool(abstaining_fired)
    confidence = 1.0 - max((inv["severity"] for inv in abstaining_fired), default=0.0)
    return {**rec, "abstained": abstained, "confidence": confidence}


# ---------------------------------------------------------------------------
# Per-invariant analysis for one (perturbed, clean) pair
# ---------------------------------------------------------------------------

def _inv_fired(rec: dict, name: str) -> bool:
    return any(
        inv["name"] == name and inv.get("fired", False)
        for inv in rec.get("invariants", [])
    )


def _safe_mean(vals: Sequence[float]) -> float:
    lst = [v for v in vals if not math.isnan(v)]
    return _mean(lst) if lst else _NA


def analyze_pair(perturbed_raw: list[dict], clean_raw: list[dict]) -> dict:
    """Per-invariant metrics + SER metrics for one run pair."""
    perturbed = [recompute_record(r) for r in perturbed_raw]
    clean = [recompute_record(r) for r in clean_raw]
    sensitive = [r for r in perturbed if r.get("expected_sensitive", True)]

    result: dict = {}

    for inv in _ALL_INVARIANTS:
        fire_p = _safe_mean([float(_inv_fired(r, inv)) for r in sensitive])
        fire_c = _safe_mean([float(_inv_fired(r, inv)) for r in clean])
        delta = (fire_p - fire_c) if not (math.isnan(fire_p) or math.isnan(fire_c)) else _NA

        fires = [r for r in sensitive if _inv_fired(r, inv)]
        prec = _safe_mean([float(not r.get("correct", True)) for r in fires])
        errors = [r for r in sensitive if not r.get("correct", True)]
        recall = _safe_mean([float(_inv_fired(r, inv)) for r in errors])

        result[inv] = {
            "fire_rate_perturbed": fire_p,
            "fire_rate_clean": fire_c,
            "delta": delta,
            "precision": prec,
            "recall": recall,
            "n_sensitive": len(sensitive),
            "n_clean": len(clean),
            "n_fires_perturbed": len(fires),
            "n_errors": len(errors),
        }

    # SER metrics
    n = len(perturbed)
    n_correct = sum(1 for r in perturbed if r.get("correct", False))
    n_abstained = sum(1 for r in perturbed if r.get("abstained", False))
    n_silent = sum(
        1 for r in perturbed
        if not r.get("correct", False) and not r.get("abstained", False)
    )
    n_sur = sum(
        1 for r in perturbed
        if r.get("correct", False) and r.get("abstained", False)
    )
    n_clean = len(clean)
    n_clean_abs = sum(1 for r in clean if r.get("abstained", False))

    result["_ser"] = {
        "n": n,
        "n_sensitive": len(sensitive),
        "precision": n_correct / n if n else _NA,
        "ser_sans": (n - n_correct) / n if n else _NA,
        "ser_avec": n_silent / n if n else _NA,
        "coverage": 1.0 - n_abstained / n if n else _NA,
        "sur_abstention": n_sur / n if n else _NA,
        "fpr_clean": n_clean_abs / n_clean if n_clean else _NA,
        "n_silent": n_silent,
        "n_abstained": n_abstained,
        "n_clean_abstained": n_clean_abs,
        "n_clean": n_clean,
    }

    return result


# ---------------------------------------------------------------------------
# Aggregate K pairs → mean ± σ + Wilson CI
# ---------------------------------------------------------------------------

def aggregate(analyses: list[dict]) -> dict:
    """Mean ± σ(a) across K runs, Wilson CI(b) on pooled n."""
    k = len(analyses)

    def _ms(vals: list[float]) -> tuple[float, float]:
        vs = [v for v in vals if not math.isnan(v)]
        if not vs:
            return _NA, _NA
        m = _mean(vs)
        s = _stdev(vs) if len(vs) > 1 else _NA
        return m, s

    agg: dict = {"k": k}

    for inv in _ALL_INVARIANTS:
        rows = [a[inv] for a in analyses if inv in a]
        if not rows:
            continue

        fp_m, fp_s = _ms([r["fire_rate_perturbed"] for r in rows])
        fc_m, fc_s = _ms([r["fire_rate_clean"] for r in rows])
        d_m, d_s = _ms([r["delta"] for r in rows])
        pr_m, pr_s = _ms([r["precision"] for r in rows])
        re_m, re_s = _ms([r["recall"] for r in rows])

        # Wilson CI on pooled observations (b)
        total_fires_p = sum(r["n_fires_perturbed"] for r in rows)
        total_n_p = sum(r["n_sensitive"] for r in rows)
        wci_p = wilson_ci(total_fires_p, total_n_p)

        # Approximate fires_clean from rate × n (rate is float, round to int)
        total_fires_c = sum(
            round(r["fire_rate_clean"] * r["n_clean"])
            for r in rows
            if not math.isnan(r["fire_rate_clean"])
        )
        total_n_c = sum(
            r["n_clean"] for r in rows if not math.isnan(r["fire_rate_clean"])
        )
        wci_c = wilson_ci(total_fires_c, total_n_c)

        # Naive delta CI: difference of independent proportions (approximate)
        wci_d = (wci_p[0] - wci_c[1], wci_p[1] - wci_c[0])

        agg[inv] = {
            "fire_rate_perturbed": (fp_m, fp_s),
            "fire_rate_clean": (fc_m, fc_s),
            "delta": (d_m, d_s),
            "precision": (pr_m, pr_s),
            "recall": (re_m, re_s),
            "wilson_fire_perturbed": wci_p,
            "wilson_fire_clean": wci_c,
            "wilson_delta_approx": wci_d,
            "abstains": inv in _ABSTAINING_INVARIANTS,
        }

    # SER aggregation
    ser_rows = [a["_ser"] for a in analyses if "_ser" in a]
    ser_m, ser_s = _ms([r["ser_sans"] for r in ser_rows])
    sa_m, sa_s = _ms([r["ser_avec"] for r in ser_rows])
    cov_m, cov_s = _ms([r["coverage"] for r in ser_rows])
    sur_m, sur_s = _ms([r["sur_abstention"] for r in ser_rows])
    fpr_m, fpr_s = _ms([r["fpr_clean"] for r in ser_rows])

    total_n = sum(r["n"] for r in ser_rows)
    total_n_correct = sum(round(r["precision"] * r["n"]) for r in ser_rows)
    total_silent = sum(r["n_silent"] for r in ser_rows)
    total_abs = sum(r["n_abstained"] for r in ser_rows)
    total_n_clean = sum(r["n_clean"] for r in ser_rows)
    total_abs_clean = sum(r["n_clean_abstained"] for r in ser_rows)

    agg["_ser"] = {
        "ser_sans": (ser_m, ser_s),
        "ser_avec": (sa_m, sa_s),
        "coverage": (cov_m, cov_s),
        "sur_abstention": (sur_m, sur_s),
        "fpr_clean": (fpr_m, fpr_s),
        "wilson_ser_sans": wilson_ci(total_n - total_n_correct, total_n),
        "wilson_ser_avec": wilson_ci(total_silent, total_n),
        "wilson_coverage": wilson_ci(total_n - total_abs, total_n),
        "wilson_fpr": wilson_ci(total_abs_clean, total_n_clean),
    }

    return agg


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt(v: float, w: int = 6) -> str:
    return "N/A".rjust(w) if math.isnan(v) else f"{v:.3f}".rjust(w)


def _fmt_ms(m: float, s: float, indicate_k: bool = False) -> str:
    suffix = "(indicative, K=3)" if indicate_k else ""
    if math.isnan(m):
        return "N/A"
    if math.isnan(s):
        return f"{m:.3f}"
    return f"{m:.3f} ± {s:.3f} {suffix}".strip()


def _fmt_wci(lo: float, hi: float) -> str:
    if math.isnan(lo) or math.isnan(hi):
        return "N/A"
    return f"[{lo:.3f}, {hi:.3f}]"


def print_report(
    agg: dict,
    perturbation: str,
    integrity_msg: str,
    distinctness_warnings: list[str],
) -> None:
    k = agg["k"]
    print()
    print(f"{'='*72}")
    print(f"  VERIDATA — Aggregated results  perturbation={perturbation}  K={k}")
    print(f"{'='*72}")

    print("\n[1] Integrity check")
    for line in integrity_msg.splitlines():
        print(f"    {line}")

    print("\n[2] Distinctness check (generated_code comparison across runs)")
    if distinctness_warnings:
        for w in distinctness_warnings:
            print(f"  {w}")
    else:
        print("  (no pairs to compare)")

    print()
    print("[3] Per-invariant discrimination")
    print("    NOTE: (a) σ = run-to-run variance (K=3, indicative — 3 points)")
    print("          (b) Wilson 95% CI = sampling uncertainty on pooled questions")
    print()
    hdr = (
        f"  {'Invariant':<26} {'fire_pert(a)':>14} {'fire_clean(a)':>14} "
        f"{'delta(a)':>14}  {'Wilson delta(b)':>18}  {'abstains'}"
    )
    print(hdr)
    print("  " + "-" * len(hdr))

    for inv in _ALL_INVARIANTS:
        if inv not in agg:
            continue
        d = agg[inv]
        fp_m, fp_s = d["fire_rate_perturbed"]
        fc_m, fc_s = d["fire_rate_clean"]
        dm, ds = d["delta"]
        wlo, whi = d["wilson_delta_approx"]
        abstains = "YES" if d["abstains"] else "NO  (trace only)"

        fp_str = f"{fp_m:.3f}±{fp_s:.3f}" if not math.isnan(fp_s) else f"{fp_m:.3f}"
        fc_str = f"{fc_m:.3f}±{fc_s:.3f}" if not math.isnan(fc_s) else f"{fc_m:.3f}"
        d_str  = f"{dm:+.3f}±{ds:.3f}" if not math.isnan(ds) else f"{dm:+.3f}"
        wci_str = _fmt_wci(wlo, whi)

        print(
            f"  {inv:<26} {fp_str:>14} {fc_str:>14} {d_str:>14}  {wci_str:>18}  {abstains}"
        )

    print()
    print("    Precision and recall per invariant (mean ± σ):")
    for inv in _ALL_INVARIANTS:
        if inv not in agg:
            continue
        d = agg[inv]
        pr_m, pr_s = d["precision"]
        re_m, re_s = d["recall"]
        pr_str = _fmt_ms(pr_m, pr_s)
        re_str = _fmt_ms(re_m, re_s)
        print(f"      {inv:<26}  precision={pr_str}  recall={re_str}")

    s = agg["_ser"]
    print()
    print("[4] SER reduction (abstained recomputed under current policy)")
    print(f"    (a) mean ± σ over K={k} runs    (b) Wilson 95% CI on pooled n")
    print()

    def _row(label: str, ms: tuple, wci: tuple) -> None:
        m, sg = ms
        lo, hi = wci
        a_str = _fmt_ms(m, sg)
        b_str = _fmt_wci(lo, hi)
        print(f"    {label:<22}  (a) {a_str:<20}  (b) {b_str}")

    _row("SER_sans",    s["ser_sans"],    s["wilson_ser_sans"])
    _row("SER_avec",    s["ser_avec"],    s["wilson_ser_avec"])
    _row("Coverage",    s["coverage"],    s["wilson_coverage"])
    _row("FPR (clean)", s["fpr_clean"],   s["wilson_fpr"])
    _row("Sur-abstention", s["sur_abstention"], (_NA, _NA))
    print()
    print(f"{'='*72}")
    print()


# ---------------------------------------------------------------------------
# Launch mode — run K new repetitions via subprocess
# ---------------------------------------------------------------------------

def _find_new_jsonl(runs_dir: Path, before: set[Path]) -> Path | None:
    """Find the newest JSONL in runs_dir not in before."""
    current = set(runs_dir.glob("*.jsonl"))
    new = sorted(current - before, key=lambda p: p.stat().st_mtime)
    return new[-1] if new else None


def launch_runs(
    manifest: Path,
    k: int,
    config: Path,
) -> tuple[list[Path], list[Path]]:
    """Run K verified runs (perturbed + clean) via run_verified.py subprocess."""
    runs_dir = _PROJECT_ROOT / "runs"
    perturbed_paths: list[Path] = []
    clean_paths: list[Path] = []
    script = _PROJECT_ROOT / "scripts" / "run_verified.py"

    for rep in range(1, k + 1):
        print(f"\n  Launching run {rep}/{k} (perturbed)…")
        before = set(runs_dir.glob("*.jsonl"))
        subprocess.run(
            [sys.executable, str(script), "--manifest", str(manifest),
             "--config", str(config)],
            check=True,
        )
        p = _find_new_jsonl(runs_dir, before)
        if p is None:
            print(f"  ERROR: could not identify output JSONL for run {rep}")
            sys.exit(1)
        perturbed_paths.append(p)

        print(f"  Launching run {rep}/{k} (clean)…")
        before = set(runs_dir.glob("*.jsonl"))
        subprocess.run(
            [sys.executable, str(script), "--manifest", str(manifest),
             "--clean", "--config", str(config)],
            check=True,
        )
        c = _find_new_jsonl(runs_dir, before)
        if c is None:
            print(f"  ERROR: could not identify output JSONL for clean run {rep}")
            sys.exit(1)
        clean_paths.append(c)

    return perturbed_paths, clean_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate K verified runs with mean ± σ + Wilson CI"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--perturbed", nargs="+", type=Path,
        metavar="JSONL",
        help="Existing verified-perturbed JSONL files (aggregate mode)",
    )
    mode.add_argument(
        "--manifest", type=Path,
        help="Manifest JSONL to use for launching new runs (launch mode)",
    )
    parser.add_argument(
        "--clean", nargs="+", type=Path,
        metavar="JSONL",
        help="Existing verified-clean JSONL files (required in aggregate mode)",
    )
    parser.add_argument("--k", type=int, default=3, help="Number of repetitions (launch mode)")
    parser.add_argument(
        "--config", type=Path,
        default=_PROJECT_ROOT / "configs" / "baseline.toml",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write aggregate results as JSON to this file",
    )
    args = parser.parse_args()

    if args.manifest:
        # Launch mode
        print(f"Launch mode: K={args.k} on {args.manifest}")
        perturbed_paths, clean_paths = launch_runs(args.manifest, args.k, args.config)
    else:
        # Aggregate mode
        perturbed_paths = args.perturbed
        clean_paths = args.clean
        if not clean_paths or len(clean_paths) != len(perturbed_paths):
            print(
                "ERROR: --clean must provide the same number of files as --perturbed",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load all runs
    perturbed_runs = [load_jsonl(p) for p in perturbed_paths]
    clean_runs = [load_jsonl(c) for c in clean_paths]

    # Filter to complete runs (30 records)
    pairs: list[tuple[list[dict], list[dict]]] = []
    for i, (pr, cr) in enumerate(zip(perturbed_runs, clean_runs)):
        if len(pr) < 30:
            print(f"  ⚠ Skipping pair {i+1}: perturbed run has only {len(pr)} records (< 30)")
            continue
        if len(cr) < 30:
            print(f"  ⚠ Skipping pair {i+1}: clean run has only {len(cr)} records (< 30)")
            continue
        pairs.append((pr, cr))

    if not pairs:
        print("ERROR: no complete run pairs (≥30 records) found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Using {len(pairs)} complete run pairs")

    # Integrity check
    all_records = [r for pr, cr in pairs for r in pr + cr]
    ok, integrity_msg = check_invariant_integrity([all_records])
    if not ok:
        print(f"\n{integrity_msg}\n", file=sys.stderr)
        sys.exit(2)

    # Distinctness check
    dist_warnings = check_distinctness([pr for pr, _ in pairs])

    # Infer perturbation from first perturbed record
    perturbation = pairs[0][0][0].get("perturbation", "unknown")

    # Analyze each pair and aggregate
    analyses = [analyze_pair(pr, cr) for pr, cr in pairs]
    agg = aggregate(analyses)

    print_report(agg, perturbation, integrity_msg, dist_warnings)

    if args.json_out:
        # Serialize (tuples → lists)
        def _convert(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, float) and math.isnan(obj):
                return None
            return obj

        import dataclasses
        out = json.loads(json.dumps(agg, default=_convert))
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  JSON written to {args.json_out}")


if __name__ == "__main__":
    main()
