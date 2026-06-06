#!/usr/bin/env python3
"""Generate report figures from verified-run JSONL files.

Produces three PNG files in the output directory:
  fig1_discrimination.png  — fire_rate_perturbed vs fire_rate_clean per mode
  fig2_ser_reduction.png   — SER_sans vs SER_avec per mode
  fig3_stability_k3.png    — K=3 row_duplication: delta ± σ and FPR ± σ

All invariant metrics are computed directly from the trace ('fired' field),
not from the stored 'abstained' field. The abstention policy is reapplied at
read time so the figures are consistent with the current veridata/verifier.py.

Usage
-----
  python scripts/make_figures.py \\
    --row-dup-perturbed  runs/verified_row_duplication_A.jsonl  B.jsonl  C.jsonl \\
    --row-dup-clean      runs/verified_clean_A.jsonl  B.jsonl  C.jsonl \\
    --locale-perturbed   runs/verified_locale_format_A.jsonl  B.jsonl  C.jsonl \\
    --locale-clean       runs/verified_clean_A.jsonl  B.jsonl  C.jsonl \\
    --outlier-perturbed  runs/verified_outlier_injection_A.jsonl  B.jsonl \\
    --outlier-clean      runs/verified_clean_A.jsonl  B.jsonl \\
    --out report/figures
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Import the core analysis functions from repeat_runs
from scripts.repeat_runs import (
    _ABSTAINING_INVARIANTS,
    _inv_fired,
    _safe_mean,
    _NA,
    aggregate,
    analyze_pair,
    load_jsonl,
    recompute_record,
    wilson_ci,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_BLUE   = "#2979b5"
_ORANGE = "#e07b39"
_GREEN  = "#4caf79"
_RED    = "#d64242"
_GREY   = "#b0b0b0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

_MODE_LABELS = {
    "row_duplication":   "Row duplication",
    "locale_format":     "Locale format",
    "outlier_injection": "Outlier injection",
}

_INV_LABELS = {
    "duplicate_rows":       "duplicate_rows",
    "numeric_outliers":     "numeric_outliers",
    "dtype_mismatch":       "dtype_mismatch",
    "unexplained_constant": "unexplained_constant",
}

# Invariant most relevant per mode (headline pair for Fig 1)
_MODE_PRIMARY_INV = {
    "row_duplication":   "duplicate_rows",
    "locale_format":     "dtype_mismatch",
    "outlier_injection": "numeric_outliers",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_runs(paths: list[Path]) -> list[list[dict]]:
    """Load and filter to complete runs (≥ 30 records)."""
    runs = []
    for p in paths:
        records = load_jsonl(p)
        if len(records) >= 30:
            runs.append(records)
        else:
            print(f"  ⚠ Skipping {p.name} ({len(records)} records < 30)")
    return runs


def pooled_metrics(
    perturbed_runs: list[list[dict]],
    clean_runs: list[list[dict]],
    inv_name: str,
) -> dict:
    """Compute pooled fire rates for one invariant across K runs."""
    sensitive_all = [
        r for run in perturbed_runs for r in run if r.get("expected_sensitive", True)
    ]
    clean_all = [r for run in clean_runs for r in run]

    n_sensitive = len(sensitive_all)
    n_clean = len(clean_all)

    fires_p = sum(_inv_fired(r, inv_name) for r in sensitive_all)
    fires_c = sum(_inv_fired(r, inv_name) for r in clean_all)

    rate_p = fires_p / n_sensitive if n_sensitive else _NA
    rate_c = fires_c / n_clean if n_clean else _NA
    delta  = (rate_p - rate_c) if not (math.isnan(rate_p) or math.isnan(rate_c)) else _NA

    wci_p = wilson_ci(fires_p, n_sensitive)
    wci_c = wilson_ci(fires_c, n_clean)

    errors = [r for r in sensitive_all if not r.get("correct", True)]
    recall = (
        sum(_inv_fired(r, inv_name) for r in errors) / len(errors)
        if errors else _NA
    )
    fires_list = [r for r in sensitive_all if _inv_fired(r, inv_name)]
    precision = (
        sum(not r.get("correct", True) for r in fires_list) / len(fires_list)
        if fires_list else _NA
    )

    return {
        "inv_name": inv_name,
        "rate_p": rate_p, "rate_c": rate_c, "delta": delta,
        "wci_p": wci_p, "wci_c": wci_c,
        "precision": precision, "recall": recall,
        "n_sensitive": n_sensitive, "n_clean": n_clean,
    }


def pooled_ser(
    perturbed_runs: list[list[dict]],
    clean_runs: list[list[dict]],
) -> dict:
    """Compute SER_sans, SER_avec, coverage, FPR on pooled records."""
    all_pert = [recompute_record(r) for run in perturbed_runs for r in run]
    all_clean = [recompute_record(r) for run in clean_runs for r in run]

    n = len(all_pert)
    n_correct  = sum(r.get("correct", False) for r in all_pert)
    n_abstained = sum(r.get("abstained", False) for r in all_pert)
    n_silent   = sum(
        1 for r in all_pert
        if not r.get("correct", False) and not r.get("abstained", False)
    )
    n_sur = sum(
        1 for r in all_pert
        if r.get("correct", False) and r.get("abstained", False)
    )
    n_clean = len(all_clean)
    n_clean_abs = sum(r.get("abstained", False) for r in all_clean)

    return {
        "ser_sans":  (n - n_correct) / n if n else _NA,
        "ser_avec":  n_silent / n if n else _NA,
        "coverage":  1.0 - n_abstained / n if n else _NA,
        "sur_abstention": n_sur / n if n else _NA,
        "fpr_clean": n_clean_abs / n_clean if n_clean else _NA,
        "n": n,
        "wilson_ser_avec": wilson_ci(n_silent, n),
        "wilson_ser_sans": wilson_ci(n - n_correct, n),
    }


# ---------------------------------------------------------------------------
# Figure 1 — Discrimination by invariant
# ---------------------------------------------------------------------------

def fig1_discrimination(
    mode_data: dict[str, tuple[list[list[dict]], list[list[dict]]]],
    out: Path,
) -> None:
    """Grouped bars: fire_rate_perturbed vs fire_rate_clean for each mode.

    Shows the relevant invariant per mode (headline comparison) plus
    numeric_outliers (the non-discriminating baseline) side by side.
    """
    modes = list(mode_data.keys())
    invs_to_show = ["duplicate_rows", "dtype_mismatch", "numeric_outliers"]

    inv_labels = {
        "duplicate_rows": "dup_rows",
        "dtype_mismatch": "dtype_mm",
        "numeric_outliers": "num_out*",
    }

    fig, axes = plt.subplots(1, len(modes), figsize=(12, 4.5), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        pert_runs, clean_runs = mode_data[mode]
        x = np.arange(len(invs_to_show))
        w = 0.35

        rates_p, rates_c = [], []
        errs_p_lo, errs_p_hi = [], []
        errs_c_lo, errs_c_hi = [], []

        for inv in invs_to_show:
            m = pooled_metrics(pert_runs, clean_runs, inv)
            rates_p.append(m["rate_p"] if not math.isnan(m["rate_p"]) else 0.0)
            rates_c.append(m["rate_c"] if not math.isnan(m["rate_c"]) else 0.0)
            # Wilson CI half-widths
            lo_p, hi_p = m["wci_p"]
            errs_p_lo.append(rates_p[-1] - lo_p)
            errs_p_hi.append(hi_p - rates_p[-1])
            lo_c, hi_c = m["wci_c"]
            errs_c_lo.append(rates_c[-1] - lo_c)
            errs_c_hi.append(hi_c - rates_c[-1])

        ax.bar(x - w/2, rates_p, w, label="Perturbed",
               color=_ORANGE, alpha=0.85,
               yerr=[errs_p_lo, errs_p_hi], capsize=4, error_kw={"elinewidth": 1.2})
        ax.bar(x + w/2, rates_c, w, label="Clean",
               color=_BLUE, alpha=0.85,
               yerr=[errs_c_lo, errs_c_hi], capsize=4, error_kw={"elinewidth": 1.2})

        ax.set_xticks(x)
        ax.set_xticklabels([inv_labels[i] for i in invs_to_show], fontsize=9)
        ax.set_title(_MODE_LABELS.get(mode, mode), fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1.15)
        if ax is axes[0]:
            ax.set_ylabel("Fire rate (Wilson 95% CI)", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

        # Mark the non-discriminating invariant
        ax.axhline(0, color="black", linewidth=0.5)

    axes[-1].legend(fontsize=9, loc="upper right")
    fig.suptitle(
        "Fig. 1 — Invariant discrimination: fire rate on perturbed vs clean data\n"
        "  (* = numeric_outliers, trace-only; bars show Wilson 95% CI)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    dest = out / "fig1_discrimination.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {dest}")


# ---------------------------------------------------------------------------
# Figure 2 — SER reduction per mode
# ---------------------------------------------------------------------------

def fig2_ser_reduction(
    mode_data: dict[str, tuple[list[list[dict]], list[list[dict]]]],
    out: Path,
) -> None:
    """Grouped bars: SER_sans vs SER_avec, coverage annotated."""
    modes = list(mode_data.keys())
    x = np.arange(len(modes))
    w = 0.28

    ser_sans_vals, ser_avec_vals, coverage_vals = [], [], []
    ser_sans_wci, ser_avec_wci = [], []

    for mode in modes:
        pert_runs, clean_runs = mode_data[mode]
        s = pooled_ser(pert_runs, clean_runs)
        ser_sans_vals.append(s["ser_sans"] if not math.isnan(s["ser_sans"]) else 0.0)
        ser_avec_vals.append(s["ser_avec"] if not math.isnan(s["ser_avec"]) else 0.0)
        coverage_vals.append(s["coverage"] if not math.isnan(s["coverage"]) else 1.0)
        wlo, whi = s["wilson_ser_sans"]
        ser_sans_wci.append((
            ser_sans_vals[-1] - (wlo if not math.isnan(wlo) else ser_sans_vals[-1]),
            (whi if not math.isnan(whi) else ser_sans_vals[-1]) - ser_sans_vals[-1],
        ))
        wlo2, whi2 = s["wilson_ser_avec"]
        ser_avec_wci.append((
            ser_avec_vals[-1] - (wlo2 if not math.isnan(wlo2) else ser_avec_vals[-1]),
            (whi2 if not math.isnan(whi2) else ser_avec_vals[-1]) - ser_avec_vals[-1],
        ))

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(
        x - w/2, ser_sans_vals, w, label="SER without invariants",
        color=_RED, alpha=0.85,
        yerr=[[v[0] for v in ser_sans_wci], [v[1] for v in ser_sans_wci]],
        capsize=4, error_kw={"elinewidth": 1.2},
    )
    bars2 = ax.bar(
        x + w/2, ser_avec_vals, w, label="SER with invariants",
        color=_GREEN, alpha=0.85,
        yerr=[[v[0] for v in ser_avec_wci], [v[1] for v in ser_avec_wci]],
        capsize=4, error_kw={"elinewidth": 1.2},
    )

    # Annotate coverage
    for i, (cov, bar) in enumerate(zip(coverage_vals, bars2)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"cov={cov:.0%}",
            ha="center", va="bottom", fontsize=8, color="dimgrey",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_MODE_LABELS.get(m, m) for m in modes], fontsize=10)
    ax.set_ylabel("Silent Error Rate (Wilson 95% CI)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Fig. 2 — SER reduction: invariants vs no invariants (pooled K runs)",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    dest = out / "fig2_ser_reduction.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {dest}")


# ---------------------------------------------------------------------------
# Figure 3 — K=3 stability for row_duplication
# ---------------------------------------------------------------------------

def fig3_stability_k3(
    pert_runs: list[list[dict]],
    clean_runs: list[list[dict]],
    out: Path,
) -> None:
    """delta and FPR for duplicate_rows: mean ± σ_K over 3 runs.

    Two panels:
      Left  — per-run delta (discriminating power) with mean ± σ highlighted
      Right — per-run FPR (false positive rate on clean) with mean ± σ
    """
    from scripts.repeat_runs import analyze_pair, aggregate

    analyses = [analyze_pair(pr, cr) for pr, cr in zip(pert_runs, clean_runs)]
    agg = aggregate(analyses)
    dr = agg["duplicate_rows"]
    ser = agg["_ser"]

    k = len(analyses)
    xs = np.arange(1, k + 1)

    # Per-run values
    deltas = [a["duplicate_rows"]["delta"] for a in analyses]
    fprs   = [a["_ser"]["fpr_clean"] for a in analyses]

    delta_m, delta_s = dr["delta"]
    fpr_m,   fpr_s   = ser["fpr_clean"]

    wlo_d, whi_d = dr["wilson_delta_approx"]
    wlo_f, whi_f = agg["_ser"]["wilson_fpr"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ---- Left: delta ----
    ax1.scatter(xs, deltas, color=_ORANGE, s=80, zorder=3, label="Per-run delta")
    ax1.axhline(delta_m, color=_RED, linewidth=1.5, linestyle="--",
                label=f"Mean = {delta_m:.3f}")
    if not math.isnan(delta_s):
        ax1.fill_between(
            [0.5, k + 0.5],
            delta_m - delta_s, delta_m + delta_s,
            color=_RED, alpha=0.12,
            label=f"± σ_K = ±{delta_s:.3f}  (a) model noise",
        )
    # Wilson CI band
    if not math.isnan(wlo_d):
        ax1.fill_between(
            [0.5, k + 0.5], wlo_d, whi_d,
            color=_ORANGE, alpha=0.10,
            label=f"Wilson 95% CI = [{wlo_d:.3f}, {whi_d:.3f}]  (b) sampling",
        )
    ax1.set_xlabel("Run index")
    ax1.set_ylabel("Δ fire rate (perturbed − clean)")
    ax1.set_title(
        "duplicate_rows — discriminating power\n(row_duplication, K=3)",
        fontsize=10, fontweight="bold",
    )
    ax1.set_xticks(xs)
    ax1.set_ylim(-0.1, 1.1)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    # ---- Right: FPR ----
    ax2.scatter(xs, fprs, color=_BLUE, s=80, zorder=3, label="Per-run FPR")
    ax2.axhline(fpr_m, color=_BLUE, linewidth=1.5, linestyle="--",
                label=f"Mean = {fpr_m:.3f}")
    if not math.isnan(fpr_s):
        ax2.fill_between(
            [0.5, k + 0.5],
            fpr_m - fpr_s, fpr_m + fpr_s,
            color=_BLUE, alpha=0.12,
            label=f"± σ_K = ±{fpr_s:.3f}  (a) model noise",
        )
    if not math.isnan(wlo_f):
        ax2.fill_between(
            [0.5, k + 0.5], wlo_f, whi_f,
            color=_BLUE, alpha=0.10,
            label=f"Wilson 95% CI = [{wlo_f:.3f}, {whi_f:.3f}]  (b) sampling",
        )
    ax2.set_xlabel("Run index")
    ax2.set_ylabel("False Positive Rate on clean data")
    ax2.set_title(
        "duplicate_rows — false positive rate\n(clean tables, K=3)",
        fontsize=10, fontweight="bold",
    )
    ax2.set_xticks(xs)
    ax2.set_ylim(-0.05, 0.3)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle(
        "Fig. 3 — K=3 stability: duplicate_rows invariant\n"
        "  (a) σ_K = model noise   (b) Wilson 95% CI = sampling uncertainty",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    dest = out / "fig3_stability_k3.png"
    fig.savefig(dest, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VERIDATA report figures")
    parser.add_argument("--row-dup-perturbed",  nargs="+", type=Path, required=True)
    parser.add_argument("--row-dup-clean",      nargs="+", type=Path, required=True)
    parser.add_argument("--locale-perturbed",   nargs="+", type=Path, required=True)
    parser.add_argument("--locale-clean",       nargs="+", type=Path, required=True)
    parser.add_argument("--outlier-perturbed",  nargs="+", type=Path, required=True)
    parser.add_argument("--outlier-clean",      nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("report/figures"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load runs
    row_dup_pert  = load_runs(args.row_dup_perturbed)
    row_dup_clean = load_runs(args.row_dup_clean)
    locale_pert   = load_runs(args.locale_perturbed)
    locale_clean  = load_runs(args.locale_clean)
    outlier_pert  = load_runs(args.outlier_perturbed)
    outlier_clean = load_runs(args.outlier_clean)

    mode_data: dict[str, tuple[list[list[dict]], list[list[dict]]]] = {
        "row_duplication":   (row_dup_pert, row_dup_clean),
        "locale_format":     (locale_pert, locale_clean),
        "outlier_injection": (outlier_pert, outlier_clean),
    }

    print("\nGenerating Fig 1 — Discrimination…")
    fig1_discrimination(mode_data, args.out)

    print("Generating Fig 2 — SER reduction…")
    fig2_ser_reduction(mode_data, args.out)

    print("Generating Fig 3 — K=3 stability…")
    fig3_stability_k3(row_dup_pert, row_dup_clean, args.out)

    print("\nDone. Figures in:", args.out)


if __name__ == "__main__":
    main()
