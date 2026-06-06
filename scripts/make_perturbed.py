#!/usr/bin/env python3
"""Generate a perturbed DataBench subset for VERIDATA Week 2 experiments.

Saves BOTH the clean and perturbed tables as Parquet so that run_perturbed.py
can run both a clean and a perturbed pass on exactly the same questions
without any additional network calls.

Usage
-----
    python scripts/make_perturbed.py --mode row_duplication --severity 0.3 --n 30
    python scripts/make_perturbed.py --mode locale_format --n 30
    python scripts/make_perturbed.py --mode outlier_injection --severity 0.3 --n 30

Output
------
    runs/perturbed/<run_id>/
        manifest.jsonl              one line per question
        tables/          <dataset>.parquet   perturbed tables
        tables_clean/    <dataset>.parquet   original clean tables

Ground truth in the manifest = CLEAN answer (from DataBench).
Both sets of tables are saved once and reused by run_perturbed.py.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from veridata.config import load_config
from veridata.evaluator import BaselineEvaluator
from veridata.perturbations import (
    expected_sensitive,
    locale_format,
    outlier_injection,
    row_duplication,
)

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.toml"


def _first_numeric_columns(df: pd.DataFrame, max_cols: int = 3) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:max_cols]


def _apply_perturbation(
    df: pd.DataFrame,
    mode: str,
    severity: float,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    if mode == "row_duplication":
        return row_duplication(df, dup_fraction=severity, seed=seed)

    if mode == "locale_format":
        cols = _first_numeric_columns(df)
        if not cols:
            return df.copy(), {"perturbation": "locale_format", "columns": []}
        return locale_format(df, columns=cols, seed=seed)

    if mode == "outlier_injection":
        cols = _first_numeric_columns(df, max_cols=1)
        if not cols:
            return df.copy(), {"perturbation": "outlier_injection", "cells_modified": 0}
        n_outliers = max(1, int(round(len(df) * severity)))
        return outlier_injection(df, column=cols[0], n_outliers=n_outliers, seed=seed)

    raise ValueError(f"Unknown perturbation mode: {mode!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a perturbed DataBench subset"
    )
    parser.add_argument(
        "--mode",
        choices=["row_duplication", "locale_format", "outlier_injection"],
        required=True,
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.3,
        help=(
            "dup_fraction for row_duplication; "
            "fraction of rows to corrupt for outlier_injection; "
            "ignored for locale_format"
        ),
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="Number of number-type questions to include",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluator = BaselineEvaluator(name=cfg.dataset.name, split=cfg.dataset.split)

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"perturbed_{args.mode}_{args.severity}_{run_ts}"
    out_dir = _PROJECT_ROOT / cfg.runs_dir / "perturbed" / run_id
    tables_dir = out_dir / "tables"
    tables_clean_dir = out_dir / "tables_clean"
    tables_dir.mkdir(parents=True, exist_ok=True)
    tables_clean_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DataBench QA — filtering to answer_type=number, n={args.n} …")
    # Load a generous pool (full sample) to find enough number-type questions
    all_qa = evaluator.load_sample(n=min(cfg.dataset.sample_size * 4, 200))
    number_qa = [row for row in all_qa if str(row.get("type", "")) == "number"]
    sample = number_qa[: args.n]

    if len(sample) < args.n:
        print(f"  Warning: only {len(sample)} number-type questions found (requested {args.n})")

    # Process each unique table once (same seed → same perturbation)
    seen_datasets: set[str] = set()
    manifest_path = out_dir / "manifest.jsonl"

    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, row in enumerate(sample):
            dataset_id: str = row["dataset"]

            if dataset_id not in seen_datasets:
                clean_df = evaluator.load_table_for(row)
                perturbed_df, pert_meta = _apply_perturbation(
                    clean_df, args.mode, args.severity, args.seed
                )
                clean_df.to_parquet(tables_clean_dir / f"{dataset_id}.parquet", index=False)
                perturbed_df.to_parquet(tables_dir / f"{dataset_id}.parquet", index=False)
                seen_datasets.add(dataset_id)

            parquet_clean = str(tables_clean_dir / f"{dataset_id}.parquet")
            parquet_perturbed = str(tables_dir / f"{dataset_id}.parquet")

            record = {
                "run_id": run_id,
                "question_idx": idx,
                "question": row["question"],
                "dataset": dataset_id,
                "answer_type": row.get("type"),
                "ground_truth": str(row.get("answer", "")),
                "perturbation": args.mode,
                "severity": args.severity,
                "seed": args.seed,
                # expected_sensitive is determined by the QUESTION + mode, not by the data
                "expected_sensitive": expected_sensitive(row["question"], args.mode),
                "parquet_path": parquet_perturbed,
                "parquet_clean_path": parquet_clean,
            }
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"  [{idx + 1}/{len(sample)}] {dataset_id} "
                f"sensitive={record['expected_sensitive']} "
                f"— {row['question'][:55]!r}"
            )

    print(f"\nDone.")
    print(f"  Manifest : {manifest_path}")
    print(f"  Tables   : {tables_dir}  (perturbed)")
    print(f"  Tables   : {tables_clean_dir}  (clean)")
    print(f"  Run ID   : {run_id}")
    print()
    print("Next steps:")
    print(f"  # Clean run (same 30 questions, original data)")
    print(f"  python scripts/run_perturbed.py --manifest {manifest_path} --clean")
    print(f"  # Perturbed run")
    print(f"  python scripts/run_perturbed.py --manifest {manifest_path}")


if __name__ == "__main__":
    main()
