#!/usr/bin/env python3
"""Run the VERIDATA agent on a perturbed (or clean) DataBench subset.

Reads the manifest produced by scripts/make_perturbed.py and runs the agent on
the Parquet tables saved in that manifest — no HuggingFace network call needed.

Usage
-----
    # Clean run on the same subset (uses tables_clean/)
    python scripts/run_perturbed.py --manifest runs/perturbed/<id>/manifest.jsonl --clean

    # Perturbed run (uses tables/)
    python scripts/run_perturbed.py --manifest runs/perturbed/<id>/manifest.jsonl

Running both with the SAME manifest guarantees that precision_clean and
precision_perturbed are computed on exactly the same questions.

Ground truth
------------
The ``ground_truth`` field in the manifest is the CLEAN DataBench answer.
correct_i ⟺ agent_answer ≈ ground_truth (tolerance 1e-4 for numbers)
            OR abstained_i = True  (Week 3 — always None in Week 2)

Non-determinism note
--------------------
At temperature=0, two runs on identical inputs may still differ (server-side
non-determinism observed in practice). For a stable delta headline, run K≥3
times and average. K=1 is the default; the delta is not yet statistically
stabilised with a single run.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from veridata.agent import DataAnalysisAgent
from veridata.config import load_config
from veridata.evaluator import normalized_compare
from veridata.logger import ResultsWriter, setup_logger
from veridata.metrics import compute

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.toml"


def _load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="VERIDATA perturbed/clean subset run")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest.jsonl produced by make_perturbed.py",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run on clean tables (parquet_clean_path) instead of perturbed tables",
    )
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logger("veridata.run_perturbed")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY is not set — aborting.")
        sys.exit(1)

    manifest = _load_manifest(args.manifest)
    if not manifest:
        log.error(f"Empty manifest: {args.manifest}")
        sys.exit(1)

    first = manifest[0]
    perturbation = first["perturbation"]
    severity = first["severity"]
    mode_label = "clean" if args.clean else perturbation

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"run_{mode_label}_{severity}_{run_ts}"
    results_path = _PROJECT_ROOT / cfg.runs_dir / f"{run_id}.jsonl"

    log.info(
        f"Starting run | model={cfg.model.model_id} mode={mode_label} "
        f"severity={severity} n={len(manifest)}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    agent = DataAnalysisAgent(client=client, config=cfg)
    writer = ResultsWriter(results_path)

    table_cache: dict[str, pd.DataFrame] = {}
    records: list[dict] = []

    for idx, row in enumerate(manifest, start=1):
        parquet_key = "parquet_clean_path" if args.clean else "parquet_path"
        parquet_path = Path(row[parquet_key])

        if str(parquet_path) not in table_cache:
            table_cache[str(parquet_path)] = pd.read_parquet(parquet_path)
        df = table_cache[str(parquet_path)]

        result = agent.answer(question=row["question"], df=df)

        # Ground truth = CLEAN answer stored in manifest (tolerance 1e-4 for numbers)
        correct = normalized_compare(
            result.answer,
            row["ground_truth"],
            row.get("answer_type", ""),
        )

        record: dict = {
            "run_id": run_id,
            "model_id": cfg.model.model_id,
            "temperature": cfg.model.temperature,
            "question_idx": row["question_idx"],
            "question": row["question"],
            "dataset": row["dataset"],
            "answer_type": row.get("answer_type"),
            "ground_truth": row["ground_truth"],
            "answer": result.answer,
            "correct": correct,
            "generated_code": result.generated_code,
            "confidence": result.confidence,
            "abstained": result.abstained,
            # Perturbation fields
            "perturbation": perturbation,
            "run_mode": mode_label,
            "severity": severity,
            "expected_sensitive": row.get("expected_sensitive", True),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        writer.write(record)
        records.append(record)

        log.info(
            f"[{idx}/{len(manifest)}] {row['question'][:70]!r} "
            f"→ {result.answer!r} correct={correct}"
        )

    m = compute(records)
    log.info(
        f"Done | precision={m['precision']:.4f} SER={m['SER']:.4f} "
        f"n={m['n']} run_id={run_id}"
    )
    log.info(f"Results: {results_path}")


if __name__ == "__main__":
    main()
