#!/usr/bin/env python3
"""Run the VERIDATA agent with invariant-based verification on a perturbed subset.

Reads the manifest produced by make_perturbed.py, runs the agent, then applies
all invariants via run_invariants(). Writes an extended JSONL that includes:
  - abstained / confidence (now non-null)
  - invariants: full audit trace (list of InvariantResult dicts)

Usage
-----
    # Verified run on perturbed tables
    python scripts/run_verified.py --manifest runs/perturbed/<id>/manifest.jsonl

    # Verified run on CLEAN tables (measures false-positive rate)
    python scripts/run_verified.py --manifest runs/perturbed/<id>/manifest.jsonl --clean

Correctness predicate
---------------------
    correct_i  =  normalized_compare(answer, ground_truth, type)   [factual]
    SER        =  mean(not correct AND not abstained)               [system metric]

The SER with invariants is lower than SER without because abstentions on wrong
answers are no longer counted as silent errors.
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
from veridata.verifier import run_invariants, verification_to_dict

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.toml"


def _load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="VERIDATA verified run")
    parser.add_argument(
        "--manifest", type=Path, required=True,
        help="manifest.jsonl from make_perturbed.py",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Use clean tables (measures false-positive rate of invariants)",
    )
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logger("veridata.run_verified")

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
    run_id = f"verified_{mode_label}_{severity}_{run_ts}"
    results_path = _PROJECT_ROOT / cfg.runs_dir / f"{run_id}.jsonl"

    log.info(
        f"Starting verified run | model={cfg.model.model_id} "
        f"mode={mode_label} n={len(manifest)}"
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

        try:
            result = agent.answer(question=row["question"], df=df)
        except Exception as exc:
            log.warning(f"[{idx}/{len(manifest)}] API error — skipping: {exc}")
            records.append({"correct": False, "abstained": True, "confidence": 0.0})
            continue

        # Invariant verification layer
        verification = run_invariants(
            df=df,
            code=result.generated_code,
            result_str=result.answer,
            question=row["question"],
            config=cfg,
        )

        # Factual correctness (ground truth = clean DataBench answer)
        correct = normalized_compare(
            result.answer,
            row["ground_truth"],
            row.get("answer_type", ""),
        )

        vdict = verification_to_dict(verification)
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
            # Verification results (now non-null)
            "abstained": vdict["abstained"],
            "confidence": vdict["confidence"],
            "invariants": vdict["invariants"],
            # Perturbation context
            "perturbation": perturbation,
            "run_mode": mode_label,
            "severity": severity,
            "expected_sensitive": row.get("expected_sensitive", True),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        writer.write(record)
        records.append(record)

        fired_names = [r["name"] for r in vdict["invariants"] if r["fired"]]
        log.info(
            f"[{idx}/{len(manifest)}] {row['question'][:60]!r} "
            f"→ {result.answer!r} correct={correct} abstained={vdict['abstained']}"
            + (f" fired={fired_names}" if fired_names else "")
        )

    m = compute(records)
    log.info(
        f"Done | precision={m['precision']:.4f} SER={m['SER']:.4f} "
        f"coverage={m['coverage']:.4f} sur_abstention={m['sur_abstention']:.4f} "
        f"n={m['n']} run_id={run_id}"
    )
    log.info(f"Results: {results_path}")


if __name__ == "__main__":
    main()
