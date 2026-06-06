#!/usr/bin/env python3
"""Run the VERIDATA baseline evaluation against a DataBench sample.

Usage
-----
    # Smoke run — 10 questions, validates the pipeline cheaply
    python scripts/run_baseline.py --smoke

    # Full baseline — 50 questions
    python scripts/run_baseline.py
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import anthropic

# Resolve project root regardless of the working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from veridata.agent import DataAnalysisAgent
from veridata.config import load_config
from veridata.evaluator import BaselineEvaluator, normalized_compare
from veridata.logger import ResultsWriter, setup_logger

_CONFIG_PATH = _PROJECT_ROOT / "configs" / "baseline.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description="VERIDATA baseline evaluation")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on smoke_size questions instead of sample_size (pipeline check)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_CONFIG_PATH,
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    log = setup_logger("veridata.run")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        log.error("ANTHROPIC_API_KEY environment variable is not set — aborting.")
        sys.exit(1)

    sample_size = cfg.dataset.smoke_size if args.smoke else cfg.dataset.sample_size
    run_mode = "smoke" if args.smoke else "full"

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_mode}_{run_ts}"
    results_path = _PROJECT_ROOT / cfg.runs_dir / f"{run_id}.jsonl"

    log.info(
        f"Starting {run_mode} run | model={cfg.model.model_id} "
        f"temperature={cfg.model.temperature} n={sample_size}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    agent = DataAnalysisAgent(client=client, config=cfg)
    evaluator = BaselineEvaluator(name=cfg.dataset.name, split=cfg.dataset.split)
    writer = ResultsWriter(results_path)

    sample = evaluator.load_sample(n=sample_size)
    responses: list[str] = []

    for idx, row in enumerate(sample, start=1):
        table_df = evaluator.load_table_for(row)
        result = agent.answer(question=row["question"], df=table_df)

        # Compute verdict here — same comparator as score(), so JSONL is
        # always consistent with the final accuracy figure.
        correct = normalized_compare(
            result.answer,
            str(row.get("answer", "")),
            str(row.get("type", "")),
        )

        # JSONL schema — includes Week-2 reserved fields at None.
        record: dict = {
            "run_id": run_id,
            "model_id": cfg.model.model_id,
            "temperature": cfg.model.temperature,
            "question_idx": idx - 1,   # 0-based index in the ordered dataset
            "question": row["question"],
            "dataset": row["dataset"],
            "answer_type": row.get("type"),
            "ground_truth": row.get("answer"),
            "answer": result.answer,
            "correct": correct,                  # auditable per-record verdict
            "generated_code": result.generated_code,
            "confidence": result.confidence,     # None — reserved for SER (Week 2)
            "abstained": result.abstained,       # None — reserved for SER (Week 2)
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        writer.write(record)
        responses.append(result.answer)

        log.info(
            f"[{idx}/{sample_size}] {row['question'][:70]!r} → {result.answer!r}"
        )

    accuracy = evaluator.score(responses, sample)
    log.info(
        f"Run complete | accuracy={accuracy:.4f} mode={run_mode} n={sample_size} "
        f"run_id={run_id}"
    )
    log.info(f"Results: {results_path}")


if __name__ == "__main__":
    main()
