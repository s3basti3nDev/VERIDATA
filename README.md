# VERIDATA

Measuring the **verifiable reliability** of data analysis agents via the
**Silent Error Rate (SER)** metric.

> SER = fraction of answers that are *wrong*, *high-confidence*, and *not flagged as uncertain*.
>
> Accuracy measures how often an agent is right. SER measures how often it is
> wrong without saying so — the silent failure mode that matters most in production.

Dataset: [cardiffnlp/databench](https://huggingface.co/datasets/cardiffnlp/databench)
(~1822 questions, 80 real-world tables). Evaluation via
[databench-eval](https://pypi.org/project/databench-eval/).

---

## Quick start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate    # macOS / Linux

# 2. Install the package and dev dependencies
pip install -e ".[dev]"

# 3. Set your Anthropic API key
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# or: copy .env.example .env and fill it in, then load it

# 4. Smoke run — 10 questions, validates the pipeline cheaply
python scripts/run_baseline.py --smoke

# 5. Full baseline — 50 questions
python scripts/run_baseline.py
```

Results are written to `runs/<run_id>.jsonl` — one JSON record per question.

---

## Running tests

```powershell
# Mocked tests — no API key required, no cost
pytest

# Live integration test — costs ~1 API call
$env:VERIDATA_LIVE_TESTS = "1"
pytest
```

---

## Configuration

Edit [`configs/baseline.toml`](configs/baseline.toml) to change the model,
sample sizes, or execution limits. The model identifier is pinned exactly for
reproducibility — change it deliberately and re-run both smoke and full runs.

---

## Repository layout

```
veridata/           Python package
  agent.py          LLM agent (code-as-reasoning) + AgentResult dataclass
  config.py         Config dataclasses + TOML loader
  evaluator.py      Wrapper around databench-eval
  executor.py       Safe exec() with timeout and row-count limit
  logger.py         JSON logger + per-run JSONL writer
scripts/
  run_baseline.py   Entry point (--smoke / full)
configs/
  baseline.toml     Pinned model, temperature, sample sizes
runs/               Per-run .jsonl result files (gitignored)
tests/
  test_pipeline.py  Mocked unit tests + gated live test
```

---

## Research roadmap

| Phase | Status | Goal |
|---|---|---|
| Week 1 — Baseline | ✅ Done | Scaffold + accuracy baseline on DataBench |
| Week 2 — SER metric | ⬜ | Confidence extraction + SER computation |
| Week 3 — Perturbations | ⬜ | Robustness under controlled data noise |
| Week 4 — Verification layer | ⬜ | Abstention mechanism + cross-check layer |

See [`CLAUDE.md`](CLAUDE.md) for full architecture notes and design decisions.
