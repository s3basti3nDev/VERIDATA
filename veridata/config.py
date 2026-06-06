"""Configuration dataclasses and TOML loader."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    model_id: str
    temperature: float
    max_tokens: int


@dataclass
class DatasetConfig:
    name: str        # databench-eval dataset name, e.g. "semeval"
    split: str       # dataset split, e.g. "dev"
    sample_size: int
    smoke_size: int


@dataclass
class ExecutionConfig:
    timeout_seconds: int
    max_rows: int


@dataclass
class InvariantsConfig:
    # duplicate_rows: fraction of exact-duplicate rows to trigger
    duplicate_row_threshold: float = 0.05
    # numeric_outliers: k in Q1 - k*IQR / Q3 + k*IQR; strict default to limit false positives
    outlier_iqr_factor: float = 5.0
    # unexplained_constant: literals excluded from scrutiny (trivial control values)
    trivial_constants: list = field(default_factory=lambda: [0, 1, 2, -1, 100, 0.5])


@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    execution: ExecutionConfig
    runs_dir: Path
    invariants: InvariantsConfig = field(default_factory=InvariantsConfig)


def load_config(path: Path) -> Config:
    """Load and validate configuration from a TOML file."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    raw_inv = raw.get("invariants", {})
    return Config(
        model=ModelConfig(**raw["model"]),
        dataset=DatasetConfig(**raw["dataset"]),
        execution=ExecutionConfig(**raw["execution"]),
        runs_dir=Path(raw["paths"]["runs_dir"]),
        invariants=InvariantsConfig(**raw_inv) if raw_inv else InvariantsConfig(),
    )
