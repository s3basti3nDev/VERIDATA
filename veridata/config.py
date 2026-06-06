"""Configuration dataclasses and TOML loader."""

import tomllib
from dataclasses import dataclass
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
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    execution: ExecutionConfig
    runs_dir: Path


def load_config(path: Path) -> Config:
    """Load and validate configuration from a TOML file."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return Config(
        model=ModelConfig(**raw["model"]),
        dataset=DatasetConfig(**raw["dataset"]),
        execution=ExecutionConfig(**raw["execution"]),
        runs_dir=Path(raw["paths"]["runs_dir"]),
    )
