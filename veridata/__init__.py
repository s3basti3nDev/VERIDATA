"""VERIDATA — verifiable reliability measurement for data analysis agents."""

from .agent import AgentResult, DataAnalysisAgent
from .config import Config, load_config

__all__ = ["AgentResult", "DataAnalysisAgent", "Config", "load_config"]
__version__ = "0.1.0"
