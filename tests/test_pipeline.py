"""Pipeline tests.

Mocked tests (TestExecutor, TestAgentMocked) run without any network calls and
cost nothing. Live tests (TestAgentLive) require a real API key and are gated
behind VERIDATA_LIVE_TESTS=1 so they never run in CI by accident.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from veridata.agent import AgentResult, DataAnalysisAgent
from veridata.config import load_config
from veridata.executor import execute_code

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "baseline.toml"


# ---------------------------------------------------------------------------
# Executor unit tests
# ---------------------------------------------------------------------------

class TestExecutor:
    def test_simple_column_sum(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, error = execute_code("result = df['a'].sum()", df)
        assert error is None
        assert int(result) == 6

    def test_row_limit_is_applied(self):
        df = pd.DataFrame({"a": range(1_000)})
        result, error = execute_code("result = len(df)", df, max_rows=10)
        assert error is None
        assert result == 10

    def test_bad_column_returns_error_not_exception(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, error = execute_code("result = df['nonexistent'].sum()", df)
        assert result is None
        assert error is not None
        assert "KeyError" in error

    def test_syntax_error_returns_error_not_exception(self):
        df = pd.DataFrame({"a": [1]})
        result, error = execute_code("result = !!!invalid!!!", df)
        assert result is None
        assert error is not None

    def test_timeout_is_enforced(self):
        """A computation that takes ~10 s is cut off at timeout=1."""
        df = pd.DataFrame({"a": [1]})
        result, error = execute_code(
            # sum over 10^9 items reliably takes > 1 s
            "result = sum(i * i for i in range(10 ** 9))",
            df,
            timeout=1,
        )
        assert result is None
        assert error is not None
        assert "TimeoutError" in error

    def test_import_blocked(self):
        """Generated code cannot import modules (no __import__ in builtins)."""
        df = pd.DataFrame({"a": [1]})
        result, error = execute_code("import os; result = os.getcwd()", df)
        assert result is None
        assert error is not None


# ---------------------------------------------------------------------------
# Agent mocked tests — no network, no cost
# ---------------------------------------------------------------------------

class TestAgentMocked:
    def _agent(self, generated_code: str) -> DataAnalysisAgent:
        cfg = load_config(_CONFIG_PATH)
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=generated_code)]
        mock_client.messages.create.return_value = mock_response
        return DataAnalysisAgent(client=mock_client, config=cfg)

    def test_correct_answer_end_to_end(self):
        agent = self._agent("result = df['value'].max()")
        df = pd.DataFrame({"value": [10, 50, 30]})
        r = agent.answer("What is the max?", df)
        assert isinstance(r, AgentResult)
        assert r.answer == "50"

    def test_execution_error_captured_in_answer(self):
        """Errors in generated code appear in answer, not as raised exceptions."""
        agent = self._agent("result = df['missing_column'].sum()")
        df = pd.DataFrame({"value": [1, 2]})
        r = agent.answer("Sum of missing column?", df)
        assert r.answer.startswith("ERROR:")

    def test_result_schema_is_ser_compatible(self):
        """AgentResult carries reserved SER fields from day one."""
        r = AgentResult(answer="42", generated_code="result=42", raw_response="result=42")
        assert hasattr(r, "confidence")
        assert hasattr(r, "abstained")
        assert r.confidence is None
        assert r.abstained is None

    def test_markdown_fences_stripped(self):
        """Model sometimes wraps code in fences despite the prompt instructions."""
        agent = self._agent("```python\nresult = df['a'].sum()\n```")
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = agent.answer("Total?", df)
        assert r.answer == "6"
        assert "```" not in r.generated_code

    def test_generated_code_and_raw_response_stored(self):
        code = "result = len(df)"
        agent = self._agent(code)
        df = pd.DataFrame({"x": [1, 2]})
        r = agent.answer("How many rows?", df)
        assert r.generated_code == code
        assert r.raw_response == code


# ---------------------------------------------------------------------------
# Live integration tests — requires VERIDATA_LIVE_TESTS=1 + ANTHROPIC_API_KEY
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("VERIDATA_LIVE_TESTS"),
    reason="Set VERIDATA_LIVE_TESTS=1 to run live tests (requires ANTHROPIC_API_KEY, costs tokens)",
)
class TestAgentLive:
    def test_single_arithmetic_question(self):
        import anthropic as _anthropic

        cfg = load_config(_CONFIG_PATH)
        client = _anthropic.Anthropic()
        agent = DataAnalysisAgent(client=client, config=cfg)

        df = pd.DataFrame({"score": [10, 20, 30, 40, 50]})
        r = agent.answer("What is the maximum score?", df)

        assert r.answer == "50"
        assert r.confidence is None    # reserved field present but empty
        assert r.abstained is None
        assert len(r.generated_code) > 0
