"""Pipeline tests.

Mocked tests (TestExecutor, TestAgentMocked, TestNormalizedCompare) run without
any network calls and cost nothing. Live tests (TestAgentLive) require a real
API key and are gated behind VERIDATA_LIVE_TESTS=1.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from veridata.agent import AgentResult, DataAnalysisAgent
from veridata.config import load_config
from veridata.evaluator import _compare_list_category, _compare_number, normalized_compare
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
# Normalized comparator unit tests
# ---------------------------------------------------------------------------

class TestNormalizedCompare:
    # --- number ---
    def test_number_exact_int(self):
        assert _compare_number("42", "42") is True

    def test_number_float_vs_int(self):
        assert _compare_number("42.0", "42") is True

    def test_number_within_tolerance(self):
        # 1234.5699 vs 1234.5700 — relative diff ≈ 8e-8 < 1e-4
        assert _compare_number("1234.5699", "1234.5700") is True

    def test_number_outside_tolerance(self):
        # 100 vs 101 — relative diff = 1% >> 1e-4
        assert _compare_number("100", "101") is False

    def test_number_zero_truth_abs_fallback(self):
        assert _compare_number("0.0", "0") is True
        assert _compare_number("0.001", "0") is False

    def test_number_negative(self):
        assert _compare_number("-42.0", "-42") is True

    def test_number_bad_value_returns_false(self):
        assert normalized_compare("not_a_number", "42", "number") is False

    # --- list[category] ---
    def test_list_category_same_order(self):
        assert _compare_list_category("apple, banana", "apple, banana") is True

    def test_list_category_different_order(self):
        assert _compare_list_category("banana, apple", "apple, banana") is True

    def test_list_category_case_insensitive(self):
        assert _compare_list_category("Apple, Banana", "apple, banana") is True

    def test_list_category_extra_spaces(self):
        assert _compare_list_category("apple ,  banana", "apple, banana") is True

    def test_list_category_different_sets(self):
        assert _compare_list_category("apple, cherry", "apple, banana") is False

    def test_list_category_different_lengths(self):
        assert _compare_list_category("apple, banana, cherry", "apple, banana") is False

    # --- other types use case-insensitive exact match ---
    def test_boolean_case_insensitive(self):
        assert normalized_compare("True", "true", "boolean") is True
        assert normalized_compare("False", "true", "boolean") is False

    def test_category_strip(self):
        assert normalized_compare("  Paris  ", "Paris", "category") is True

    def test_unknown_semantic_falls_back_to_str(self):
        assert normalized_compare("hello", "hello", "list[number]") is True
        assert normalized_compare("hello", "world", "list[number]") is False


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
