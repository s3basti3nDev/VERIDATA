"""Pipeline tests.

Mocked tests (TestExecutor, TestAgentMocked, TestNormalizedCompare,
TestMetrics, TestPerturbations) run without any network calls and cost nothing.
Live tests (TestAgentLive) require a real API key and are gated behind
VERIDATA_LIVE_TESTS=1.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from veridata.agent import AgentResult, DataAnalysisAgent
from veridata.config import load_config
from veridata.evaluator import (
    BaselineEvaluator,
    _compare_list_category,
    _compare_number,
    normalized_compare,
)
from veridata.executor import execute_code
from veridata.metrics import compare, compute
from veridata.perturbations import (
    expected_sensitive,
    locale_format,
    outlier_injection,
    row_duplication,
)

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

    def test_natural_language_prefix_stripped(self):
        """q02 pattern: model outputs a sentence before the code."""
        raw = "Sure, here's the code to answer your question:\nresult = df['a'].sum()"
        agent = self._agent(raw)
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = agent.answer("Total?", df)
        assert r.answer == "6"
        assert "Sure" not in r.generated_code

    def test_fence_anywhere_in_response(self):
        """Fence buried after an explanation preamble — extract only the block."""
        raw = (
            "I'll compute this step by step.\n\n"
            "```python\n"
            "result = df['value'].max()\n"
            "```\n\n"
            "This gives the maximum value."
        )
        agent = self._agent(raw)
        df = pd.DataFrame({"value": [10, 50, 30]})
        r = agent.answer("Max?", df)
        assert r.answer == "50"
        assert "I'll" not in r.generated_code
        assert "This gives" not in r.generated_code

    def test_intermediate_variable_preserved(self):
        """Setup code before result= must be kept when it compiles cleanly."""
        raw = "Here is the solution:\nfiltered = df[df['a'] > 1]\nresult = filtered['a'].sum()"
        agent = self._agent(raw)
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = agent.answer("Sum of values > 1?", df)
        assert r.answer == "5"   # 2 + 3

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

    def test_list_category_different_order_is_wrong(self):
        # DataBench uses ordered comparison — position matters
        assert _compare_list_category("banana, apple", "apple, banana") is False

    def test_list_category_case_insensitive(self):
        assert _compare_list_category("Apple, Banana", "apple, banana") is True

    def test_list_category_extra_spaces(self):
        assert _compare_list_category("apple ,  banana", "apple, banana") is True

    def test_list_category_different_sets(self):
        assert _compare_list_category("apple, cherry", "apple, banana") is False

    def test_list_category_different_lengths(self):
        assert _compare_list_category("apple, banana, cherry", "apple, banana") is False

    def test_list_category_literal_strips_brackets_and_quotes(self):
        assert _compare_list_category("['apple','banana']", "apple, banana") is True

    # --- list[category] parametrized: literal vs CSV — these FAIL on current impl ---
    @pytest.mark.parametrize("value,truth", [
        # q15 — exact case from run output
        ("['reply','original']",              "reply, original"),
        # q32–q35 style: list literal vs CSV, same content, same order
        ("['es','es','es','es','es']",        "es, es, es, es, es"),
        ("['paris','london','berlin']",       "paris, london, berlin"),
        ("['foo','bar','baz']",               "foo, bar, baz"),
        ("['alpha','beta','gamma','delta']",  "alpha, beta, gamma, delta"),
    ])
    def test_list_category_literal_vs_csv(self, value, truth):
        """List literal and CSV with identical content must compare as equal.

        All five cases fail with the set-based implementation (brackets and
        quotes are not stripped before the split, so token shapes differ).
        They must all pass after the _parse_list fix.
        """
        assert normalized_compare(value, truth, "list[category]") is True

    # --- list[number] ---
    def test_list_number_exact(self):
        assert normalized_compare("1, 2, 3", "1, 2, 3", "list[number]") is True

    def test_list_number_literal_vs_csv(self):
        assert normalized_compare("['1.0','2.0','3.0']", "1, 2, 3", "list[number]") is True

    def test_list_number_tolerance(self):
        assert normalized_compare("1.00009, 2.0", "1.0, 2.0", "list[number]") is True

    def test_list_number_different_order_is_wrong(self):
        assert normalized_compare("2, 1", "1, 2", "list[number]") is False

    def test_list_number_different_lengths(self):
        assert normalized_compare("1, 2, 3", "1, 2", "list[number]") is False

    # --- other types use case-insensitive exact match ---
    def test_boolean_case_insensitive(self):
        assert normalized_compare("True", "true", "boolean") is True
        assert normalized_compare("False", "true", "boolean") is False

    def test_category_strip(self):
        assert normalized_compare("  Paris  ", "Paris", "category") is True

    def test_unknown_semantic_falls_back_to_str(self):
        # list[number] is now a known handler; use a truly unknown type
        assert normalized_compare("hello", "hello", "set[category]") is True
        assert normalized_compare("hello", "world", "set[category]") is False

    # --- score() alignment test ---
    def test_score_three_correct_one_wrong(self):
        """Core alignment test: score() must return 0.75 for 3/4 correct pairs.

        The list[category] response uses list-literal format vs CSV truth,
        same order — tests both alignment and the _parse_list normalisation.
        The number case is deliberately >1e-4 off to be the single wrong answer.
        """
        sample = [
            {"answer": "42",            "type": "number"},         # ✓ float vs int
            {"answer": "Paris",         "type": "category"},       # ✓ case-insensitive
            {"answer": "apple, banana", "type": "list[category]"}, # ✓ literal vs CSV, same order
            {"answer": "100",           "type": "number"},         # ✗ 1 % off → False
        ]
        responses = ["42.0", "paris", "['apple','banana']", "101"]

        evaluator = BaselineEvaluator.__new__(BaselineEvaluator)  # skip __init__ / HF download
        acc = evaluator.score(responses, sample)
        assert acc == pytest.approx(0.75)

    def test_score_raises_on_alignment_mismatch(self):
        """score() must refuse mismatched lengths — alignment guard."""
        evaluator = BaselineEvaluator.__new__(BaselineEvaluator)
        with pytest.raises(ValueError, match="Alignment mismatch"):
            evaluator.score(["a", "b"], [{"answer": "a", "type": "category"}])


# ---------------------------------------------------------------------------
# Metrics unit tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def _rec(self, correct: bool, abstained=None) -> dict:
        return {"correct": correct, "abstained": abstained, "confidence": None}

    def test_ser_equals_one_minus_precision_when_no_abstention(self):
        """Week 2 invariant: abstained=None everywhere → SER = 1 − precision."""
        records = [
            self._rec(True),
            self._rec(True),
            self._rec(False),
            self._rec(False),
        ]
        m = compute(records)
        assert m["precision"] == pytest.approx(0.5)
        assert m["SER"] == pytest.approx(0.5)
        assert m["coverage"] == pytest.approx(1.0)
        assert m["SER"] == pytest.approx(1.0 - m["precision"])

    def test_abstained_reduces_SER(self):
        """An abstained record is not counted as a silent error."""
        records = [
            self._rec(False, abstained=True),   # abstained → not a silent error
            self._rec(False, abstained=None),    # wrong + not abstained → silent error
        ]
        m = compute(records)
        assert m["SER"] == pytest.approx(0.5)    # only 1/2 is a silent error
        assert m["coverage"] == pytest.approx(0.5)

    def test_empty_records_returns_zeros(self):
        m = compute([])
        assert m["n"] == 0
        assert m["precision"] == 0.0

    def test_compare_clean_vs_perturbed(self):
        clean = [self._rec(True)] * 4
        perturbed = [self._rec(True)] * 2 + [self._rec(False)] * 2
        result = compare(clean, perturbed)
        assert result["precision_clean"] == pytest.approx(1.0)
        assert result["precision_perturbed"] == pytest.approx(0.5)
        assert result["delta_precision"] == pytest.approx(-0.5)
        assert result["SER_clean"] == pytest.approx(0.0)
        assert result["SER_perturbed"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Perturbation unit tests
# ---------------------------------------------------------------------------

class TestPerturbations:
    def test_row_duplication_changes_sum(self):
        """Sum on perturbed table differs from clean; clean ground truth is preserved."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        clean_sum = df["value"].sum()
        df_pert, meta = row_duplication(df, dup_fraction=0.4, seed=42)
        assert df_pert["value"].sum() != clean_sum
        assert meta["n_rows_after"] > meta["n_rows_before"]

    def test_row_duplication_does_not_change_max(self):
        """Max is an order-statistic — unaffected by row duplication."""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        clean_max = df["value"].max()
        df_pert, _ = row_duplication(df, dup_fraction=0.5, seed=42)
        assert df_pert["value"].max() == clean_max

    def test_locale_format_breaks_naive_parse(self):
        """After locale_format, column is string-like and direct float() fails."""
        df = pd.DataFrame({"price": [1234.56, 7890.12]})
        df_pert, meta = locale_format(df, columns=["price"])
        # dtype becomes string-like (object or StringDtype depending on pandas version)
        assert not pd.api.types.is_numeric_dtype(df_pert["price"])
        with pytest.raises((ValueError, TypeError)):
            float(df_pert["price"].iloc[0])
        assert "price" in meta["columns"]

    def test_locale_format_output_shape(self):
        """French format: comma decimal, space thousands."""
        df = pd.DataFrame({"v": [1234.5]})
        df_pert, _ = locale_format(df, columns=["v"])
        val = df_pert["v"].iloc[0]
        assert "," in val         # decimal separator
        assert "." not in val     # no US decimal point

    def test_outlier_injection_skews_mean(self):
        """Mean changes after injection; count (len) does not."""
        df = pd.DataFrame({"v": list(range(20))})
        clean_mean = df["v"].mean()
        df_pert, meta = outlier_injection(df, column="v", n_outliers=3, magnitude=50.0)
        assert df_pert["v"].mean() != pytest.approx(clean_mean, rel=0.01)
        assert len(df_pert) == len(df)
        assert meta["cells_modified"] == 3

    def test_perturbation_deterministic(self):
        """Same seed → identical output."""
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        out1, _ = row_duplication(df, dup_fraction=0.3, seed=7)
        out2, _ = row_duplication(df, dup_fraction=0.3, seed=7)
        pd.testing.assert_frame_equal(out1, out2)

    # expected_sensitive labels
    def test_expected_sensitive_row_dup_max_question_is_false(self):
        assert expected_sensitive("What is the maximum revenue?", "row_duplication") is False

    def test_expected_sensitive_row_dup_min_question_is_false(self):
        assert expected_sensitive("What is the minimum score?", "row_duplication") is False

    def test_expected_sensitive_row_dup_sum_question_is_true(self):
        assert expected_sensitive("What is the total revenue?", "row_duplication") is True

    def test_expected_sensitive_locale_always_true(self):
        assert expected_sensitive("What is the maximum value?", "locale_format") is True

    def test_expected_sensitive_outlier_count_is_false(self):
        assert expected_sensitive("How many rows have a positive value?", "outlier_injection") is False

    def test_expected_sensitive_outlier_mean_is_true(self):
        assert expected_sensitive("What is the average salary?", "outlier_injection") is True


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
