"""LLM-powered data analysis agent using code-as-reasoning.

The agent asks the model to write pandas code, executes it via executor.py,
and returns a structured result. The ``confidence`` and ``abstained`` fields
are reserved stubs for the SER metric (Week 2).
"""

import re
from dataclasses import dataclass
from typing import Optional

import anthropic
import pandas as pd

from .config import Config
from .executor import execute_code

_SYSTEM_PROMPT = """\
You are a data analysis assistant. Given a pandas DataFrame `df` and a question,
write minimal Python/pandas code to compute the answer.

Rules:
- Use `df` as the DataFrame variable name (already loaded).
- Store the final answer in a variable called `result`.
- `result` must be a scalar: int, float, str, or bool.
- `pd` is available as pandas; do not import anything else.
- Output ONLY the code — no markdown fences, no comments, no explanations.\
"""


@dataclass
class AgentResult:
    """Structured result returned by the agent for each question.

    Fields ``confidence`` and ``abstained`` are reserved for Week 2 (SER metric)
    and intentionally left as None in the Week 1 baseline.
    """

    answer: str
    generated_code: str
    raw_response: str
    confidence: Optional[float] = None   # reserved — SER (Week 2)
    abstained: Optional[bool] = None     # reserved — SER (Week 2)


class DataAnalysisAgent:
    """Calls the Anthropic API to generate and execute pandas code."""

    def __init__(self, client: anthropic.Anthropic, config: Config) -> None:
        self._client = client
        self._cfg = config

    def answer(self, question: str, df: pd.DataFrame) -> AgentResult:
        """Answer a question about ``df`` using LLM-generated code."""
        schema_desc = _describe_schema(df)
        user_msg = f"DataFrame schema:\n{schema_desc}\n\nQuestion: {question}"

        response = self._client.messages.create(
            model=self._cfg.model.model_id,
            max_tokens=self._cfg.model.max_tokens,
            temperature=self._cfg.model.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )

        raw = response.content[0].text
        code = _extract_code(raw)

        value, error = execute_code(
            code=code,
            df=df,
            timeout=self._cfg.execution.timeout_seconds,
            max_rows=self._cfg.execution.max_rows,
        )

        answer_str = str(value) if value is not None else f"ERROR: {error}"

        return AgentResult(
            answer=answer_str,
            generated_code=code,
            raw_response=raw,
        )


def _describe_schema(df: pd.DataFrame) -> str:
    """Compact schema: shape + column name / dtype / one sample value."""
    lines = [f"shape: {df.shape[0]} rows × {df.shape[1]} cols"]
    for col, dtype in df.dtypes.items():
        non_null = df[col].dropna()
        sample = repr(non_null.iloc[0]) if not non_null.empty else "N/A"
        lines.append(f"  {col} ({dtype}): e.g. {sample}")
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    """Extract executable Python code from an LLM response.

    Strategy:
    1. If a fenced block (```python or ```) exists anywhere in the text,
       return its content and ignore everything outside the fence.
    2. Otherwise, find the first ``result =`` line. Walk backwards and drop
       any preceding line that does not compile as Python — those are natural-
       language preambles inserted by the model despite the prompt instruction.
    """
    text = text.strip()

    # 1. Extract from the first fenced code block found anywhere in the text.
    fence_match = re.search(r"```(?:python)?\s*\n(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. No fence: locate first `result =` and strip non-code lines above it.
    lines = text.splitlines()
    result_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"\s*result\s*=", ln)),
        None,
    )
    if result_idx is None:
        return text  # no result= found; return as-is and let executor report the error

    start = result_idx
    for i in range(result_idx - 1, -1, -1):
        if _is_python_line(lines[i]):
            start = i
        else:
            break  # first non-code line going up → everything above is prose

    return "\n".join(lines[start:]).strip()


def _is_python_line(line: str) -> bool:
    """Return True if the line is empty, a comment, or compiles as Python."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    try:
        compile(stripped, "<check>", "exec")
        return True
    except SyntaxError:
        return False
