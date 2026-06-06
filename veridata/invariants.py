"""Agnostic data-quality invariants for VERIDATA Week 3.

Each invariant verifies a property of the data or generated code WITHOUT
knowing the correct answer. All are standard data-quality checks that are
justifiable independently of our perturbation scripts.

Signature: (df, code, result_str, question, **params) -> InvariantResult

Conservative fallback for dynamic column references
----------------------------------------------------
When the AST detects a subscript whose key is a Name (i.e. a variable, not a
literal string), the exact column cannot be resolved statically. Rather than
silently narrowing scope, numeric_outliers and dtype_mismatch widen to ALL
numeric / non-numeric columns and record this in the detail field:
    scope=all_columns, reason=indirection
This reflects the design principle: over-abstaining is preferable to missing
a silent error.

Four invariants
---------------
1. duplicate_rows        — exact duplicate fraction > threshold AND sensitive agg
2. numeric_outliers      — values beyond k×IQR in referenced numeric columns
3. dtype_mismatch        — column used in numeric agg has non-numeric dtype
4. unexplained_constant  — float BinOp operand absent from data and question
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class InvariantResult:
    name: str
    fired: bool
    severity: float  # [0.0, 1.0]
    detail: str


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

_SENSITIVE_ATTRS = frozenset({"sum", "count", "mean", "size", "cumsum", "nunique"})
_NUMERIC_BIN_OPS = (ast.Mult, ast.Div, ast.Add, ast.Sub)


def _parse(code: str) -> ast.AST | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _uses_sensitive_aggregation(tree: ast.AST) -> bool:
    """True if the code calls a sensitive aggregation method or len/sum."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _SENSITIVE_ATTRS:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"len", "sum"}:
                return True
    return False


def _extract_referenced_columns(
    tree: ast.AST, df_columns: frozenset[str]
) -> tuple[list[str], bool]:
    """Extract column names used as subscript keys (df['col'], df[['a','b']]).

    Returns (columns, had_dynamic_ref).
    had_dynamic_ref=True means a Name-keyed subscript was found; caller should
    widen scope to all columns.
    """
    cols: list[str] = []
    had_dynamic = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        sl = node.slice
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            if sl.value in df_columns:
                cols.append(sl.value)
        elif isinstance(sl, ast.List):
            for elt in sl.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    if elt.value in df_columns:
                        cols.append(elt.value)
        elif isinstance(sl, ast.Name):
            # Dynamic key: variable used as column selector
            had_dynamic = True

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_cols = [c for c in cols if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
    return unique_cols, had_dynamic


def _all_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# Invariant 1 — duplicate_rows
# ---------------------------------------------------------------------------

def duplicate_rows(
    df: pd.DataFrame,
    code: str,
    result_str: str,
    question: str,
    *,
    threshold: float = 0.05,
) -> InvariantResult:
    """Fires when exact-duplicate row fraction > threshold AND code uses sum/count/mean.

    Rationale: massively duplicated rows are a data-quality red flag; they
    systematically bias sum, count, and mean without affecting max, min, nunique.
    """
    dup_frac = float(df.duplicated().mean())

    if dup_frac <= threshold:
        return InvariantResult(
            "duplicate_rows", False, 0.0,
            f"dup_fraction={dup_frac:.4f} ≤ threshold={threshold}",
        )

    tree = _parse(code)
    uses_agg = _uses_sensitive_aggregation(tree) if tree is not None else True  # conservative

    if not uses_agg:
        return InvariantResult(
            "duplicate_rows", False, 0.0,
            f"dup_fraction={dup_frac:.4f} > threshold but no sensitive aggregation in code",
        )

    severity = min(1.0, dup_frac / 0.5)  # 50 % dup → max severity
    return InvariantResult(
        "duplicate_rows", True, severity,
        f"dup_fraction={dup_frac:.4f} > threshold={threshold}; sensitive aggregation detected",
    )


# ---------------------------------------------------------------------------
# Invariant 2 — numeric_outliers
# ---------------------------------------------------------------------------

def numeric_outliers(
    df: pd.DataFrame,
    code: str,
    result_str: str,
    question: str,
    *,
    k: float = 5.0,
) -> InvariantResult:
    """Fires when referenced numeric columns contain values beyond k×IQR.

    Rationale: standard statistical outlier detection. k=5 (default) is strict
    to limit false positives on real-world data that legitimately has skewed
    distributions (e.g. follower counts, revenues).

    If column resolution is ambiguous (dynamic subscript), widens to ALL numeric
    columns and records this in the detail.
    """
    df_cols = frozenset(df.columns)
    tree = _parse(code)

    if tree is None:
        cols = _all_numeric_cols(df)
        fallback = True
    else:
        cols, had_dynamic = _extract_referenced_columns(tree, df_cols)
        fallback = had_dynamic or not cols
        if fallback:
            cols = _all_numeric_cols(df)

    if not cols:
        return InvariantResult("numeric_outliers", False, 0.0, "no numeric columns to check")

    extreme: list[str] = []
    for col in cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        outliers = series[(series < lo) | (series > hi)]
        if not outliers.empty:
            sample = outliers.values[:3].tolist()
            extreme.append(f"{col}: {sample}")

    scope_tag = "scope=all_columns, reason=indirection; " if fallback else ""

    if not extreme:
        return InvariantResult(
            "numeric_outliers", False, 0.0,
            f"{scope_tag}no values beyond k={k}×IQR",
        )

    return InvariantResult(
        "numeric_outliers", True, 0.8,
        f"{scope_tag}" + "; ".join(extreme),
    )


# ---------------------------------------------------------------------------
# Invariant 3 — dtype_mismatch
# ---------------------------------------------------------------------------

def dtype_mismatch(
    df: pd.DataFrame,
    code: str,
    result_str: str,
    question: str,
) -> InvariantResult:
    """Fires when a column referenced in a numeric aggregation has a non-numeric dtype.

    Rationale: a numeric operation on an object/string column will either fail
    (noisy error) or silently produce wrong results if pandas coerces.

    If column resolution is ambiguous, widens to ALL columns and records this.
    """
    tree = _parse(code)

    # Only relevant if the code performs a numeric aggregation
    if tree is not None and not _uses_sensitive_aggregation(tree):
        return InvariantResult(
            "dtype_mismatch", False, 0.0,
            "no numeric aggregation detected in code",
        )

    df_cols = frozenset(df.columns)

    if tree is None:
        cols = list(df.columns)
        fallback = True
    else:
        cols, had_dynamic = _extract_referenced_columns(tree, df_cols)
        fallback = had_dynamic or not cols
        if fallback:
            cols = list(df.columns)

    mismatched: list[str] = []
    for col in cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            mismatched.append(f"{col}: dtype={df[col].dtype}")

    scope_tag = "scope=all_columns, reason=indirection; " if fallback else ""

    if not mismatched:
        return InvariantResult(
            "dtype_mismatch", False, 0.0,
            f"{scope_tag}all checked columns are numeric",
        )

    return InvariantResult(
        "dtype_mismatch", True, 0.9,
        f"{scope_tag}" + "; ".join(mismatched),
    )


# ---------------------------------------------------------------------------
# Invariant 4 — unexplained_constant
# ---------------------------------------------------------------------------

_DEFAULT_TRIVIALS: frozenset[float] = frozenset(
    float(v) for v in [0, 1, 2, -1, -2, 3, 4, 5, 10, 100, 0.5, -0.5, 0.0, 1.0, 2.0]
)


def unexplained_constant(
    df: pd.DataFrame,
    code: str,
    result_str: str,
    question: str,
    *,
    trivials: list | None = None,
) -> InvariantResult:
    """Fires when a float literal used as a BinOp operand (×÷+−) cannot be
    explained by the data or question text.

    Scope: only BinOp operands (multiplicative/additive factors applied to data).
    Excluded: literals inside Compare nodes (thresholds) and Subscript slices (indices).

    Rationale: a hardcoded factor that doesn't appear in the data or question
    is an unexplained external reference — a hallucinated conversion factor or
    stale constant.
    """
    triv = _DEFAULT_TRIVIALS | frozenset(float(v) for v in (trivials or []))

    tree = _parse(code)
    if tree is None:
        return InvariantResult("unexplained_constant", False, 0.0, "code unparseable")

    # Collect node IDs that live inside Compare comparators or Subscript slices
    # (these are legitimate thresholds / indices — not data factors)
    excluded_ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for comp in node.comparators:
                for n in ast.walk(comp):
                    excluded_ids.add(id(n))
            # Also exclude the left operand of the comparison itself
            for n in ast.walk(node.left):
                excluded_ids.add(id(n))
        if isinstance(node, ast.Subscript):
            for n in ast.walk(node.slice):
                excluded_ids.add(id(n))

    # Find float constants used as BinOp operands (×÷+−) not in excluded set
    candidates: list[float] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        if not isinstance(node.op, _NUMERIC_BIN_OPS):
            continue
        for operand in (node.left, node.right):
            if id(operand) in excluded_ids:
                continue
            if not isinstance(operand, ast.Constant):
                continue
            if not isinstance(operand.value, (int, float)):
                continue
            val = float(operand.value)
            if val in triv:
                continue
            candidates.append(val)

    if not candidates:
        return InvariantResult("unexplained_constant", False, 0.0, "no candidate constants")

    # Filter: remove values that appear verbatim in the question text
    q = question.lower()
    def _in_question(v: float) -> bool:
        return str(v) in q or (v == int(v) and str(int(v)) in q)

    candidates = [v for v in candidates if not _in_question(v)]

    # Filter: remove values that appear as cell values in any numeric column
    # (sample to avoid scanning huge tables)
    df_vals: set[float] = set()
    for col in df.select_dtypes(include="number").columns:
        df_vals.update(float(x) for x in df[col].dropna().head(500))

    remaining = [v for v in candidates if v not in df_vals]

    if not remaining:
        return InvariantResult(
            "unexplained_constant", False, 0.0,
            f"constants {candidates} explained by question or data",
        )

    return InvariantResult(
        "unexplained_constant", True, 0.6,
        f"unexplained float factors in BinOp: {remaining}",
    )
