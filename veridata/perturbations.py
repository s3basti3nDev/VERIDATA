"""Controlled perturbation library for VERIDATA.

Each perturbation is a pure function:
    perturb(df, **params) -> (df_perturbed: DataFrame, metadata: dict)

The ground truth is NOT recomputed — it stays the clean answer.
Perturbations corrupt data to induce agent errors while keeping the
true answer known.

Three modes
-----------
row_duplication  : duplicates a random fraction of rows then shuffles.
                   Biases sum, count, mean. Does NOT affect max/min/nunique.
locale_format    : converts numeric column(s) to French-locale strings.
                   dtype becomes object; naive float() or pandas agg fails.
outlier_injection: replaces n cells with extreme values (mean ± k·std).
                   Biases mean, sum, max. Does NOT affect count.

expected_sensitive
------------------
Whether a perturbation is "sensitive" for a given question depends on the
QUESTION (what aggregation is asked), not on the data alone.
Use the ``expected_sensitive()`` function to determine this at the question level.
The perturbation functions themselves only return factual metadata about what
was done to the DataFrame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def row_duplication(
    df: pd.DataFrame,
    dup_fraction: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Duplicate dup_fraction of rows then shuffle to hide the pattern.

    Biases: sum, count, mean.
    NOT biased: max, min, nunique.

    Returns metadata: perturbation, dup_fraction, n_rows_before, n_rows_after.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    n_dup = max(1, int(round(n * dup_fraction)))
    idx_to_dup = rng.choice(n, size=n_dup, replace=False)
    combined = pd.concat([df, df.iloc[idx_to_dup]], ignore_index=True)
    shuffled = combined.sample(frac=1, random_state=int(seed)).reset_index(drop=True)
    return shuffled, {
        "perturbation": "row_duplication",
        "dup_fraction": dup_fraction,
        "n_rows_before": n,
        "n_rows_after": len(shuffled),
    }


def locale_format(
    df: pd.DataFrame,
    columns: list[str],
    seed: int = 42,  # noqa: ARG001 — kept for uniform signature
) -> tuple[pd.DataFrame, dict]:
    """Convert numeric columns to French-locale strings ("1 234,56").

    dtype changes to object; any direct numeric aggregation or float()
    call on the column will fail or produce wrong results.

    Returns metadata: perturbation, columns (actually converted).
    """
    result = df.copy()
    converted: list[str] = []
    for col in columns:
        if col not in result.columns:
            continue
        series = pd.to_numeric(result[col], errors="coerce")
        result[col] = series.apply(_to_fr_locale)
        converted.append(col)
    return result, {
        "perturbation": "locale_format",
        "columns": converted,
    }


def outlier_injection(
    df: pd.DataFrame,
    column: str,
    n_outliers: int,
    magnitude: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Replace n_outliers cells with extreme values (mean ± magnitude × std).

    Alternates sign to avoid one-directional bias.
    Biases: mean, sum, max. Does NOT affect count.

    Returns metadata: perturbation, column, cells_modified, magnitude.
    """
    rng = np.random.default_rng(seed)
    result = df.copy()
    # Cast to float64 so integer columns can accept fractional outlier values
    result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    series = result[column]
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        std = abs(mean) if mean != 0 else 1.0

    n_actual = min(n_outliers, len(result))
    idx_to_corrupt = rng.choice(len(result), size=n_actual, replace=False)
    signs = rng.choice([-1, 1], size=n_actual)
    outlier_values = mean + signs * magnitude * std

    col_pos = result.columns.get_loc(column)
    result.iloc[idx_to_corrupt, col_pos] = outlier_values

    return result, {
        "perturbation": "outlier_injection",
        "column": column,
        "cells_modified": n_actual,
        "magnitude": magnitude,
    }


# ---------------------------------------------------------------------------
# Question-level sensitivity label
# ---------------------------------------------------------------------------

# Keywords whose presence in the question implies the answer is NOT
# affected by the given perturbation mode.
_INSENSITIVE_KEYWORDS: dict[str, set[str]] = {
    "row_duplication": {
        # max/min are order-statistics — unaffected by row duplication
        "maximum", "max", "minimum", "min", "highest", "lowest",
        "largest", "smallest", "unique", "distinct", "median",
    },
    "outlier_injection": {
        # count/how-many is not affected by value replacement
        "how many", "count", "number of",
    },
}


def expected_sensitive(question: str, mode: str) -> bool:
    """Heuristic: will this perturbation likely change the answer to this question?

    Based on the aggregation implied by the question text.
    This is an a priori label — the real verdict is the observed delta.

    - row_duplication: NOT sensitive for max/min/unique questions.
    - locale_format: always sensitive (dtype change breaks all numeric ops).
    - outlier_injection: NOT sensitive for count/how-many questions.
    """
    if mode == "locale_format":
        return True
    insensitive = _INSENSITIVE_KEYWORDS.get(mode, set())
    q = question.lower()
    return not any(kw in q for kw in insensitive)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_fr_locale(value: float) -> str:
    """Format a float as a French-locale string ('1 234,56')."""
    if pd.isna(value):
        return ""
    # f"{v:,.2f}" → "1,234.56" (US): swap , → space and . → ,
    formatted = f"{value:,.2f}"
    integer_part, decimal_part = formatted.split(".")
    integer_fr = integer_part.replace(",", " ")
    return f"{integer_fr},{decimal_part}"
