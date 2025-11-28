"""Utilities for lightweight data exploration."""

from __future__ import annotations

import polars as pl


def filter_dataframe(
    df: pl.DataFrame,
    column: str | None,
    value_substring: str | None,
) -> pl.DataFrame:
    """Return rows whose column contains the provided substring."""

    if not column or not value_substring:
        return df

    return df.filter(
        pl.col(column).cast(pl.Utf8).str.contains(value_substring, literal=True)
    )

