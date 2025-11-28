"""Minimal Altair chart helpers."""

from __future__ import annotations

import altair as alt
import polars as pl


def build_interactive_scatter(
    df: pl.DataFrame,
    x_field: str,
    y_field: str,
    color_field: str | None = None,
) -> alt.Chart:
    """Return a scatter plot with optional color encoding."""

    encoding = {
        "x": alt.X(x_field, type="quantitative"),
        "y": alt.Y(y_field, type="quantitative"),
    }
    if color_field:
        encoding["color"] = alt.Color(color_field, type="nominal")
    else:
        encoding["color"] = alt.value("#2563eb")

    return (
        alt.Chart(df.to_pandas())
        .mark_circle(size=120, opacity=0.7)
        .encode(**encoding, tooltip=list(df.columns))
    )

