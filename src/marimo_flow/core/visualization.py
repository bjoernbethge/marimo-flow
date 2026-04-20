"""Visualization utilities built on Optuna's native plotting APIs."""

from __future__ import annotations

import altair as alt
import optuna
import polars as pl
from optuna.visualization import (
    plot_optimization_history as build_optuna_history_figure,
)
from optuna.visualization import (
    plot_parallel_coordinate as build_optuna_parallel_figure,
)
from optuna.visualization import (
    plot_param_importances as build_optuna_param_importance_figure,
)

__all__ = [
    "build_optuna_history_figure",
    "build_optuna_parallel_figure",
    "build_optuna_param_importance_figure",
    "build_trials_scatter_chart",
    "study_trials_dataframe",
]


def study_trials_dataframe(study: optuna.Study) -> pl.DataFrame:
    """Return completed Optuna trials as a typed dataframe."""

    rows: list[dict[str, object]] = []
    for trial in study.trials:
        if trial.value is None:
            continue
        row = {"trial": trial.number, "loss": float(trial.value)}
        row.update(trial.params)
        rows.append(row)
    return pl.DataFrame(rows).sort("loss")


def build_trials_scatter_chart(df: pl.DataFrame, color_by: str | None = None) -> alt.Chart:
    """Build a lightweight Altair scatter chart for trial/loss overview."""
    if df.is_empty():
        return alt.Chart(pl.DataFrame({"trial": [], "loss": []}).to_pandas()).mark_point()

    base_chart = alt.Chart(df.to_pandas()).mark_point(filled=True, size=80)
    if color_by and color_by in df.columns:
        chart = base_chart.encode(
            x=alt.X("trial:Q", title="Trial"),
            y=alt.Y("loss:Q", scale=alt.Scale(type="log"), title="Loss"),
            color=alt.Color(f"{color_by}:N"),
            tooltip=list(df.columns),
        )
    else:
        chart = base_chart.encode(
            x=alt.X("trial:Q", title="Trial"),
            y=alt.Y("loss:Q", scale=alt.Scale(type="log"), title="Loss"),
            tooltip=list(df.columns),
        )
    return chart.properties(height=260, title="Loss per Trial").interactive()

