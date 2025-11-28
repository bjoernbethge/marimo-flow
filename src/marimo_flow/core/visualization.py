"""Visualization utilities for the PINA demo."""

from __future__ import annotations

import altair as alt
import numpy as np
import polars as pl
import torch
from pina import LabelTensor
from pina.solver import PINN


def generate_heatmap_data(
    solver: PINN,
    grid_size: int = 50,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Evaluate the solver on a regular grid and return the results."""

    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)
    pts.requires_grad = True
    input_tensor = LabelTensor(pts, labels=["x", "y"])

    with torch.no_grad():
        u_pred = solver.model(input_tensor).numpy().flatten()

    df = pl.DataFrame({"x": X.ravel(), "y": Y.ravel(), "u": u_pred})
    return df, X, Y


def build_heatmap_chart(df: pl.DataFrame, title: str = "PINN Solution u(x,y)") -> alt.Chart:
    """Return an Altair heatmap for the predicted solution."""

    return (
        alt.Chart(df.to_pandas())
        .mark_rect()
        .encode(
            x="x:Q",
            y="y:Q",
            color="u:Q",
            tooltip=["x", "y", "u"],
        )
        .properties(title=title)
    )


def generate_error_data(
    solver: PINN,
    exact_solution_fn: callable,
    grid_size: int = 50,
) -> pl.DataFrame:
    """Generate prediction, exact solution, and error data on a grid.
    
    Args:
        solver: Trained PINA solver
        exact_solution_fn: Function that takes input tensor and returns exact solution tensor
        grid_size: Resolution of the evaluation grid
        
    Returns:
        DataFrame with x, y, u_pred, u_exact, error
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)

    pts_np = np.column_stack([X.ravel(), Y.ravel()])
    pts = torch.tensor(pts_np, dtype=torch.float32)
    pts.requires_grad = True
    input_tensor = LabelTensor(pts, labels=["x", "y"])

    with torch.no_grad():
        u_pred = solver.model(input_tensor).numpy().flatten()
        
        # Calculate exact solution
        # Note: exact_solution_fn might expect LabelTensor or just tensor
        try:
            u_exact = exact_solution_fn(input_tensor)
        except Exception:
            u_exact = exact_solution_fn(pts)
            
        if hasattr(u_exact, "numpy"):
            u_exact = u_exact.numpy()
        u_exact = u_exact.flatten()

    error = np.abs(u_pred - u_exact)

    df = pl.DataFrame({
        "x": X.ravel(),
        "y": Y.ravel(),
        "u_pred": u_pred,
        "u_exact": u_exact,
        "error": error
    })
    return df


def build_comparison_chart(df: pl.DataFrame) -> alt.Chart:
    """Create a side-by-side comparison chart: Prediction | Exact | Error.
    
    Args:
        df: DataFrame with columns x, y, u_pred, u_exact, error
        
    Returns:
        Altair chart with 3 faceted heatmaps
    """
    # Convert to long format for faceting
    df_long = df.unpivot(
        index=["x", "y"], 
        on=["u_pred", "u_exact", "error"], 
        variable_name="type", 
        value_name="value"
    )
    
    return (
        alt.Chart(df_long.to_pandas())
        .mark_rect()
        .encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color("value:Q", scale=alt.Scale(scheme="viridis")),
            column=alt.Column("type:N", title=None),
            tooltip=["x", "y", "value"]
        )
        .properties(title="Solution Comparison")
        .resolve_scale(color="independent")
    )
