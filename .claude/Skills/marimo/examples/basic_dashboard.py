"""Basic Dashboard Template

A simple interactive dashboard template demonstrating:
- Core imports
- UI controls (slider, dropdown, checkbox)
- Data processing
- Reactive visualization
"""

import marimo

app = marimo.App(width="medium", app_title="Basic Dashboard")


@app.cell
def imports():
    """Import required libraries"""
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    return mo, pd, np, alt


@app.cell
def load_data(pd, np):
    """Load or generate sample data"""
    # Generate sample dataset
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.randn(100).cumsum() + 100,
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    return df,


@app.cell
def ui_controls(mo):
    """Create interactive UI controls"""
    # Numeric control
    threshold = mo.ui.slider(
        start=0,
        stop=200,
        value=100,
        label="Threshold"
    )

    # Selection control
    category_filter = mo.ui.dropdown(
        options=['All', 'A', 'B', 'C'],
        value='All',
        label="Category"
    )

    # Boolean control
    show_trend = mo.ui.checkbox(
        value=True,
        label="Show trend line"
    )

    # Layout controls
    controls = mo.vstack([
        mo.md("## Controls"),
        threshold,
        category_filter,
        show_trend
    ], gap=1.5)

    controls  # Display

    return threshold, category_filter, show_trend,


@app.cell
def process_data(df, threshold, category_filter):
    """Process data based on UI controls"""
    # Filter by threshold
    filtered_df = df[df['value'] >= threshold.value]

    # Filter by category
    if category_filter.value != 'All':
        filtered_df = filtered_df[
            filtered_df['category'] == category_filter.value
        ]

    return filtered_df,


@app.cell
def create_visualization(filtered_df, show_trend, alt, mo):
    """Create reactive visualization"""
    # Base chart
    base = alt.Chart(filtered_df).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('value:Q', title='Value'),
        color=alt.Color('category:N', title='Category')
    )

    # Points
    points = base.mark_circle(size=60, opacity=0.7)

    # Trend line (conditional)
    if show_trend.value:
        trend = base.transform_regression(
            'date', 'value'
        ).mark_line(strokeDash=[5, 5], color='gray')
        chart = points + trend
    else:
        chart = points

    chart = chart.properties(
        width=600,
        height=400,
        title=f"Dashboard - {len(filtered_df)} data points"
    )

    mo.ui.altair_chart(chart)
    return chart,


@app.cell
def summary_metrics(filtered_df, mo):
    """Display summary metrics"""
    metrics = mo.hstack([
        mo.stat(
            label="Total Points",
            value=len(filtered_df)
        ),
        mo.stat(
            label="Average Value",
            value=f"{filtered_df['value'].mean():.2f}"
        ),
        mo.stat(
            label="Max Value",
            value=f"{filtered_df['value'].max():.2f}"
        )
    ], gap=2, widths="equal")

    metrics  # Display
    return


@app.cell
def data_table(filtered_df, mo):
    """Display interactive data table"""
    mo.md("## Data Table")
    mo.ui.table(
        filtered_df,
        sortable=True,
        pagination=True,
        page_size=10
    )
    return


if __name__ == "__main__":
    app.run()
