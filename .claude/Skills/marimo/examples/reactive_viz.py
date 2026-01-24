"""Reactive Visualization Example

Demonstrates reactive data visualization patterns:
- UI controls that drive data generation
- Automatic chart updates
- Multiple linked visualizations
- Interactive chart selection
"""

import marimo

app = marimo.App(width="full", app_title="Reactive Visualization")


@app.cell
def imports():
    """Import required libraries"""
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    return mo, np, pd, alt


@app.cell
def data_controls(mo):
    """Controls for data generation"""
    # Number of points
    n_points = mo.ui.slider(
        start=10,
        stop=500,
        value=100,
        label="Number of Points",
        step=10
    )

    # Distribution type
    distribution = mo.ui.dropdown(
        options=['normal', 'uniform', 'exponential'],
        value='normal',
        label="Distribution"
    )

    # Random seed for reproducibility
    seed = mo.ui.number(
        start=0,
        stop=1000,
        value=42,
        label="Random Seed"
    )

    # Noise level
    noise = mo.ui.slider(
        start=0.0,
        stop=2.0,
        value=0.1,
        step=0.1,
        label="Noise Level"
    )

    # Layout controls in sidebar
    controls = mo.vstack([
        mo.md("## Data Controls"),
        n_points,
        distribution,
        seed,
        noise
    ], gap=1.5)

    controls  # Display

    return n_points, distribution, seed, noise,


@app.cell
def generate_data(n_points, distribution, seed, noise, np, pd):
    """Generate data based on controls"""
    # Set random seed
    np.random.seed(seed.value)

    n = n_points.value

    # Generate X values
    x = np.linspace(0, 10, n)

    # Generate Y values based on distribution
    if distribution.value == 'normal':
        y = np.sin(x) + np.random.normal(0, noise.value, n)
    elif distribution.value == 'uniform':
        y = np.sin(x) + np.random.uniform(-noise.value, noise.value, n)
    else:  # exponential
        y = np.sin(x) + np.random.exponential(noise.value, n) - noise.value

    # Create DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'category': np.random.choice(['A', 'B', 'C'], n)
    })

    return df,


@app.cell
def scatter_plot(df, alt, mo):
    """Interactive scatter plot with selection"""
    # Create scatter plot with interval selection
    brush = alt.selection_interval()

    scatter = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('x:Q', title='X Value', scale=alt.Scale(zero=False)),
        y=alt.Y('y:Q', title='Y Value', scale=alt.Scale(zero=False)),
        color=alt.condition(
            brush,
            alt.Color('category:N', title='Category'),
            alt.value('lightgray')
        ),
        tooltip=['x:Q', 'y:Q', 'category:N']
    ).add_params(
        brush
    ).properties(
        width=600,
        height=400,
        title=f"Scatter Plot ({len(df)} points)"
    )

    # Make chart reactive
    scatter_chart = mo.ui.altair_chart(scatter)
    scatter_chart  # Display

    return scatter_chart, brush,


@app.cell
def histogram(df, scatter_chart, alt, mo):
    """Histogram of Y values (updates with selection)"""
    # Get selected data or use all data
    selected_data = scatter_chart.value
    data_to_plot = selected_data if len(selected_data) > 0 else df

    hist = alt.Chart(data_to_plot).mark_bar(opacity=0.7).encode(
        x=alt.X('y:Q', bin=alt.Bin(maxbins=30), title='Y Value'),
        y=alt.Y('count()', title='Frequency'),
        color=alt.Color('category:N', title='Category')
    ).properties(
        width=600,
        height=200,
        title=f"Distribution ({len(data_to_plot)} points selected)"
    )

    hist  # Display

    return hist,


@app.cell
def statistics_panel(df, scatter_chart, mo):
    """Display statistics for selected data"""
    selected_data = scatter_chart.value
    data_to_analyze = selected_data if len(selected_data) > 0 else df

    # Calculate statistics
    stats = mo.hstack([
        mo.stat(
            label="Points",
            value=len(data_to_analyze),
            caption=f"out of {len(df)}"
        ),
        mo.stat(
            label="Mean Y",
            value=f"{data_to_analyze['y'].mean():.3f}",
            caption=f"std: {data_to_analyze['y'].std():.3f}"
        ),
        mo.stat(
            label="Min Y",
            value=f"{data_to_analyze['y'].min():.3f}"
        ),
        mo.stat(
            label="Max Y",
            value=f"{data_to_analyze['y'].max():.3f}"
        )
    ], gap=2, widths="equal")

    stats  # Display

    return


@app.cell
def category_breakdown(df, scatter_chart, mo):
    """Show category breakdown of selected data"""
    selected_data = scatter_chart.value
    data_to_analyze = selected_data if len(selected_data) > 0 else df

    # Count by category
    category_counts = data_to_analyze.groupby('category').size()

    breakdown = mo.md(f"""
    ### Category Breakdown

    - **Category A**: {category_counts.get('A', 0)} points
    - **Category B**: {category_counts.get('B', 0)} points
    - **Category C**: {category_counts.get('C', 0)} points
    """)

    breakdown  # Display

    return


@app.cell
def time_series_view(df, alt, mo):
    """Time series view with trend line"""
    # Sort by x for proper line chart
    df_sorted = df.sort_values('x')

    # Points
    points = alt.Chart(df_sorted).mark_circle(size=40, opacity=0.5).encode(
        x=alt.X('x:Q', title='X Value'),
        y=alt.Y('y:Q', title='Y Value'),
        color=alt.Color('category:N', title='Category')
    )

    # Line connecting points
    line = alt.Chart(df_sorted).mark_line(strokeWidth=2).encode(
        x='x:Q',
        y='y:Q',
        color='category:N'
    )

    # Trend line
    trend = alt.Chart(df_sorted).transform_regression(
        'x', 'y'
    ).mark_line(strokeDash=[5, 5], color='red', strokeWidth=2).encode(
        x='x:Q',
        y='y:Q'
    )

    chart = (points + line + trend).properties(
        width=600,
        height=300,
        title="Time Series View with Trend"
    )

    chart  # Display

    return


@app.cell
def usage_info(mo):
    """Display usage instructions"""
    mo.callout(
        mo.md("""
        **Interactive Visualization Demo**

        - Adjust controls on the left to change data generation
        - Click and drag on scatter plot to select points
        - Other visualizations update based on selection
        - Charts automatically update when controls change
        """),
        kind="info"
    )
    return


if __name__ == "__main__":
    app.run()
