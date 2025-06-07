import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interactive Altair Visualization""")
    return

@app.cell
def _():
    """Generate Sample Data"""
    import polars as pl
    import numpy as np
    
    # Create sample dataset
    np.random.seed(42)
    n_points = 150
    
    df = pl.DataFrame({
        "x": np.random.randn(n_points),
        "y": np.random.randn(n_points) * 2,
        "category": np.random.choice(["Alpha", "Beta", "Gamma"], n_points),
        "size": np.random.randint(20, 200, n_points),
        "value": np.random.uniform(0, 100, n_points)
    })
    
    print(f"✅ Generated {len(df)} data points")
    return df, np, pl

@app.cell
def _(mo):
    """Chart Configuration"""
    chart_type = mo.ui.dropdown(
        options=["scatter", "line", "bar"],
        value="scatter",
        label="📊 Chart Type"
    )
    
    color_scheme = mo.ui.dropdown(
        options=["category10", "viridis", "plasma", "set2"],
        value="category10", 
        label="🎨 Color Scheme"
    )
    
    mo.md(f"""
    ## 🎯 Visualization Settings
    {chart_type}
    {color_scheme}
    """)
    return chart_type, color_scheme

@app.cell
def _(chart_type, color_scheme, df, mo):
    """Interactive Altair Chart"""
    import altair as alt
    
    # Base chart configuration with modern Altair syntax
    base = alt.Chart(df).add_params(
        alt.selection_interval(bind='scales')
    )
    
    # Create chart based on type
    if chart_type.value == "scatter":
        chart = base.mark_circle(size=60).encode(
            x=alt.X('x:Q', title='X Values'),
            y=alt.Y('y:Q', title='Y Values'),
            color=alt.Color('category:N', scale=alt.Scale(scheme=color_scheme.value)),
            size=alt.Size('size:Q', scale=alt.Scale(range=[50, 400])),
            tooltip=['x:Q', 'y:Q', 'category:N', 'value:Q']
        )
    elif chart_type.value == "line":
        chart = base.mark_line(point=True).encode(
            x=alt.X('x:Q', title='X Values'),
            y=alt.Y('y:Q', title='Y Values'),
            color=alt.Color('category:N', scale=alt.Scale(scheme=color_scheme.value)),
            tooltip=['x:Q', 'y:Q', 'category:N']
        )
    else:  # bar
        chart = base.mark_bar().encode(
            x=alt.X('category:N', title='Category'),
            y=alt.Y('mean(value):Q', title='Average Value'),
            color=alt.Color('category:N', scale=alt.Scale(scheme=color_scheme.value)),
            tooltip=['category:N', 'mean(value):Q']
        )
    
    chart = chart.properties(
        title=f"Interactive {chart_type.value.title()} Plot",
        width=500,
        height=350
    ).interactive()
    
    mo.md(f"""
    ## 📈 {chart_type.value.title()} Visualization
    {mo.ui.altair_chart(chart)}
    """)
    return alt, base, chart

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run() 