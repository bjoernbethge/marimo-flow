import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interactive Data Visualization with Altair""")
    return


@app.cell
def _():
    """Import visualization libraries"""

    from datetime import datetime, timedelta

    import altair as alt
    import numpy as np
    import polars as pl

    # Enable Altair data transformations
    alt.data_transformers.enable("default")
    return alt, datetime, np, pl, timedelta


@app.cell
def _(mo):
    """Dataset selection"""

    dataset_choice = mo.ui.dropdown(
        options=["Synthetic Time Series", "Iris Dataset", "Random Scatter"],
        value="Synthetic Time Series",
        label="📊 Choose Dataset",
    )

    mo.md(f"## 📈 Dataset Selection\n{dataset_choice}")
    return (dataset_choice,)


@app.cell
def _(dataset_choice, datetime, np, pl, timedelta):
    """Load selected dataset"""

    if dataset_choice.value == "Synthetic Time Series":
        # Generate time series data
        dates = pl.date_range(
            start=datetime.now() - timedelta(days=365), end=datetime.now(), interval="D"
        )

        df = pl.DataFrame(
            {
                "date": dates,
                "value": pl.cumsum(pl.lit(np.random.randn(len(dates))) + 100),
                "category": pl.lit(np.random.choice(["A", "B", "C"], len(dates))),
                "volume": pl.lit(np.random.randint(50, 200, len(dates))),
            }
        )

    elif dataset_choice.value == "Iris Dataset":
        from sklearn.datasets import load_iris

        iris = load_iris()
        df = pl.DataFrame(data=iris.data, columns=iris.feature_names)
        df["species"] = [iris.target_names[i] for i in iris.target]

    else:  # Random Scatter
        n_points = 500
        df = pl.DataFrame(
            {
                "x": pl.lit(np.random.randn(n_points)),
                "y": pl.lit(np.random.randn(n_points)),
                "size": pl.lit(np.random.randint(10, 100, n_points)),
                "category": pl.lit(
                    np.random.choice(["Group 1", "Group 2", "Group 3"], n_points)
                ),
            }
        )

    print(f"✅ Loaded {dataset_choice.value}: {len(df)} rows")
    return (df,)


@app.cell
def _(df, mo):
    """Data preview"""

    mo.md(f"""
    ### 🔍 Data Preview
    {mo.ui.table(df.head(10))}
    """)
    return


@app.cell
def _(dataset_choice, df, mo):
    """Visualization controls"""

    if dataset_choice.value == "Synthetic Time Series":
        chart_type = mo.ui.dropdown(
            options=["Line Chart", "Area Chart", "Bar Chart"],
            value="Line Chart",
            label="📈 Chart Type",
        )
        show_points = mo.ui.checkbox(label="Show Points", value=False)
        show_volume = mo.ui.checkbox(label="Show Volume", value=True)

    elif dataset_choice.value == "Iris Dataset":
        x_axis = mo.ui.dropdown(
            options=list(df.columns[:-1]), value="sepal length (cm)", label="X Axis"
        )
        y_axis = mo.ui.dropdown(
            options=list(df.columns[:-1]), value="sepal width (cm)", label="Y Axis"
        )
        show_regression = mo.ui.checkbox(label="Show Regression", value=False)

    else:  # Random Scatter
        color_by_category = mo.ui.checkbox(label="Color by Category", value=True)
        size_by_value = mo.ui.checkbox(label="Size by Value", value=True)
        opacity = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.7, label="Opacity")
    return (
        chart_type,
        color_by_category,
        opacity,
        show_points,
        show_regression,
        show_volume,
        size_by_value,
        x_axis,
        y_axis,
    )


@app.cell
def _(mo):
    """Theme selection"""

    theme = mo.ui.dropdown(
        options=["Default", "Dark", "Quartz", "Vox", "FiveThirtyEight"],
        value="Default",
        label="🎨 Chart Theme",
    )

    interactive = mo.ui.checkbox(label="Enable Interactivity", value=True)

    mo.md(f"### 🎨 Visualization Settings\n{theme} {interactive}")
    return interactive, theme


@app.cell
def _(alt, theme):
    """Apply theme"""

    theme_map = {
        "Default": alt.themes.enable("default"),
        "Dark": alt.themes.enable("dark"),
        "Quartz": alt.themes.enable("quartz"),
        "Vox": alt.themes.enable("vox"),
        "FiveThirtyEight": alt.themes.enable("fivethirtyeight"),
    }

    # Note: In practice, themes need to be registered first
    # This is a simplified version
    print(f"✅ Theme set to: {theme.value}")
    return


@app.cell
def _(
    alt,
    chart_type,
    color_by_category,
    dataset_choice,
    df,
    interactive,
    opacity,
    pl,
    show_points,
    show_regression,
    show_volume,
    size_by_value,
    x_axis,
    y_axis,
):
    """Create visualization"""

    if dataset_choice.value == "Synthetic Time Series":
        # Base chart
        if chart_type.value == "Line Chart":
            base = (
                alt.Chart(df)
                .mark_line()
                .encode(x="date:T", y="value:Q", color="category:N")
            )
        elif chart_type.value == "Area Chart":
            base = (
                alt.Chart(df)
                .mark_area(opacity=0.7)
                .encode(x="date:T", y="value:Q", color="category:N")
            )
        else:  # Bar Chart
            # Aggregate by month for bar chart
            df_monthly = (
                df.groupby([pl.col("date").dt.month, "category"])
                .agg({"value": "mean"})
                .reset_index()
            )
            base = (
                alt.Chart(df_monthly)
                .mark_bar()
                .encode(x="date:T", y="value:Q", color="category:N")
            )

        # Add points if requested
        if show_points.value and chart_type.value == "Line Chart":
            points = (
                alt.Chart(df)
                .mark_circle(size=50)
                .encode(x="date:T", y="value:Q", color="category:N")
            )
            chart = base + points
        else:
            chart = base

        # Add volume subplot if requested
        if show_volume.value:
            volume = (
                alt.Chart(df)
                .mark_bar(opacity=0.3)
                .encode(x="date:T", y="volume:Q")
                .properties(height=100)
            )

            chart = alt.vconcat(chart.properties(height=300), volume)

    elif dataset_choice.value == "Iris Dataset":
        # Scatter plot
        base = (
            alt.Chart(df)
            .mark_circle(size=100)
            .encode(
                x=alt.X(x_axis.value, scale=alt.Scale(zero=False)),
                y=alt.Y(y_axis.value, scale=alt.Scale(zero=False)),
                color="species:N",
                tooltip=[x_axis.value, y_axis.value, "species"],
            )
        )

        # Add regression line if requested
        if show_regression.value:
            regression = base.transform_regression(
                x_axis.value, y_axis.value, groupby=["species"]
            ).mark_line()
            chart = base + regression
        else:
            chart = base

    else:  # Random Scatter
        encoding = {"x": "x:Q", "y": "y:Q", "tooltip": ["x", "y", "size", "category"]}

        if color_by_category.value:
            encoding["color"] = "category:N"

        if size_by_value.value:
            encoding["size"] = alt.Size("size:Q", scale=alt.Scale(range=[50, 400]))

        chart = alt.Chart(df).mark_circle(opacity=opacity.value).encode(**encoding)

    # Make interactive if requested
    if interactive.value:
        chart = chart.interactive()

    # Set width
    chart = chart.properties(width=700)
    return (chart,)


@app.cell
def _(chart, mo):
    """Display chart"""

    mo.md(f"""
    ## 📊 Visualization

    {mo.ui.altair_chart(chart)}
    """)
    return


@app.cell
def _(df, mo, np):
    """Statistical summary"""

    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        summary = df.describe()

        mo.md(f"""
        ## 📈 Statistical Summary

        {mo.ui.table(summary)}
        """)
    else:
        mo.md("## 📈 No numeric columns for statistical summary")
    return


@app.cell
def _(dataset_choice, mo):
    """Export options"""

    mo.md(f"""
    ## 💾 Export Options

    ### Chart Export
    The Altair chart can be saved as:
    - **PNG/SVG**: Right-click on the chart
    - **JSON**: Use `chart.save('chart.json')`
    - **HTML**: Use `chart.save('chart.html')`

    ### Data Export
    - **CSV**: `df.to_csv('data.csv')`
    - **Excel**: `df.to_excel('data.xlsx')`
    - **JSON**: `df.to_json('data.json')`

    ### Interactive Features
    {"- **Zoom**: Click and drag to zoom" if dataset_choice.value != "Synthetic Time Series" else ""}
    {"- **Pan**: Hold shift and drag" if dataset_choice.value != "Synthetic Time Series" else ""}
    - **Tooltip**: Hover over data points
    - **Legend**: Click to filter categories
    """)
    return


if __name__ == "__main__":
    app.run()
