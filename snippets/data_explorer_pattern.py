"""
Data Explorer Pattern - Interactive Data Analysis Dashboard

Based on refs/integration-patterns.md Pattern 1
Combines: Marimo + Polars + Plotly
Demonstrates:
- Interactive data loading
- Column selection and filtering
- Dynamic visualizations
- Statistical summaries
"""

import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Interactive Data Explorer

        **Full-stack pattern:** Marimo + Polars + Plotly

        Load, filter, visualize, and analyze data interactively.
        """
    )
    return


@app.cell
def _():
    """Import libraries"""
    import polars as pl
    import plotly.express as px
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer

    return load_breast_cancer, load_iris, load_wine, pl, px


@app.cell
def _(load_breast_cancer, load_iris, load_wine, mo, pl):
    """Dataset selection"""
    dataset_choice = mo.ui.dropdown(
        options=["Iris", "Wine", "Breast Cancer"],
        value="Iris",
        label="📊 Dataset",
    )

    # Load selected dataset
    if dataset_choice.value == "Iris":
        data = load_iris()
    elif dataset_choice.value == "Wine":
        data = load_wine()
    else:
        data = load_breast_cancer()

    df = pl.DataFrame(data.data, schema=data.feature_names)
    df = df.with_columns(pl.Series("target", data.target.astype(str)))

    mo.md(f"## Dataset: {dataset_choice}\n\n**Shape:** {df.shape}")
    return (dataset_choice, df)


@app.cell
def _(df, mo):
    """Column selection"""
    columns = mo.ui.multiselect(
        options=list(df.columns[:-1]), value=list(df.columns[:2]), label="📋 Columns"
    )

    mo.md(f"### Select Columns\n\n{columns}")
    return (columns,)


@app.cell
def _(columns, df, mo):
    """Data preview"""
    selected_df = df.select(columns.value + ["target"]) if columns.value else df

    mo.md(
        f"""
        ### Data Preview

        {mo.ui.table(selected_df.head(10))}
        """
    )
    return (selected_df,)


@app.cell
def _(columns, mo, px, selected_df):
    """Visualization"""
    if len(columns.value) >= 2:
        fig = px.scatter(
            selected_df.to_pandas(),
            x=columns.value[0],
            y=columns.value[1],
            color="target",
            title=f"{columns.value[0]} vs {columns.value[1]}",
        )

        mo.md(f"### Visualization\n\n{mo.ui.plotly(fig)}")
    else:
        mo.md("_Select at least 2 columns for visualization_")
    return


@app.cell
def _(mo, selected_df):
    """Statistics"""
    stats = selected_df.select(pl.col(pl.NUMERIC_DTYPES)).describe()

    mo.md(f"### Statistics\n\n{mo.ui.table(stats)}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 💡 Pattern: marimo + polars + plotly

        **Architecture:**
        1. **Load data** with Polars (fast!)
        2. **Interactive controls** with marimo.ui
        3. **Filter/transform** with Polars expressions
        4. **Visualize** with Plotly
        5. **Display** with marimo

        See `refs/integration-patterns.md` for full pattern.
        """
    )
    return


if __name__ == "__main__":
    app.run()
