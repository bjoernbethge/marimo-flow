import marimo as mo
import polars as pl
from marimo_flow.snippets import filter_dataframe

__generated_with = "0.18.0"
app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from marimo_flow.snippets import filter_dataframe
    
    df = pl.DataFrame(
        {
            "city": ["Berlin", "Berlin", "Hamburg", "Munich"],
            "category": ["A", "B", "A", "B"],
            "value": [10, 15, 7, 20],
        }
    )
    column_selector = mo.ui.dropdown(df.columns, label="Select column")
    filter_value = mo.ui.text(label="Filter value")
    limit = mo.ui.slider(1, 10, value=5, label="Max rows")
    mo.hstack([column_selector, filter_value, limit])
    return column_selector, df, filter_value, limit


@app.cell
def _(column_selector, df, filter_value, limit):
    import marimo as mo
    from marimo_flow.snippets import filter_dataframe
    
    filtered = filter_dataframe(
        df, column_selector.value, filter_value.value or None
    )
    preview = filtered.head(limit.value)
    mo.md(f"Matching rows: **{filtered.height}**")
    preview
    return


if __name__ == "__main__":
    app.run()


