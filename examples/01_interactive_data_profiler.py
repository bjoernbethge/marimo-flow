# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo[mcp]",
#     "pyzmq",
#     "duckdb",
#     "altair",
#     "pyarrow==22.0.0",
#     "polars==1.35.2",
#     "polars-runtime-32==1.35.2",
#     "openai==2.8.1",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import polars as pl
    from marimo_flow.snippets import build_interactive_scatter, filter_dataframe
    return build_interactive_scatter, duckdb, filter_dataframe, mo, pl


@app.cell
def _(mo):
    db_path = mo.ui.text(
        value="data/rag_production.duckdb",
        label="DuckDB database",
        placeholder="Path to .duckdb file",
    )
    refresh = mo.ui.button("Refresh catalog")
    mo.hstack([db_path, refresh], justify="start")
    return (db_path,)


@app.cell
def _(db_path, mo, run_query):
    tables_df = run_query(db_path.value, "SHOW TABLES")
    tables = tables_df["name"].to_list() if not tables_df.is_empty() else []
    table_select = mo.ui.dropdown(
        options=tables,
        value=tables[0] if tables else None,
        label="Table",
    )
    return (table_select,)


@app.cell
def _(duckdb, pl):
    def run_query(db_path: str, query: str) -> pl.DataFrame:
        """Execute a SQL query on DuckDB and return a Polars DataFrame."""
        conn = duckdb.connect(db_path)
        try:
            result = conn.execute(query).pl()
            return result
        finally:
            conn.close()
    return (run_query,)


@app.cell
def _(db_path, pl, run_query, table_select):
    if not table_select.value:
        df = pl.DataFrame()
    else:
        df = run_query(
            db_path.value,
            f"SELECT * FROM {table_select.value} LIMIT 5000",
        )
    return (df,)


@app.cell
def _(df, mo, table_select):
    if df.is_empty():
        mo.md("_No rows available. Choose another table or refresh the database._")
    else:
        mo.md(
            f"""
            ### Overview: `{table_select.value}`

            - Rows fetched: **{df.height}**
            - Columns: **{df.width}**
            """
        )
        df.head(10)
    return


@app.cell
def _(df, mo):


    column_filter = mo.ui.dropdown(
        options=df.columns,
        label="Column to filter",
    )
    text_filter = mo.ui.text(label="Contains text")
    max_rows = mo.ui.slider(10, 200, value=50, label="Rows to view")
    mo.hstack([column_filter, text_filter, max_rows])

    return column_filter, max_rows, text_filter


@app.cell
def _(column_filter, df, filter_dataframe, max_rows, mo, text_filter):
    filtered = filter_dataframe(df, column_filter.value, text_filter.value)
    preview = filtered.head(max_rows.value)

    mo.md(f"Matching rows: **{filtered.height}**")
    preview
    return (filtered,)


app._unparsable_cell(
    r"""
    if filtered.is_empty():
        return ()

    numeric_cols = [
        col
        for col, dtype in zip(filtered.columns, filtered.dtypes)
        if pl.datatypes.is_numeric(dtype)
    ]

    if len(numeric_cols) < 2:
        mo.md(\"_Need at least two numeric columns for a scatter plot._\")
        return ()

    x_axis = mo.ui.dropdown(numeric_cols, value=numeric_cols[0], label=\"X axis\")
    y_axis = mo.ui.dropdown(numeric_cols, value=numeric_cols[1], label=\"Y axis\")
    color = mo.ui.dropdown(
        options=[None, *filtered.columns],
        value=None,
        label=\"Color by\",
    )
    mo.hstack([x_axis, y_axis, color])

    """,
    name="_"
)


@app.cell
def _(build_interactive_scatter, color, filtered, x_axis, y_axis):
    chart = build_interactive_scatter(
        filtered,
        x_field=x_axis.value,
        y_field=y_axis.value,
        color_field=color.value,
    )
    chart.display()
    return


if __name__ == "__main__":
    app.run()
