import marimo as mo
import polars as pl
from marimo_flow.snippets import build_interactive_scatter

__generated_with = "0.18.0"
app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from marimo_flow.snippets import build_interactive_scatter
    
    df = pl.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 6.2, 5.7],
            "sepal_width": [3.5, 3.0, 2.8, 2.9],
            "species": ["setosa", "setosa", "virginica", "versicolor"],
        }
    )
    x_axis = mo.ui.dropdown(df.columns, value="sepal_length", label="X axis")
    y_axis = mo.ui.dropdown(df.columns, value="sepal_width", label="Y axis")
    color = mo.ui.dropdown(["species", None], value="species", label="Color")
    mo.hstack([x_axis, y_axis, color])
    return color, df, x_axis, y_axis


@app.cell
def _(color, df, x_axis, y_axis):
    from marimo_flow.snippets import build_interactive_scatter
    
    chart = build_interactive_scatter(
        df,
        x_field=x_axis.value,
        y_field=y_axis.value,
        color_field=color.value,
    )
    chart.display()
    return


if __name__ == "__main__":
    app.run()


