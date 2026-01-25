import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # PINA: Solution Visualization
        
        This snippet demonstrates how to generate heatmap visualizations
        of PINA solver solutions using Altair charts.
        """
    )
    return


@app.cell
def _():
    from marimo_flow.core import (
        build_heatmap_chart,
        build_problem,
        build_solver,
        generate_heatmap_data,
    )
    
    # Create problem and solver (assume already trained)
    problem = build_problem()
    solver = build_solver(problem)
    
    return (
        build_heatmap_chart,
        build_problem,
        build_solver,
        generate_heatmap_data,
        problem,
        solver,
    )


@app.cell
def _(build_heatmap_chart, generate_heatmap_data, solver):
    # Generate heatmap data from solver
    df, X, Y = generate_heatmap_data(solver, grid_size=50)
    
    # Create Altair heatmap chart
    chart = build_heatmap_chart(df)
    
    return X, Y, chart, df


@app.cell
def _(chart):
    # Display the chart
    chart.display()
    return


if __name__ == "__main__":
    app.run()

