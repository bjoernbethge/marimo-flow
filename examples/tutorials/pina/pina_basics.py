import marimo as mo

__generated_with = "0.18.0"
app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from marimo_flow.core.modeling import build_solver
    from marimo_flow.core.problem import build_problem
    from marimo_flow.core.training import train_solver
    from marimo_flow.core.visualization import build_heatmap_chart, generate_heatmap_data
    return (mo, build_solver, build_problem, train_solver, build_heatmap_chart, generate_heatmap_data)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Physics-Informed Neural Networks with PINA
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Define the Problem (Poisson Equation)
    """)
    return


@app.cell
def _(build_problem):
    problem = build_problem()
    return (problem,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Define the Model and Solver
    """)
    return


@app.cell
def _(build_solver, problem):
    solver = build_solver(problem)
    return (solver,)


@app.cell
def _(mo):
    """Training control"""
    train_button = mo.ui.run_button(label="🚀 Train PINN")
    mo.md(f"{train_button}")
    return (train_button,)


@app.cell
def _(mo, solver, train_button, train_solver):
    mo.stop(not train_button.value, "Click 'Train PINN' to start training")

    with mo.status.spinner("Training PINN..."):
        trainer = train_solver(solver)

    print("✅ Training complete")
    return (trainer,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Visualize Solution
    """)
    return


@app.cell
def _(build_heatmap_chart, generate_heatmap_data, solver):
    df, X, Y = generate_heatmap_data(solver)
    chart = build_heatmap_chart(df)
    return (chart,)


@app.cell
def _(chart):
    chart.display()
    return


if __name__ == "__main__":
    app.run()
