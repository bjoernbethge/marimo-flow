# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pina-mathlab",
#     "torch",
#     "altair",
#     "polars",
#     "numpy",
# ]
# ///

import marimo as mo
import torch

from marimo_flow.core import (
    ModelFactory,
    ProblemManager,
    SolverManager,
    WalrusAdapter,
    build_heatmap_chart,
    generate_heatmap_data,
    train_solver,
)

__generated_with = "0.18.0"
app = mo.App(width="medium")


@app.cell
def _():
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Physics-Informed Solver with Walrus

        Toggle between the baseline PINN and the Walrus foundation model
        (adapter + lightweight head) to solve the Poisson equation on the
        unit square.
        """
    )
    return


@app.cell
def _():
    problem_class = ProblemManager.create_poisson_problem()
    problem = problem_class()
    return problem, problem_class


@app.cell
def _(mo):
    model_choice = mo.ui.dropdown(
        options=["baseline", "walrus"],
        value="baseline",
        label="Backbone",
    )
    lr = mo.ui.number(value=1e-3, label="Learning rate")
    epochs = mo.ui.slider(100, 1500, value=400, step=50, label="Epochs")
    freeze = mo.ui.switch(value=True, label="Freeze Walrus backbone")
    mo.hstack([model_choice, lr, epochs, freeze])
    return epochs, freeze, lr, model_choice


@app.cell
def _(epochs, freeze, lr, model_choice, problem):
    model = ModelFactory.create_model_for_problem(problem)
    solver = SolverManager.create_pinn(problem, model)
    
    if model_choice.value == "walrus":
        solver.model = WalrusAdapter(freeze_backbone=freeze.value)

    trainable = filter(lambda p: p.requires_grad, solver.model.parameters())
    solver.optimizer = torch.optim.Adam(trainable, lr=lr.value)
    solver._max_epochs = epochs.value  # type: ignore[attr-defined]
    return (solver,)


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Start training")
    mo.md(f"{train_button}")
    return (train_button,)


@app.cell
def _(epochs, mo, solver, train_button):
    mo.stop(not train_button.value, "Click the button to start training.")
    with mo.status.spinner("Training in progress..."):
        trainer = train_solver(
            solver,
            max_epochs=epochs.value,
        )
    mo.md("✅ Training complete")
    return (trainer,)


@app.cell
def _(mo):
    mo.md("## Visualize solution")
    return


@app.cell
def _(solver):
    df, X, Y = generate_heatmap_data(solver)
    chart = build_heatmap_chart(df)
    return chart, df, X, Y


@app.cell
def _(chart):
    chart.display()
    return


if __name__ == "__main__":
    app.run()
