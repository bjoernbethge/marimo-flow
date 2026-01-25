# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pina-mathlab",
#     "torch",
# ]
# ///

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
        # PINA: PINN Solver Training
        
        This snippet shows how to create a PINN solver, set up training controls
        with marimo UI elements, and train the model using PINA's Trainer.
        """
    )
    return


@app.cell
def _():
    from marimo_flow.core import (
        ProblemManager,
        SolverManager,
        train_solver,
    )
    
    # Create problem and solver using Managers
    problem_class = ProblemManager.create_poisson_problem()
    problem = problem_class()
    
    # Use SolverManager to create a PINN solver with a default FeedForward model
    from marimo_flow.core import ModelFactory
    
    model = ModelFactory.create_model_for_problem(problem)
    solver = SolverManager.create_pinn(problem, model)
    
    return (
        ModelFactory,
        ProblemManager,
        SolverManager,
        model,
        problem,
        problem_class,
        solver,
        train_solver,
    )


@app.cell
def _(mo):
    # Training controls
    epochs = mo.ui.slider(100, 1000, value=400, step=50, label="Max epochs")
    train_button = mo.ui.run_button(label="🚀 Start Training")
    
    mo.hstack([epochs, train_button])
    
    return epochs, train_button


@app.cell
def _(epochs, mo, solver, train_button, train_solver):
    # Training with status spinner
    mo.stop(not train_button.value, "Click 'Start Training' to begin.")
    
    with mo.status.spinner("Training PINN solver..."):
        trainer = train_solver(
            solver,
            max_epochs=epochs.value,
        )
    
    mo.md("✅ Training complete!")
    
    return trainer


if __name__ == "__main__":
    app.run()
