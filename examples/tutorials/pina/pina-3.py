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
        # PINA: Walrus Adapter Pattern
        
        This snippet demonstrates how to use the Walrus foundation model
        adapter with PINA, including the freeze backbone pattern for
        efficient fine-tuning.
        """
    )
    return


@app.cell
def _():
    from marimo_flow.core import (
        ModelFactory,
        ProblemManager,
        SolverManager,
        WalrusAdapter,
    )
    import torch
    
    # Create problem
    problem_class = ProblemManager.create_poisson_problem()
    problem = problem_class()
    
    # Initial solver with a standard model (will be swapped later)
    model = ModelFactory.create_model_for_problem(problem)
    solver = SolverManager.create_pinn(problem, model)
    
    return (
        ModelFactory,
        ProblemManager,
        SolverManager,
        WalrusAdapter,
        model,
        problem,
        problem_class,
        solver,
        torch,
    )


@app.cell
def _():
    import marimo as mo
    
    # Model selection and configuration
    model_choice = mo.ui.dropdown(
        options=["baseline", "walrus"],
        value="baseline",
        label="Backbone Model",
    )
    
    freeze_backbone = mo.ui.switch(
        value=True,
        label="Freeze Walrus Backbone",
    )
    
    learning_rate = mo.ui.number(
        value=1e-3,
        label="Learning Rate",
    )
    
    mo.hstack([model_choice, freeze_backbone, learning_rate])
    
    return freeze_backbone, learning_rate, model_choice, mo


@app.cell
def _(WalrusAdapter, freeze_backbone, learning_rate, model_choice, solver, torch):
    # Apply Walrus adapter if selected
    if model_choice.value == "walrus":
        solver.model = WalrusAdapter(freeze_backbone=freeze_backbone.value)
        
        # Update optimizer to only train unfrozen parameters
        trainable_params = filter(
            lambda p: p.requires_grad, solver.model.parameters()
        )
        solver.optimizer = torch.optim.Adam(
            trainable_params, lr=learning_rate.value
        )
    
    # Count trainable parameters
    trainable_count = sum(
        p.numel() for p in solver.model.parameters() if p.requires_grad
    )
    total_count = sum(p.numel() for p in solver.model.parameters())
    
    return solver, total_count, trainable_count, trainable_params


@app.cell
def _(mo, total_count, trainable_count):
    mo.md(
        f"""
        ## Model Parameters
        
        - **Total parameters**: {total_count:,}
        - **Trainable parameters**: {trainable_count:,}
        - **Frozen parameters**: {total_count - trainable_count:,}
        """
    )
    return


if __name__ == "__main__":
    app.run()
