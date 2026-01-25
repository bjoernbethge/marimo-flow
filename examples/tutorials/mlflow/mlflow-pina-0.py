# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "pina-mathlab",
#     "mlflow",
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
        # MLflow + PINA: Training with Tracking
        
        This snippet demonstrates how to integrate PINA training with MLflow
        tracking, logging hyperparameters, metrics, and the trained model.
        """
    )
    return


@app.cell
def _():
    import mlflow
    import mlflow.pytorch
    from marimo_flow.core import (
        ModelFactory,
        ProblemManager,
        SolverManager,
        train_solver,
    )
    from pathlib import Path
    
    # Setup MLflow
    tracking_path = Path("./data/experiments")
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_path.resolve()}")
    
    return (
        ModelFactory,
        Path,
        ProblemManager,
        SolverManager,
        mlflow,
        train_solver,
    )


@app.cell
def _(mlflow):
    # Create or get experiment
    experiment_name = "pina_training"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    
    return experiment, experiment_id, experiment_name


@app.cell
def _(ModelFactory, ProblemManager, SolverManager):
    # Create PINA problem and solver
    problem_class = ProblemManager.create_poisson_problem()
    problem = problem_class()
    
    model = ModelFactory.create_model_for_problem(problem)
    solver = SolverManager.create_pinn(problem, model)
    
    # Training hyperparameters
    max_epochs = 500
    learning_rate = 1e-3
    
    return learning_rate, max_epochs, model, problem, problem_class, solver


@app.cell
def _(experiment_id, learning_rate, max_epochs, mlflow, solver, train_solver):
    # Train with MLflow tracking
    with mlflow.start_run(run_name="pina_training_run") as run:
        # Log hyperparameters
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("solver_type", type(solver).__name__)
        mlflow.set_tag("framework", "pina")
        mlflow.set_tag("problem_type", "poisson")
        
        # Train solver
        trainer = train_solver(
            solver,
            max_epochs=max_epochs,
        )
        
        # Extract training metrics from trainer
        if hasattr(trainer, 'trainer') and trainer.trainer.callback_metrics:
            for key, value in trainer.trainer.callback_metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"train_{key}", float(value))
        
        # Log the trained model
        mlflow.pytorch.log_model(solver.model, "pina_model")
        
        run_id = run.info.run_id
    
    return run, run_id, trainer


if __name__ == "__main__":
    app.run()
