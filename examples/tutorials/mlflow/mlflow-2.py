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
        # MLflow: Model Registry Operations
        
        This snippet shows how to register models in MLflow Model Registry,
        manage model versions, transition stages (Staging → Production),
        and load models for inference.
        """
    )
    return


@app.cell
def _():
    import mlflow
    import mlflow.pytorch
    import polars as pl
    import torch
    import torch.nn as nn
    from pathlib import Path
    
    # Setup MLflow
    tracking_path = Path("./data/experiments")
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_path.resolve()}")
    
    return Path, mlflow, nn, pl, torch


@app.cell
def _(mlflow, nn, torch):
    # Create a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=64, num_classes=3):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes),
            )
        
        def forward(self, x):
            return self.network(x)
    
    model = SimpleModel()
    
    return model, SimpleModel


@app.cell
def _(mlflow, model):
    # Create or get experiment
    experiment_name = "model_registry_demo"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # Register model in a run
    registry_name = "demo_model"
    
    with mlflow.start_run(run_name="register_model") as run:
        mlflow.log_param("hidden_size", 64)
        mlflow.log_param("num_classes", 3)
        mlflow.log_metric("accuracy", 0.95)
        
        # Log and register model
        model_uri = mlflow.pytorch.log_model(
            model, "model", registered_model_name=registry_name
        )
        
        registered_model = mlflow.register_model(
            model_uri.model_uri, registry_name
        )
    
    return experiment, experiment_name, model_uri, registered_model, registry_name, run


@app.cell
def _(mlflow, pl, registry_name):
    # List registered models
    try:
        registered_models = mlflow.search_registered_models(
            filter_string=f"name='{registry_name}'"
        )
        
        if registered_models:
            model_info = []
            for model in registered_models:
                if model.latest_versions:
                    latest = model.latest_versions[0]
                    model_info.append({
                        "Name": model.name,
                        "Latest Version": latest.version,
                        "Stage": latest.current_stage,
                    })
            
            models_df = pl.DataFrame(model_info) if model_info else pl.DataFrame()
        else:
            models_df = pl.DataFrame()
    except Exception:
        models_df = pl.DataFrame()
    
    return model_info, models_df, registered_models


@app.cell
def _(mlflow, registry_name):
    # Get model versions
    try:
        model_versions = mlflow.search_model_versions(f"name='{registry_name}'")
        
        versions_data = []
        for version in model_versions:
            versions_data.append({
                "Version": version.version,
                "Stage": version.current_stage,
                "Run ID": version.run_id,
            })
        
        versions_df = pl.DataFrame(versions_data).sort("Version", descending=True) if versions_data else pl.DataFrame()
    except Exception:
        versions_df = pl.DataFrame()
    
    return model_versions, versions_data, versions_df


@app.cell
def _(mlflow, registry_name, versions_df):
    # Transition model stage (example: to Staging)
    try:
        if versions_df is not None and not versions_df.is_empty():
            version = int(versions_df["Version"][0])
            mlflow.tracking.MlflowClient().transition_model_version_stage(
                name=registry_name,
                version=version,
                stage="Staging",
            )
            stage_transition = f"Version {version} transitioned to Staging"
        else:
            stage_transition = "No versions available"
    except Exception as e:
        stage_transition = f"Error: {e}"
    
    return stage_transition


@app.cell
def _(mlflow, registry_name):
    # Load model for inference
    try:
        model_name = f"models:/{registry_name}/Staging"
        loaded_model = mlflow.pytorch.load_model(model_name)
        model_loaded = True
    except Exception as e:
        loaded_model = None
        model_loaded = False
        load_error = str(e)
    
    return load_error, loaded_model, model_loaded, model_name


if __name__ == "__main__":
    app.run()

