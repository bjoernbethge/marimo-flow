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
        # MLflow: Setup & Basic Tracking
        
        This snippet shows how to set up MLflow tracking, create or get experiments,
        and log parameters, metrics, and artifacts in a run.
        """
    )
    return


@app.cell
def _():
    import mlflow
    import mlflow.pytorch
    from pathlib import Path
    
    # Setup MLflow tracking URI (file-based or server)
    tracking_path = Path("./data/experiments")
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_path.resolve()}")
    
    # Alternative: Use MLflow server
    # mlflow.set_tracking_uri("http://localhost:5000")
    
    return Path, mlflow, tracking_path


@app.cell
def _(mlflow):
    # Create or get experiment
    experiment_name = "my_experiment"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    return experiment, experiment_id, experiment_name


@app.cell
def _(mlflow):
    # Start a run and log data
    with mlflow.start_run(run_name="example_run") as run:
        # Log parameters
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("epochs", 100)
        mlflow.log_param("batch_size", 32)
        
        # Log metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.05)
        mlflow.log_metric("f1_score", 0.92)
        
        # Log tags
        mlflow.set_tag("model_type", "neural_network")
        mlflow.set_tag("framework", "pytorch")
        
        # Log artifact (example: save a file)
        from tempfile import NamedTemporaryFile
        import json
        
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"config": "example"}, f)
            artifact_path = f.name
        
        mlflow.log_artifact(artifact_path, "configs")
        
        run_id = run.info.run_id
    
    return artifact_path, f, json, run, run_id


if __name__ == "__main__":
    app.run()

