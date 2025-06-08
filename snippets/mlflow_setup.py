import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MLflow Experiment Setup""")
    return

@app.cell
def _(mo):
    """MLflow Configuration UI"""
    mlflow_uri = mo.ui.text(
        label="🔗 MLflow Tracking URI",
        value="http://localhost:5000"
    )
    
    experiment_name = mo.ui.text(
        label="🧪 Experiment Name", 
        value="my-ml-experiment"
    )
    
    mo.md(f"""
    ## ⚙️ MLflow Configuration
    {mlflow_uri}
    {experiment_name}
    """)
    return experiment_name, mlflow_uri

@app.cell
def _(experiment_name, mlflow_uri):
    """Initialize MLflow Tracking"""
    import mlflow
    
    mlflow.set_tracking_uri(mlflow_uri.value)
    
    try:
        experiment_id = mlflow.create_experiment(experiment_name.value)
        status = f"✅ Created new experiment: {experiment_name.value}"
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name.value)
        experiment_id = experiment.experiment_id
        status = f"✅ Using existing experiment: {experiment_name.value}"
    
    mlflow.set_experiment(experiment_name.value)
    print(status)
    
    return experiment_id, mlflow, status

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run() 