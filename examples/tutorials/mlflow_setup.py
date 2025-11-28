import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# MLflow Experiment Tracking Setup""")
    return


@app.cell
def _():
    """Import MLflow and dependencies"""
    import os
    from pathlib import Path

    import mlflow
    import mlflow.pytorch
    import mlflow.sklearn
    import numpy as np
    import polars as pl
    return mlflow, pl


@app.cell
def _():
    
    import mlflow

    # local development
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    with mlflow.start_run():
        mlflow.log_metrics({'accuracy': 0.95})


@app.cell
def _(mo):
    """MLflow configuration UI"""
    tracking_uri = mo.ui.text(
        label="📂 Tracking URI",
        value="file:///./mlruns",
        placeholder="file:///path/to/mlruns or http://localhost:5000"
    )

    experiment_name = mo.ui.text(
        label="🔬 Experiment Name",
        value="marimo_experiments",
        placeholder="my_experiment"
    )

    return experiment_name, tracking_uri


@app.cell
def _(experiment_name, mo, tracking_uri):
    """Display configuration"""
    mo.md(f"""
    ## ⚙️ MLflow Configuration

    {tracking_uri}
    {experiment_name}
    """)
    return


@app.cell
def _(experiment_name, mlflow, tracking_uri):
    """Configure MLflow"""
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri.value)

    # Create or set experiment
    try:
        experiment = mlflow.set_experiment(experiment_name.value)
        experiment_id = experiment.experiment_id
        print(f"✅ Experiment '{experiment_name.value}' (ID: {experiment_id})")
    except Exception as e:
        print(f"❌ Error setting experiment: {e}")
        experiment_id = None

    return experiment, experiment_id


@app.cell
def _(experiment_id, mlflow, mo):
    """Display experiment info"""
    if experiment_id:
        exp_info = mlflow.get_experiment(experiment_id)

        mo.output.append(mo.md(f"""
        ### 📊 Experiment Details

        - **Name:** {exp_info.name}
        - **ID:** {exp_info.experiment_id}
        - **Artifact Location:** {exp_info.artifact_location}
        - **Lifecycle Stage:** {exp_info.lifecycle_stage}
        """))
    else:
        mo.md("### ❌ No experiment configured")

    return


@app.cell
def _(mo):
    """Run configuration"""
    run_name = mo.ui.text(
        label="🏃 Run Name",
        value="test_run",
        placeholder="my_model_run"
    )

    tags = mo.ui.text_area(
        label="🏷️ Tags (key=value, one per line)",
        value="model_type=random_forest\nframework=sklearn",
        placeholder="key1=value1\nkey2=value2"
    )

    return run_name, tags


@app.cell
def _(mo, run_name, tags):
    """Display run configuration"""
    mo.md(f"""
    ## 🚀 Run Configuration

    {run_name}
    {tags}
    """)
    return


@app.cell
def _(tags):
    """Parse tags"""
    def parse_tags(tag_string):
        """Parse tags from string format"""
        tag_dict = {}
        for line in tag_string.value.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                tag_dict[key.strip()] = value.strip()
        return tag_dict

    parsed_tags = parse_tags(tags)
    print(f"✅ Parsed {len(parsed_tags)} tags: {list(parsed_tags.keys())}")

    return (parsed_tags,)


@app.cell
def _(mo):
    """Demo run button"""
    demo_button = mo.ui.run_button(label="🎯 Run MLflow Demo")

    mo.md(f"""
    ## 🧪 MLflow Demo

    Click to run a demo MLflow tracking example:

    {demo_button}
    """)

    return (demo_button,)


@app.cell
def _(demo_button, mlflow, mo, parsed_tags, pl, run_name):
    """Run MLflow demo"""
    mo.stop(not demo_button.value, "Click 'Run MLflow Demo' to start")

    # Import demo dependencies
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name.value, tags=parsed_tags) as run:
        # Log parameters
        n_estimators = 100
        max_depth = 5

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", 42)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log artifact (feature importances)
        feature_importance = pl.DataFrame({
            'feature': [f'feature_{i}' for i in range(10)],
            'importance': model.feature_importances_
        }).sort('importance', descending=True)

        # Save and log artifact
        feature_importance.write_csv('/tmp/feature_importance.csv')
        mlflow.log_artifact('/tmp/feature_importance.csv')

        run_id = run.info.run_id

    print(f"✅ MLflow run completed!")
    print(f"📊 Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    print(f"🔗 Run ID: {run_id}")

    return


@app.cell
def _(experiment, mlflow, mo):
    """List recent runs"""
    try:
        # Get experiment


        if experiment:
            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=10,
                order_by=["start_time DESC"]
            )

            if not runs.empty:
                # Select columns to display
                display_cols = ['run_id', 'status', 'start_time', 'tags.model_type', 
                               'metrics.accuracy', 'metrics.f1_score']
                available_cols = [col for col in display_cols if col in runs.columns]

                mo.md(f"""
                ## 📋 Recent Runs

                {mo.ui.table(runs[available_cols])}
                """)
            else:
                mo.md("## 📋 No runs found")
        else:
            mo.md("## 📋 Experiment not found")

    except Exception as e:
        mo.md(f"## ❌ Error listing runs: {e}")

    return


@app.cell
def _(mo):
    """MLflow UI instructions"""
    mo.md("""
    ## 🖥️ MLflow UI

    ### Starting the MLflow UI

    To view your experiments in the MLflow web interface:

    ```bash
    # Navigate to your project directory
    cd /path/to/your/project

    # Start MLflow UI
    mlflow ui --host 0.0.0.0
    ```

    Then open http://localhost:5000 in your browser.

    ### Features Available in MLflow UI

    - **Experiment Comparison**: Compare multiple runs side-by-side
    - **Metric Visualization**: Plot metrics over time
    - **Artifact Browser**: View logged files and models
    - **Model Registry**: Register and version models
    - **Run Details**: Inspect parameters, metrics, and artifacts

    ### Best Practices

    1. **Consistent Naming**: Use descriptive experiment and run names
    2. **Tag Everything**: Add relevant tags for easy filtering
    3. **Log Artifacts**: Save plots, data samples, and model files
    4. **Version Control**: Track code versions with each run
    5. **Clean Up**: Archive old experiments to keep UI responsive
    """)
    return


if __name__ == "__main__":
    app.run()
