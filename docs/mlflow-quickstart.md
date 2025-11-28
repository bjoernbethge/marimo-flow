# MLflow Reference - Quickstart & Core Concepts

**Last Updated**: 2025-11-25
**Source Version**: MLflow 2.0+
**Status**: Production Ready

## What is MLflow?

MLflow is an open-source platform for managing the machine learning lifecycle, from experimentation through deployment. It provides tools for:

- **Experiment Tracking**: Log parameters, metrics, and artifacts
- **Model Registry**: Manage and version ML models
- **Model Serving**: Deploy models to production
- **Project Management**: Package ML projects with dependencies

Key features:
- **Language Agnostic**: Works with any ML framework (scikit-learn, PyTorch, TensorFlow, etc.)
- **Framework Neutral**: Log metrics from any training framework
- **Production Ready**: Deploy to various environments
- **Git Integration**: Track code alongside experiments

## Installation & Setup

### Basic Installation

```bash
# Install MLflow
pip install mlflow

# Install with optional dependencies
pip install "mlflow[extras]"

# Using uv
uv add mlflow

# Using conda
conda install -c conda-forge mlflow
```

### Recommended Setup for marimo-flow

```bash
# Install MLflow with all useful dependencies
pip install mlflow[extras] scikit-learn pandas numpy
```

### Starting the MLflow Server

```bash
# Start MLflow UI locally (default: http://localhost:5000)
mlflow ui

# With custom host/port
mlflow ui --host 0.0.0.0 --port 8080

# With backend store (persists to database)
mlflow ui --backend-store-uri sqlite:///mlflow.db

# With artifact store location
mlflow ui --artifacts-destination ./artifacts
```

## Core Concepts

### 1. Experiments & Runs

An **Experiment** is a collection of related runs. A **Run** is a single execution of your training code.

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Set experiment
mlflow.set_experiment("my_classification_model")

# Start a run
with mlflow.start_run():
    # Your training code
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Log parameters (hyperparameters)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Log metrics (performance indicators)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

**Key Points:**
- Each experiment has a unique name
- Each run has a unique ID
- Runs are organized chronologically
- Parameters are logged once, metrics can be updated

### 2. Parameters vs. Metrics

**Parameters**: Configuration values (hyperparameters, feature selections)
```python
mlflow.log_param("learning_rate", 0.001)
mlflow.log_param("batch_size", 32)
mlflow.log_param("optimizer", "adam")
```

**Metrics**: Performance measurements (loss, accuracy, F1-score)
```python
mlflow.log_metric("train_loss", 0.342)
mlflow.log_metric("val_accuracy", 0.92)
mlflow.log_metric("test_f1", 0.89)
```

### 3. Artifacts

Artifacts are files associated with a run (models, plots, data, etc.)

```python
import mlflow
import matplotlib.pyplot as plt

with mlflow.start_run():
    # Train model...

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log plot
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.savefig("training_curve.png")
    mlflow.log_artifact("training_curve.png")

    # Log text file
    with open("summary.txt", "w") as f:
        f.write("Model training complete")
    mlflow.log_artifact("summary.txt")

    # Log entire directory
    mlflow.log_artifacts("./plots")
```

### 4. Model Registry

The Model Registry is a central place to manage model versions and transitions.

```python
import mlflow

# Register a model after training
mlflow.register_model(
    model_uri="runs:/1234567890/model",
    name="my_production_model"
)

# Transition model to different stages
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Move to staging
client.transition_model_version_stage(
    name="my_production_model",
    version=1,
    stage="Staging"
)

# Promote to production
client.transition_model_version_stage(
    name="my_production_model",
    version=1,
    stage="Production"
)
```

**Model Stages:**
- **None**: Initial stage (experimental)
- **Staging**: Pre-production testing
- **Production**: Active production model
- **Archived**: Retired/no longer used

## Common Patterns

### Pattern: Complete ML Workflow with Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd

# Load data
df = pd.read_csv("data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Set experiment
mlflow.set_experiment("gradient_boosting_experiments")

# Hyperparameters to try
params_list = [
    {"n_estimators": 50, "learning_rate": 0.1},
    {"n_estimators": 100, "learning_rate": 0.05},
    {"n_estimators": 200, "learning_rate": 0.01},
]

# Run experiments
for params in params_list:
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)

        # Train model
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print(f"Logged run with params: {params}")
```

### Pattern: Deep Learning with PyTorch

```python
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Training loop
mlflow.set_experiment("pytorch_experiments")

with mlflow.start_run():
    # Hyperparameters
    lr = 0.001
    epochs = 10
    batch_size = 32

    mlflow.log_params({
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size
    })

    # Initialize model
    model = NeuralNet(784, 128, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Log metrics every N batches
            if batch_idx % 100 == 0:
                mlflow.log_metric("loss", loss.item(), step=epoch * len(train_loader) + batch_idx)

    # Log model
    mlflow.pytorch.log_model(model, "model")
```

### Pattern: Hyperparameter Tuning with Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

mlflow.set_experiment("hyperparameter_tuning")

# Define parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10]
}

# GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5)

with mlflow.start_run():
    # Run grid search
    grid_search.fit(X_train, y_train)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Log best score
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metric("test_accuracy", grid_search.score(X_test, y_test))

    # Log best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")

    print(f"Best params: {grid_search.best_params_}")
```

### Pattern: Tracking with Custom Metrics

```python
import mlflow
from sklearn.metrics import confusion_matrix, classification_report
import json

mlflow.set_experiment("custom_metrics")

with mlflow.start_run():
    # Train model...

    # Standard metrics
    mlflow.log_metric("accuracy", 0.95)

    # Custom metrics from confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_metric("tn", cm[0, 0])
    mlflow.log_metric("fp", cm[0, 1])
    mlflow.log_metric("fn", cm[1, 0])
    mlflow.log_metric("tp", cm[1, 1])

    # Calculated metrics
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("sensitivity", sensitivity)

    # Log classification report as artifact
    report = classification_report(y_test, y_pred, output_dict=True)
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("classification_report.json")
```

### Pattern: Model Serving & Loading

```python
import mlflow
from mlflow.pyfunc import load_model

# Load model from registry
model = mlflow.pyfunc.load_model("models:/my_model/Production")

# Or load from specific run
model = mlflow.pyfunc.load_model("runs:/run_id/model")

# Make predictions
predictions = model.predict(X_new)
```

## Integration with Marimo

MLflow integrates seamlessly with marimo notebooks for interactive experiment tracking:

### Pattern: Interactive MLflow Dashboard in Marimo

```python
import marimo as mo
import mlflow
import pandas as pd

# Connect to MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment information
experiment = mlflow.get_experiment_by_name("my_experiment")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Create interactive selector
selected_run = mo.ui.dropdown(
    [f"{row['run_id'][:8]} - {row['params.model_type']}" for _, row in runs.iterrows()],
    label="Select run:"
)

# Display run details
if selected_run.value:
    run_idx = int(selected_run.value.split()[0])
    run_id = runs.iloc[run_idx]["run_id"]
    run = mlflow.get_run(run_id)

    mo.md(f"""
    ## Run Details
    - **Run ID**: {run.info.run_id}
    - **Status**: {run.info.status}
    - **Duration**: {run.info.end_time - run.info.start_time}ms
    """)

    # Display metrics
    metrics_df = pd.DataFrame([
        {"metric": k, "value": v}
        for k, v in run.data.metrics.items()
    ])
    mo.md(metrics_df.to_markdown())
```

### Pattern: Real-time Metric Tracking in Marimo

```python
import marimo as mo
import mlflow

# Interactive training with live updates
get_epoch, set_epoch = mo.state(0)
get_loss, set_loss = mo.state([])

mlflow.set_experiment("live_training")

with mlflow.start_run():
    for epoch in range(10):
        # Training step
        loss = train_step()

        # Update state
        set_epoch(epoch + 1)
        losses = get_loss()
        losses.append(loss)
        set_loss(losses)

        # Log to MLflow
        mlflow.log_metric("loss", loss, step=epoch)

        # Display in marimo
        mo.md(f"Epoch: {get_epoch()}/10")
        mo.md(f"Loss: {get_loss()[-1]:.4f}")

    mlflow.sklearn.log_model(model, "model")
```

### Pattern: Experiment Comparison Dashboard

```python
import marimo as mo
import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://localhost:5000")

# Get all runs for an experiment
experiment_name = mo.ui.text(
    label="Experiment name:",
    value="my_experiment"
)

if experiment_name.value:
    experiment = mlflow.get_experiment_by_name(experiment_name.value)
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        # Create comparison table
        comparison_data = []
        for _, row in runs.iterrows():
            comparison_data.append({
                "Run ID": row["run_id"][:8],
                "Status": row["status"],
                "Accuracy": row.get("metrics.accuracy", "-"),
                "Loss": row.get("metrics.loss", "-"),
                "Model Type": row.get("params.model_type", "-")
            })

        comparison_df = pd.DataFrame(comparison_data)
        mo.md(comparison_df.to_markdown())

        # Select best model
        best_run = runs.loc[runs["metrics.accuracy"].idxmax()]
        mo.md(f"**Best Model**: {best_run['run_id'][:8]} (accuracy: {best_run['metrics.accuracy']:.4f})")
```

## Best Practices

### ✅ DO: Use Experiments for Organization

```python
# GOOD - clear experiment naming
mlflow.set_experiment("production/classification_v2")

# or by date
import datetime
date = datetime.date.today().isoformat()
mlflow.set_experiment(f"exp/{date}/hyperparameter_search")
```

### ✅ DO: Log Both Parameters and Metrics

```python
# GOOD - comprehensive logging
with mlflow.start_run():
    # Configuration
    mlflow.log_params(model_config)

    # Training progress
    for epoch in range(10):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Final results
    mlflow.log_metrics(final_metrics)
```

### ✅ DO: Use Context Manager for Automatic Cleanup

```python
# GOOD - handles run cleanup automatically
with mlflow.start_run():
    # Training code
    pass
# Run is automatically ended
```

### ❌ DON'T: Log Too Many Metrics

```python
# BAD - excessive logging slows down tracking
for i in range(1000):
    mlflow.log_metric("internal_state_" + str(i), value)

# GOOD - aggregate or sample
if i % 100 == 0:
    mlflow.log_metric("loss", loss, step=i)
```

### ✅ DO: Version Your Data & Code

```python
# GOOD - include references
mlflow.log_param("data_version", "2.1.0")
mlflow.log_param("training_script_version", "v3.2")
mlflow.log_artifact("requirements.txt")
```

### ✅ DO: Use Model Registry for Production Models

```python
# GOOD - manage versions in registry
mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="prod_model")

# Later: promote to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="prod_model",
    version=1,
    stage="Production"
)
```

## API Reference - Quick Lookup

### Experiment Management

```python
mlflow.set_experiment("experiment_name")          # Set active experiment
mlflow.get_experiment_by_name("name")             # Get experiment details
mlflow.search_experiments()                       # List all experiments
mlflow.create_experiment("name", artifact_location="./path")  # Create new
```

### Run Management

```python
mlflow.start_run()                                # Start a new run
mlflow.end_run()                                  # End current run
mlflow.active_run()                               # Get active run
mlflow.search_runs()                              # Search runs
mlflow.get_run(run_id)                            # Get run details
```

### Logging

```python
mlflow.log_param(key, value)                      # Log single parameter
mlflow.log_params(dict)                           # Log multiple parameters
mlflow.log_metric(key, value, step=None)          # Log metric
mlflow.log_metrics(dict)                          # Log multiple metrics
mlflow.log_artifact(path)                         # Log file
mlflow.log_artifacts(path)                        # Log directory
```

### Model Logging (Framework-specific)

```python
mlflow.sklearn.log_model(model, "model")          # scikit-learn
mlflow.pytorch.log_model(model, "model")          # PyTorch
mlflow.tensorflow.log_model(model, "model")       # TensorFlow
mlflow.xgboost.log_model(model, "model")          # XGBoost
mlflow.pyfunc.log_model(model, "model")           # Custom/Generic
```

### Model Registry

```python
mlflow.register_model(model_uri, name)            # Register model
mlflow.get_registered_model(name)                 # Get model info
client.transition_model_version_stage(            # Change model stage
    name, version, stage
)
mlflow.pyfunc.load_model(model_uri)               # Load model for inference
```

## Common Issues & Solutions

### Issue: Metrics Not Showing in UI

**Problem**: Logged metrics don't appear in MLflow UI.

**Solution**: Make sure you're using `mlflow.log_metric()` or `mlflow.log_metrics()` instead of just printing.

```python
# Good
mlflow.log_metric("accuracy", 0.95)

# Bad - this just prints
print("Accuracy: 0.95")
```

### Issue: Model Can't Be Loaded

**Problem**: `mlflow.pyfunc.load_model()` fails.

**Solution**: Ensure the model was logged with the correct flavor and the artifact directory is accessible.

```python
# Make sure model was logged
mlflow.sklearn.log_model(model, "model")

# Then load with run_id
mlflow.pyfunc.load_model("runs:/run_id/model")
```

### Issue: Database Locked

**Problem**: SQLite backend gives "database is locked" error.

**Solution**: Use a proper database backend for production instead of SQLite.

```bash
# Use PostgreSQL for production
mlflow server --backend-store-uri postgresql://user:pass@localhost/mlflow
```

## Additional Resources

- **Official Docs**: https://mlflow.org/docs/latest/index.html
- **GitHub**: https://github.com/mlflow/mlflow
- **Tracking Server Setup**: https://mlflow.org/docs/latest/tracking/server.html
- **Model Registry**: https://mlflow.org/docs/latest/model-registry.html
- **MLflow Examples**: https://github.com/mlflow/mlflow/tree/master/examples

## Integration with marimo-flow Ecosystem

MLflow integrates with other marimo-flow tools:

- **Marimo Notebooks**: Interactive experiment tracking dashboards
- **Polars DataFrames**: Efficient data handling for experiments
- **PINA Models**: Track physics-informed neural network experiments
- **OpenVINO Inference**: Log optimized models and performance metrics

For complete integration examples, see the main documentation and example notebooks in `examples/`.
