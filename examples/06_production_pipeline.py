# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "mlflow>=2.17.0",
#     "numpy>=1.26.4",
#     "plotly>=5.24.0",
#     "polars>=1.12.0",
#     "scikit-learn>=1.5.0",
#     "torch>=2.0.0",
# ]
# ///

import marimo
import marimo as mo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """End-to-End Production Pipeline"""

    import warnings
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_wine
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Set modern Plotly theme
    px.defaults.template = "plotly_white"

    warnings.filterwarnings("ignore")

    mo.md("""
    # 🚀 Production Pipeline

    End-to-end ML pipeline from training to deployment using **PINA** (PyTorch-based).

    Builds on all previous examples (00-05). Implements:
    - Data validation and quality checks
    - Neural network training with PINA/PyTorch
    - Performance validation gates
    - Model registration
    - Deployment workflow
    - Production monitoring

    Pipeline stages:
    1. Data validation
    2. Model training (PINA/PyTorch)
    3. Model validation
    4. Model registration
    5. Deployment
    6. Monitoring
    """)
    return (
        Path,
        accuracy_score,
        alt,
        classification_report,
        go,
        load_wine,
        mlflow,
        mo,
        nn,
        np,
        optim,
        pl,
        px,
        StandardScaler,
        torch,
        train_test_split,
        warnings,
    )


@app.cell
def _(mo):
    """Pipeline Configuration"""

    mlflow_uri = mo.ui.text(
        label="🔗 MLflow Tracking URI",
        value="./mlflow",
        placeholder="Enter MLflow server URL",
    )

    experiment_name = mo.ui.text(
        label="🧪 Experiment Name",
        value="production-pipeline",
        placeholder="Enter experiment name",
    )

    model_name = mo.ui.text(
        label="📦 Model Name",
        value="wine_classifier_prod",
        placeholder="model_name",
    )

    mo.md(f"""
    ## ⚙️ Pipeline Configuration

    {mlflow_uri}
    {experiment_name}
    {model_name}
    """)
    return experiment_name, mlflow_uri, model_name


@app.cell
def _(experiment_name, mlflow, mlflow_uri):
    """Initialize MLflow"""

    mlflow.set_tracking_uri(mlflow_uri.value)

    try:
        experiment_id = mlflow.create_experiment(experiment_name.value)
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name.value)
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name.value)

    f"✅ MLflow initialized: {experiment_name.value}"
    return experiment_id,


@app.cell
@mo.cache
def _(load_wine, np, pl):
    """Load & Validate Data"""

    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # Data validation
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))

    # Check for missing values
    has_nan = np.isnan(X).any()

    # Check data ranges
    feature_stats = {
        name: {"min": X[:, i].min(), "max": X[:, i].max(), "mean": X[:, i].mean()}
        for i, name in enumerate(feature_names[:5])  # First 5 features
    }

    validation_results = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "has_nan": has_nan,
        "feature_stats": feature_stats,
        "status": "✅ Valid" if not has_nan else "❌ Invalid",
    }

    df = pl.DataFrame(
        {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
    )

    f"✅ Data loaded: {n_samples} samples, {n_features} features, {n_classes} classes"
    return (
        X,
        data,
        df,
        feature_names,
        feature_stats,
        has_nan,
        n_classes,
        n_features,
        n_samples,
        target_names,
        validation_results,
        y,
    )


@app.cell
def _(mo, validation_results):
    """Display Data Validation"""

    mo.md(f"""
    ## 📊 Data Validation

    - **Samples**: {validation_results['n_samples']}
    - **Features**: {validation_results['n_features']}
    - **Classes**: {validation_results['n_classes']}
    - **Missing Values**: {'❌ Found' if validation_results['has_nan'] else '✅ None'}
    - **Status**: {validation_results['status']}
    """)
    return


@app.cell
def _(X, train_test_split, y):
    """Split Data (Train/Val/Test)"""

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
    return X_temp, X_test, X_train, X_val, y_temp, y_test, y_train, y_val


@app.cell
def _(mo):
    """Model Training Configuration"""

    hidden_size = mo.ui.slider(
        start=32, stop=256, step=16, value=128, label="Hidden layer size"
    )

    num_layers = mo.ui.slider(
        start=1, stop=4, step=1, value=2, label="Number of layers"
    )

    learning_rate = mo.ui.slider(
        start=0.0001, stop=0.01, step=0.0001, value=0.001, label="Learning rate"
    )

    epochs = mo.ui.slider(
        start=50, stop=500, step=50, value=200, label="Training epochs"
    )

    min_accuracy_threshold = mo.ui.slider(
        start=0.7, stop=1.0, step=0.01, value=0.85, label="Min accuracy threshold"
    )

    mo.md(f"""
    ## 🎯 Training Configuration (PINA/PyTorch)

    {hidden_size}
    {num_layers}
    {learning_rate}
    {epochs}
    {min_accuracy_threshold}
    """)
    return epochs, hidden_size, learning_rate, min_accuracy_threshold, num_layers


@app.cell
def _(
    X_train,
    X_val,
    accuracy_score,
    epochs,
    experiment_id,
    hidden_size,
    learning_rate,
    mlflow,
    model_name,
    n_classes,
    n_features,
    nn,
    np,
    num_layers,
    optim,
    StandardScaler,
    torch,
    y_train,
    y_val,
):
    """Train PINA/PyTorch Model with Validation"""

    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)

    # Define neural network model
    class WineClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(WineClassifier, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
            
            layers.append(nn.Linear(hidden_size, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    model = WineClassifier(
        n_features, hidden_size.value, num_layers.value, n_classes
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate.value)

    # Training loop
    model.train()
    for epoch in range(epochs.value):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Validate on validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        _, y_val_pred = torch.max(val_outputs, 1)
        val_accuracy = accuracy_score(y_val, y_val_pred.cpu().numpy())

    # Log training run
    with mlflow.start_run(run_name="training_run") as run:
        mlflow.log_param("hidden_size", hidden_size.value)
        mlflow.log_param("num_layers", num_layers.value)
        mlflow.log_param("learning_rate", learning_rate.value)
        mlflow.log_param("epochs", epochs.value)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("model_name", model_name.value)
        mlflow.set_tag("framework", "pina-pytorch")

        # Log model
        model_uri = mlflow.pytorch.log_model(model, "model")

    f"✅ Model trained! Validation accuracy: {val_accuracy:.4f}"
    return (
        device,
        model,
        model_uri,
        run,
        scaler,
        val_accuracy,
        y_val_pred,
    )


@app.cell
def _(min_accuracy_threshold, mo, val_accuracy):
    """Model Validation Check"""

    passes_threshold = val_accuracy >= min_accuracy_threshold.value

    if passes_threshold:
        validation_status = "✅ PASSED - Model meets accuracy threshold"
        can_deploy = True
    else:
        validation_status = f"❌ FAILED - Accuracy {val_accuracy:.4f} below threshold {min_accuracy_threshold.value:.4f}"
        can_deploy = False

    mo.md(f"""
    ## ✅ Model Validation

    **Validation Accuracy**: {val_accuracy:.4f}
    **Threshold**: {min_accuracy_threshold.value:.4f}
    **Status**: {validation_status}
    """)
    return can_deploy, passes_threshold, validation_status


@app.cell
def _(
    X_test,
    accuracy_score,
    can_deploy,
    classification_report,
    experiment_id,
    mlflow,
    model,
    model_name,
    model_uri,
    mo,
    np,
    scaler,
    torch,
    y_test,
):
    """Register Model (if validation passed)"""

    mo.stop(not can_deploy, "Model validation failed. Adjust parameters and retrain.")

    # Final test evaluation
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, y_test_pred = torch.max(test_outputs, 1)
        y_test_pred = y_test_pred.numpy()
    
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Register model
    with mlflow.start_run(run_name="registration_run") as reg_run:
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.set_tag("stage", "registration")
        mlflow.set_tag("model_name", model_name.value)

        # Register model
        registered_model = mlflow.register_model(
            model_uri.model_uri, model_name.value
        )

    f"✅ Model registered: {model_name.value} v{registered_model.version} (Test accuracy: {test_accuracy:.4f})"
    return reg_run, registered_model, test_accuracy, y_test_pred


@app.cell
def _(mlflow, model_name, mo):
    """Deploy to Production"""

    # Transition to Production
    try:
        # Get latest version
        model_versions = mlflow.search_model_versions(f"name='{model_name.value}'")
        if model_versions:
            latest_version = model_versions[0]
            version = latest_version.version

            # Transition to Production
            mlflow.tracking.MlflowClient().transition_model_version_stage(
                name=model_name.value, version=version, stage="Production"
            )

            deployment_status = f"✅ Deployed: {model_name.value} v{version} to Production"
        else:
            deployment_status = "⚠️ No model versions found"
    except Exception as e:
        deployment_status = f"❌ Deployment error: {str(e)}"

    mo.md(f"""
    ## 🚀 Deployment

    {deployment_status}
    """)
    return deployment_status,


@app.cell
def _(mlflow, model_name):
    """Load Production Model"""

    try:
        # Load production model
        model_uri = f"models:/{model_name.value}/Production"
        production_model = mlflow.pytorch.load_model(model_uri)
        model_loaded = True
        load_error = None
    except Exception:
        # Fallback to latest version
        try:
            model_versions = mlflow.search_model_versions(
                f"name='{model_name.value}'"
            )
            if model_versions:
                version = model_versions[0].version
                model_uri = f"models:/{model_name.value}/{version}"
                production_model = mlflow.pytorch.load_model(model_uri)
                model_loaded = True
                load_error = None
            else:
                production_model = None
                model_loaded = False
                load_error = "No model versions found"
        except Exception as e2:
            production_model = None
            model_loaded = False
            load_error = str(e2)

    return load_error, model_loaded, model_uri, production_model


@app.cell
def _(X_test, mo, np, production_model, scaler, torch, y_test):
    """Model Monitoring - Simulate Production Predictions"""

    if production_model is not None:
        # Simulate production predictions
        X_prod = X_test[:10]  # First 10 samples
        X_prod_scaled = scaler.transform(X_prod)
        X_prod_tensor = torch.FloatTensor(X_prod_scaled)
        
        production_model.eval()
        with torch.no_grad():
            outputs = production_model(X_prod_tensor)
            _, production_predictions = torch.max(outputs, 1)
            production_predictions = production_predictions.numpy()
        
        production_actual = y_test[:10]

        # Calculate metrics
        from sklearn.metrics import accuracy_score

        monitoring_accuracy = accuracy_score(production_actual, production_predictions)

        # Track prediction distribution
        unique, counts = np.unique(production_predictions, return_counts=True)
        prediction_dist = dict(zip(unique, counts))

        monitoring_data = {
            "accuracy": monitoring_accuracy,
            "n_predictions": len(production_predictions),
            "prediction_distribution": prediction_dist,
        }

        mo.md(f"""
        ## 📈 Production Monitoring

        **Recent Predictions**: {monitoring_data['n_predictions']} samples
        **Accuracy**: {monitoring_data['accuracy']:.4f}
        **Prediction Distribution**: {monitoring_data['prediction_distribution']}
        """)
    else:
        monitoring_data = None
        mo.md("## 📈 Production Monitoring\n⚠️ Model not loaded")

    return monitoring_accuracy, monitoring_data, prediction_dist, production_actual, production_predictions


@app.cell
def _(mo, pl, px, production_actual, production_predictions):
    """Monitoring Visualization with Plotly"""

    if production_predictions is not None:
        # Prediction vs Actual comparison
        comparison_data = [
            {"Sample": i, "Actual": int(actual), "Predicted": int(pred)}
            for i, (actual, pred) in enumerate(
                zip(production_actual, production_predictions)
            )
        ]

        comparison_df = pl.DataFrame(comparison_data)

        comparison_fig = px.scatter(
            comparison_df.to_pandas(),
            x="Actual",
            y="Predicted",
            color="Actual",
            title="Predictions vs Actual",
            labels={"Actual": "Actual Class", "Predicted": "Predicted Class"},
            color_discrete_sequence=px.colors.qualitative.Set3,
            width=500,
            height=400,
            opacity=0.7,
        )
        comparison_fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            showlegend=False,
        )
        comparison_fig.update_traces(marker=dict(size=10))

        mo.md(f"""
        ### Predictions Monitoring

        {mo.ui.plotly_chart(comparison_fig)}
        """)
    else:
        mo.md("")
    return comparison_data, comparison_df, comparison_fig


@app.cell
def _(mlflow_uri, model_name, mo):
    """Pipeline Summary & Next Steps"""

    mo.md(f"""
    ## Pipeline Summary

    Completed Stages:
    1. Data Validation - Quality checks passed
    2. Model Training - Trained on training set (PINA/PyTorch)
    3. Model Validation - Validated on validation set
    4. Model Registration - Registered to Model Registry
    5. Deployment - Deployed to Production
    6. Monitoring - Tracking production predictions

    Next Steps:

    1. View Pipeline: [Open MLflow UI]({mlflow_uri.value})
       - Check Experiments tab for training runs
       - Check Models tab for registered models
       - Monitor production metrics

    2. Model Serving: Deploy with MLflow serving
       ```bash
       mlflow models serve -m "models:/{model_name.value}/Production" --port 5001
       ```

    3. Continuous Monitoring:
       - Track prediction accuracy over time
       - Monitor for data drift
       - Set up alerts for performance degradation

    4. Model Updates:
       - Retrain with new data
       - Validate new model version
       - A/B test before full deployment
       - Promote to production when validated

    Best Practices Demonstrated:
    - Train/Val/Test Split: Proper data splitting
    - Validation Thresholds: Quality gates before deployment
    - Model Registry: Centralized model management
    - Staging Workflow: Staging → Production promotion
    - Production Monitoring: Track real-world performance
    - PINA/PyTorch: Deep learning with scientific ML framework
    """)
    return


if __name__ == "__main__":
    app.run()

