# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "mlflow>=2.17.0",
#     "numpy>=1.26.4",
#     "plotly>=5.24.0",
#     "polars>=1.12.0",
#     "pina>=0.1.0",
#     "torch>=2.0.0",
# ]
# ///

import marimo
import marimo as mo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """Model Registry & Deployment with MLflow (PINA/PyTorch)"""

    import warnings

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
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    warnings.filterwarnings("ignore")

    mo.md("""
    # 📦 Model Registry & Deployment (PINA/PyTorch)

    Model lifecycle management using MLflow Model Registry.

    Builds on previous examples:
    - Uses PINA/PyTorch models from 02_basic_ml_workflow.py, 03_model_comparison.py, and 04_hyperparameter_tuning.py
    - Registers models for deployment
    - Manages model versions and stages (Staging → Production)
    - Demonstrates model loading and inference

    Includes:
    - Model registration and versioning
    - Stage transitions (Staging → Production)
    - Model search and discovery
    - Model loading for inference
    """)
    return (
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
    """Configuration"""

    mlflow_uri = mo.ui.text(
        label="MLflow Tracking URI",
        value="./mlflow",
        placeholder="Enter MLflow server URL",
    )

    registry_name = mo.ui.text(
        label="Model Name",
        value="wine_classifier",
        placeholder="model_name",
    )

    mo.md(f"""
    ## ⚙️ Configuration

    {mlflow_uri}
    {registry_name}
    """)
    return mlflow_uri, registry_name


@app.cell
def _(mlflow, mlflow_uri):
    """Initialize MLflow"""

    mlflow.set_tracking_uri(mlflow_uri.value)

    f"MLflow initialized"
    return


@app.cell
@mo.cache
def _(load_wine, pl):
    """Load Dataset"""

    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    df = pl.DataFrame(
        {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
    )

    return X, data, df, feature_names, target_names, y


@app.cell
def _(X, train_test_split, y):
    """Split Data"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    """Model Training & Registration"""

    train_new_model = mo.ui.button(
        value=False, label="🚀 Train & Register New Model"
    )

    model_params = mo.ui.form(
        {
            "hidden_size": mo.ui.slider(32, 256, value=128, step=16),
            "num_layers": mo.ui.slider(1, 4, value=2, step=1),
            "learning_rate": mo.ui.slider(0.0001, 0.01, value=0.001, step=0.0001),
            "epochs": mo.ui.slider(50, 500, value=200, step=50),
            "description": mo.ui.text_area(
                placeholder="Model description...", value="PINA/PyTorch Neural Network for wine dataset"
            ),
        }
    )

    mo.md(f"""
    ## 🎯 Train & Register Model (PINA/PyTorch)

    {model_params}
    {train_new_model}
    """)
    return model_params, train_new_model


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    mlflow,
    model_params,
    nn,
    np,
    optim,
    registry_name,
    StandardScaler,
    torch,
    train_new_model,
    y_test,
    y_train,
):
    """Train and Register PINA/PyTorch Model"""

    mo.stop(not train_new_model.value, "Click 'Train & Register New Model' to proceed")

    # Get parameters
    params = model_params.value
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    description = params["description"]

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    # Define neural network model
    class Classifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(Classifier, self).__init__()
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

    # Create and train model
    model = Classifier(n_features, hidden_size, num_layers, n_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y_test, y_pred)

    # Register model
    with mlflow.start_run(run_name=f"{registry_name.value}_v1") as run:
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("model_type", "pina-pytorch")
        mlflow.set_tag("framework", "pina-pytorch")
        mlflow.set_tag("dataset", "wine")
        mlflow.set_tag("description", description)

        # Log and register model
        model_uri = mlflow.pytorch.log_model(
            model, "model", registered_model_name=registry_name.value
        )

        registered_model = mlflow.register_model(
            model_uri.model_uri, registry_name.value
        )

    f"✅ Model registered: {registered_model.name} (Version {registered_model.version})"
    return accuracy, model, model_uri, registered_model, run, y_pred


@app.cell
def _(mlflow, registry_name):
    """List Registered Models"""

    try:
        registered_models = mlflow.search_registered_models(
            filter_string=f"name='{registry_name.value}'"
        )

        if registered_models:
            model_info = []
            for model in registered_models:
                latest_version = model.latest_versions[0] if model.latest_versions else None
                if latest_version:
                    model_info.append(
                        {
                            "Name": model.name,
                            "Latest Version": latest_version.version,
                            "Stage": latest_version.current_stage,
                            "Created": str(latest_version.creation_timestamp)[:19],
                        }
                    )

            models_df = pl.DataFrame(model_info) if model_info else None
        else:
            models_df = None
    except Exception as e:
        models_df = None
        error_msg = str(e)

    return error_msg, model_info, models_df, models_info, registered_models


@app.cell
def _(mo, models_df):
    """Display Registered Models"""

    if models_df is not None:
        mo.md(f"""
        ## 📚 Registered Models

        {mo.ui.table(models_df)}
        """)
    else:
        mo.md("## 📚 No registered models found")
    return


@app.cell
def _(mlflow, registry_name):
    """Get Model Versions"""

    try:
        model_versions = mlflow.search_model_versions(f"name='{registry_name.value}'")

        versions_data = []
        for version in model_versions:
            versions_data.append(
                {
                    "Version": version.version,
                    "Stage": version.current_stage,
                    "Created": str(version.creation_timestamp)[:19],
                    "Run ID": version.run_id,
                }
            )

        versions_df = pl.DataFrame(versions_data).sort("Version", descending=True) if versions_data else None
    except Exception as e:
        versions_df = None
        version_error = str(e)

    return version_error, versions_data, versions_df, model_versions


@app.cell
def _(mo, versions_df):
    """Display Model Versions"""

    if versions_df is not None:
        mo.md(f"""
        ## 🔄 Model Versions

        {mo.ui.table(versions_df)}
        """)
    else:
        mo.md("## 🔄 No versions found")
    return


@app.cell
def _(mo):
    """Model Stage Management"""

    version_to_stage = mo.ui.number(
        start=1, stop=10, step=1, value=1, label="Version to stage"
    )

    target_stage = mo.ui.dropdown(
        options=["None", "Staging", "Production", "Archived"],
        value="Staging",
        label="Target stage",
    )

    transition_button = mo.ui.button(
        value=False, label="🔄 Transition Model Stage"
    )

    mo.md(f"""
    ## 🚀 Model Stage Management

    {version_to_stage}
    {target_stage}
    {transition_button}
    """)
    return target_stage, transition_button, version_to_stage


@app.cell
def _(
    mlflow,
    registry_name,
    target_stage,
    transition_button,
    version_to_stage,
):
    """Transition Model Stage"""

    mo.stop(not transition_button.value, "Click 'Transition Model Stage' to proceed")

    try:
        version = str(int(version_to_stage.value))
        stage = target_stage.value if target_stage.value != "None" else None

        if stage:
            mlflow.tracking.MlflowClient().transition_model_version_stage(
                name=registry_name.value, version=version, stage=stage
            )
            result = f"✅ Model {registry_name.value} v{version} transitioned to {stage}"
        else:
            result = "⚠️ Select a valid stage"
    except Exception as e:
        result = f"❌ Error: {str(e)}"

    result
    return result,


@app.cell
def _(mlflow, registry_name):
    """Load Model for Inference"""

    try:
        # Get latest production model, or latest version if no production
        model_versions = mlflow.search_model_versions(f"name='{registry_name.value}'")

        # Try to find production model first
        production_model = None
        for version in model_versions:
            if version.current_stage == "Production":
                production_model = version
                break

        # If no production, use latest version
        if not production_model and model_versions:
            production_model = model_versions[0]

        if production_model:
            model_uri = f"models:/{registry_name.value}/{production_model.current_stage or production_model.version}"
            loaded_model = mlflow.sklearn.load_model(model_uri)
            model_info = f"✅ Loaded: {registry_name.value} v{production_model.version} ({production_model.current_stage or 'None'})"
        else:
            loaded_model = None
            model_info = "⚠️ No model found"
    except Exception as e:
        loaded_model = None
        model_info = f"❌ Error loading model: {str(e)}"

    return loaded_model, model_info, model_uri, production_model


@app.cell
def _(X_test, loaded_model, model_info, mo, y_test):
    """Test Loaded Model"""

    if loaded_model is not None:
        # Make predictions
        predictions = loaded_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)

        mo.md(f"""
        ## 🧪 Model Inference Test

        {model_info}

        **Test Accuracy**: {test_accuracy:.4f}

        Model successfully loaded and tested!
        """)
    else:
        mo.md(f"""
        ## 🧪 Model Inference Test

        {model_info}
        """)
    return predictions, test_accuracy


@app.cell
def _(go, mo, predictions, y_test):
    """Prediction Visualization with Plotly"""

    if predictions is not None:
        # Confusion matrix with Plotly
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_test, predictions)
        
        confusion_fig = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Pred {i}" for i in range(len(cm))],
                y=[f"True {i}" for i in range(len(cm))],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
                colorbar=dict(title="Count"),
            )
        )
        confusion_fig.update_layout(
            title="Confusion Matrix",
            template="plotly_white",
            width=500,
            height=500,
            font=dict(size=12),
        )

        mo.md(f"""
        ### Predictions Visualization

        {mo.ui.plotly_chart(confusion_fig)}
        """)
    else:
        mo.md("")
    return cm, confusion_fig


@app.cell
def _(mlflow_uri, mo):
    """Next Steps"""

    mo.md(f"""
    ## Next Steps

    1. View Registry: [Open MLflow UI]({mlflow_uri.value}) → Models tab
    2. Production Pipeline: Use 06_production_pipeline.py for automated deployment
    3. Model Serving: Deploy registered models with MLflow serving

    Model Registry Workflow:
    1. Register: Train and register models (this notebook)
    2. Stage: Transition to Staging for testing
    3. Promote: Move to Production when validated
    4. Monitor: Track model performance in production
    5. Update: Register new versions as needed

    MLflow Model Registry Features:
    - Version Control: Automatic versioning
    - Stage Management: Staging → Production workflow
    - Model Lineage: Track which run created each version
    - Annotations: Add descriptions and tags
    - Model Serving: REST API for inference
    """)
    return


if __name__ == "__main__":
    app.run()

