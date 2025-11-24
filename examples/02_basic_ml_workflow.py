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
    """Basic ML Workflow with Marimo Flow (PINA/PyTorch)"""

    import marimo as mo
    import mlflow
    import mlflow.pytorch
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import altair as alt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from pathlib import Path
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.datasets import make_classification, load_wine
    from sklearn.preprocessing import StandardScaler
    import warnings

    # Set modern Plotly theme
    px.defaults.template = "plotly_white"
    px.defaults.color_continuous_scale = "viridis"

    warnings.filterwarnings("ignore")

    mo.md("""
    # 🚀 Basic ML Workflow (PINA/PyTorch)

    **End-to-end ML experiment with MLflow tracking using PINA/PyTorch**

    Workflow:
    - 📊 **Data Generation**: Synthetic classification dataset
    - 🎯 **Model Training**: Neural Network (PINA/PyTorch)
    - 🔬 **MLflow Tracking**: Experiment logging
    - 📈 **Visualization**: Results and metrics
    """)
    return (
        Path,
        accuracy_score,
        alt,
        go,
        load_wine,
        make_classification,
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
    )


@app.cell
def _(mo):
    """MLflow Configuration"""

    mlflow_uri = mo.ui.text(
        label="🔗 MLflow Tracking URI",
        value="./mlflow",
        placeholder="Enter MLflow server URL",
    )

    experiment_name = mo.ui.text(
        label="🧪 Experiment Name",
        value="basic-ml-workflow",
        placeholder="Enter experiment name",
    )

    mo.md(f"""
    ## ⚙️ MLflow Configuration
    {mlflow_uri}
    {experiment_name}
    """)
    return experiment_name, mlflow_uri


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
    return


@app.cell
def _(mo):
    """Data Source Selection"""

    data_source = mo.ui.dropdown(
        options=["builtin", "file", "synthetic"],
        value="builtin",
        label="📊 Data Source",
    )

    dataset_choice = mo.ui.dropdown(
        options=["wine"],
        value="wine",
        label="📦 Built-in Dataset",
    )

    file_upload = mo.ui.file(
        filetypes=[".csv", ".parquet", ".json"],
        max_size=100 * 1024 * 1024,
        label="📁 Upload File",
    )

    # Synthetic data parameters (only shown if synthetic selected)
    n_samples = mo.ui.slider(
        start=500, stop=2000, step=100, value=1000, label="Number of samples"
    )

    n_features = mo.ui.slider(
        start=10, stop=50, step=5, value=20, label="Number of features"
    )

    random_state = mo.ui.number(
        start=0, stop=100, step=1, value=42, label="Random state"
    )

    mo.md(f"""
    ## 📊 Dataset Configuration
    
    {data_source}
    
    {dataset_choice if data_source.value == "builtin" else (file_upload if data_source.value == "file" else mo.vstack([n_samples, n_features, random_state]))}
    """)
    return data_source, dataset_choice, file_upload, n_features, n_samples, random_state


@app.cell
def _(
    Path,
    data_source,
    dataset_choice,
    file_upload,
    load_wine,
    make_classification,
    mo,
    n_features,
    n_samples,
    np,
    pl,
    random_state,
):
    """Load Dataset - Cached for performance"""

    if data_source.value == "file" and file_upload.value:
        # Load from file
        file_path = Path(file_upload.value)
        
        if file_path.suffix == ".csv":
            df = pl.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df = pl.read_parquet(file_path)
        elif file_path.suffix == ".json":
            df = pl.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Try to identify target column
        target_col = None
        for col in ["target", "label", "y", "class"]:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = df.columns[-1]
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col]
        X = df.select(feature_cols).to_numpy()
        y = df[target_col].to_numpy()
        feature_names = feature_cols
        
    elif data_source.value == "synthetic":
        # Generate synthetic data
        X, y = make_classification(
            n_samples=n_samples.value,
            n_features=n_features.value,
            n_classes=2,
            n_redundant=2,
            n_informative=n_features.value - 2,
            random_state=random_state.value,
            n_clusters_per_class=1,
        )
        feature_names = [f"feature_{i}" for i in range(n_features.value)]
        df = pl.DataFrame(
            {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
        )
    else:
        # Load built-in dataset
        data = load_wine()
        X, y = data.data, data.target
        feature_names = data.feature_names
        df = pl.DataFrame(
            {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
        )

    n_features = len(feature_names)
    f"✅ Dataset loaded: {len(X)} samples, {n_features} features"
    return X, df, feature_names, n_features, y


@app.cell
def _(df, go, mo, px):
    """Data Visualization with Plotly"""

    import polars as pl

    # Class distribution with Plotly (using Polars)
    class_counts = df["target"].value_counts().sort("target")
    class_dist_fig = px.bar(
        class_counts.to_pandas(),
        x="count",
        y="target",
        color="target",
        title="Class Distribution",
        labels={"count": "Count", "target": "Class"},
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=300,
    )
    class_dist_fig.update_layout(
        template="plotly_white",
        showlegend=False,
        font=dict(size=12),
    )

    # Feature scatter plot (first 2 features)
    feature_cols = [col for col in df.columns if col != "target"][:2]
    if len(feature_cols) >= 2:
        scatter_fig = px.scatter(
            df.to_pandas(),
            x=feature_cols[0],
            y=feature_cols[1],
            color="target",
            title=f"{feature_cols[0]} vs {feature_cols[1]}",
            labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]},
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=300,
            opacity=0.7,
        )
        scatter_fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
        )
        scatter_fig.update_traces(marker=dict(size=5))
    else:
        scatter_fig = None

    if scatter_fig:
        mo.md(f"""
        ## 📈 Data Exploration

        {mo.ui.plotly_chart(class_dist_fig)}
        {mo.ui.plotly_chart(scatter_fig)}
        """)
    else:
        mo.md(f"""
        ## 📈 Data Exploration

        {mo.ui.plotly_chart(class_dist_fig)}
        """)
    return class_dist_fig, scatter_fig


@app.cell
def _(mo):
    """Model Parameters"""

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

    test_size = mo.ui.slider(start=0.1, stop=0.4, step=0.05, value=0.2, label="Test size")

    mo.md(f"""
    ## 🎯 Model Configuration (PINA/PyTorch)
    {hidden_size}
    {num_layers}
    {learning_rate}
    {epochs}
    {test_size}
    """)
    return epochs, hidden_size, learning_rate, num_layers, test_size


@app.cell
def _(
    X,
    accuracy_score,
    epochs,
    hidden_size,
    learning_rate,
    mlflow,
    n_features,
    nn,
    np,
    num_layers,
    optim,
    random_state,
    StandardScaler,
    test_size,
    torch,
    train_test_split,
    y,
):
    """Train PINA/PyTorch Model with MLflow"""

    # Determine number of classes
    n_classes = len(np.unique(y))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=random_state.value, stratify=y
    )

    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

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

    model = Classifier(
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

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, y_pred = torch.max(test_outputs, 1)
        y_pred = y_pred.cpu().numpy()
        accuracy = accuracy_score(y_test, y_pred)

    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"pina_h{hidden_size.value}_l{num_layers.value}") as run:
        # Log tags
        mlflow.set_tag("env", "marimo-flow")
        mlflow.set_tag("model_type", "pina-pytorch")
        mlflow.set_tag("framework", "pina-pytorch")
        mlflow.set_tag("dataset_size", len(X))

        # Log parameters and metrics
        mlflow.log_param("hidden_size", hidden_size.value)
        mlflow.log_param("num_layers", num_layers.value)
        mlflow.log_param("learning_rate", learning_rate.value)
        mlflow.log_param("epochs", epochs.value)
        mlflow.log_param("test_size", test_size.value)
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.pytorch.log_model(model, "model", registered_model_name="basic_pina_model")

        run_id = run.info.run_id

    training_results = {
        "accuracy": accuracy,
        "run_id": run_id,
        "model": model,
        "y_pred": y_pred,
    }

    f"✅ Model trained! Accuracy: {accuracy:.4f}"
    return training_results, y_pred, y_test


@app.cell
def _(go, mo, px, training_results, y_pred, y_test):
    """Results Visualization with Plotly"""

    from sklearn.metrics import confusion_matrix

    # Confusion matrix with Plotly
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = go.Figure(
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
    cm_fig.update_layout(
        title="Confusion Matrix",
        template="plotly_white",
        width=500,
        height=500,
        font=dict(size=12),
    )

    mo.md(f"""
    ## 📊 Results (PINA/PyTorch)

    **Accuracy: {training_results["accuracy"]:.4f}**

    ### Confusion Matrix
    {mo.ui.plotly_chart(cm_fig)}

    **Run ID**: `{training_results["run_id"]}`
    
    *Note: Feature importance visualization is not available for neural networks. Use gradient-based methods or SHAP for interpretability.*
    """)
    return cm_fig,


@app.cell
def _(mlflow_uri, mo):
    """Next Steps"""

    mo.md(f"""
    ## 🔗 Next Steps

    1. **View Results**: [Open MLflow UI]({mlflow_uri.value})
    2. **Experiment**: Modify parameters above to see real-time updates
    3. **Advanced**: Try the multi-model comparison notebook

    ### 🎯 Try This:
    - Adjust model parameters and see accuracy changes
    - Modify dataset size and complexity
    - Compare different runs in MLflow UI
    """)
    return


if __name__ == "__main__":
    app.run()
