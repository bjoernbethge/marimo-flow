# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "numpy>=1.26.4",
#     "polars>=1.12.0",
#     "pina>=0.1.0",
#     "torch>=2.0.0",
# ]
# ///

import marimo
import marimo as mo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def __():
    """Model Comparison with Marimo Flow (PINA/PyTorch)"""
    import warnings

    import altair as alt
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_breast_cancer, load_iris, load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    
    # Set modern Plotly theme
    px.defaults.template = "plotly_white"

    import mlflow
    import mlflow.pytorch
    warnings.filterwarnings('ignore')

    mo.md("""
    # 🏆 Model Comparison (PINA/PyTorch)
    
    **Compare multiple Neural Network architectures with MLflow tracking**
    
    Overview:
    - 🎯 **Multiple Architectures**: Different NN configurations
    - 📊 **Real Datasets**: Wine, Breast Cancer, Iris
    - 🔬 **Cross-Validation**: Robust evaluation
    - 📈 **Comparison**: Side-by-side metrics
    """)
    return (
        accuracy_score,
        alt,
        f1_score,
        go,
        load_breast_cancer,
        load_iris,
        load_wine,
        mlflow,
        mo,
        nn,
        np,
        optim,
        pl,
        precision_score,
        px,
        recall_score,
        StandardScaler,
        torch,
        train_test_split,
        warnings,
    )


@app.cell
def __(mo):
    """Configuration"""
    mlflow_uri = mo.ui.text(
        label="🔗 MLflow Tracking URI",
        value="./mlflow"
    )
    
    experiment_name = mo.ui.text(
        label="🧪 Experiment Name", 
        value="model-comparison"
    )
    
    dataset_choice = mo.ui.dropdown(
        options=["wine", "breast_cancer", "iris"],
        value="wine",
        label="📊 Dataset"
    )
    
    mo.md(f"""
    ## ⚙️ Configuration
    {mlflow_uri}
    {experiment_name}
    {dataset_choice}
    """)
    return dataset_choice, experiment_name, mlflow_uri


@app.cell
def __(experiment_name, mlflow, mlflow_uri):
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
def __(dataset_choice, load_breast_cancer, load_iris, load_wine, pl):
    """Load Dataset"""
    dataset_loaders = {
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "iris": load_iris
    }
    
    data = dataset_loaders[dataset_choice.value]()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    # Create DataFrame for visualization
    df = pl.DataFrame({
        **{name: X[:, i] for i, name in enumerate(feature_names)},
        "target": y
    })
    
    dataset_info = {
        "name": dataset_choice.value,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "n_classes": len(target_names),
        "feature_names": feature_names,
        "target_names": target_names
    }
    
    return X, data, dataset_info, df, feature_names, target_names, y


@app.cell
def __(df, go, mo, px, target_names, pl):
    """Dataset Visualization with Plotly"""
    
    # Class distribution (Polars)
    class_counts = df['target'].value_counts().sort('target')
    class_counts_pd = class_counts.to_pandas()
    class_counts_pd['target_name'] = [target_names[i] for i in class_counts_pd['target']]
    
    class_dist_fig = px.bar(
        class_counts_pd,
        x='count',
        y='target_name',
        color='target',
        title="Class Distribution",
        labels={'count': 'Count', 'target_name': 'Class'},
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=300,
    )
    class_dist_fig.update_layout(
        template="plotly_white",
        showlegend=False,
        font=dict(size=12),
    )
    
    # Feature correlation heatmap (first 10 features) - Polars
    feature_cols = [col for col in df.columns if col != 'target'][:10]
    corr_df = df.select(feature_cols)
    corr_matrix = corr_df.corr()
    
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.to_numpy(),
            x=feature_cols,
            y=feature_cols,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.to_numpy().round(2),
            texttemplate="%{text}",
            textfont={"size": 9},
            colorbar=dict(title="Correlation"),
        )
    )
    heatmap_fig.update_layout(
        title="Feature Correlation (Top 10)",
        template="plotly_white",
        width=500,
        height=500,
        font=dict(size=11),
        xaxis=dict(side="bottom"),
    )
    
    mo.md(f"""
    ## 📈 Dataset Overview
    
    ### Class Distribution
    {mo.ui.plotly_chart(class_dist_fig)}
    
    ### Feature Correlation
    {mo.ui.plotly_chart(heatmap_fig)}
    """)
    return class_counts, class_dist_fig, corr_matrix, heatmap_fig


@app.cell
def __(mo):
    """Model Configuration"""
    test_size = mo.ui.slider(
        start=0.1, stop=0.4, step=0.05, value=0.2,
        label="Test size"
    )
    
    cv_folds = mo.ui.slider(
        start=3, stop=10, step=1, value=5,
        label="Cross-validation folds"
    )
    
    random_state = mo.ui.number(
        start=0, stop=100, step=1, value=42,
        label="Random state"
    )
    
    mo.md(f"""
    ## 🎯 Model Configuration
    {test_size}
    {cv_folds}
    {random_state}
    """)
    return cv_folds, random_state, test_size


@app.cell
def __(
    X,
    accuracy_score,
    cv_folds,
    dataset_info,
    f1_score,
    mlflow,
    nn,
    np,
    optim,
    precision_score,
    random_state,
    recall_score,
    StandardScaler,
    test_size,
    torch,
    train_test_split,
    y,
):
    """Train Multiple PINA/PyTorch Models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=random_state.value, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    # Define different neural network architectures
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(SimpleNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, num_classes)
            )
        def forward(self, x):
            return self.network(x)
    
    class DeepNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(DeepNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, num_classes)
            )
        def forward(self, x):
            return self.network(x)
    
    class WideNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(WideNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size * 2, num_classes)
            )
        def forward(self, x):
            return self.network(x)
    
    # Define model configurations
    model_configs = {
        "Simple NN (64)": {"class": SimpleNN, "hidden": 64, "epochs": 150},
        "Simple NN (128)": {"class": SimpleNN, "hidden": 128, "epochs": 150},
        "Deep NN (64)": {"class": DeepNN, "hidden": 64, "epochs": 200},
        "Wide NN (128)": {"class": WideNN, "hidden": 128, "epochs": 150},
    }
    
    # Train and evaluate models
    model_results = []
    models = {}
    
    for model_name, config in model_configs.items():
        with mlflow.start_run(run_name=f"{model_name}_{dataset_info['name']}") as run:
            # Create model
            model = config["class"](n_features, config["hidden"], n_classes).to(device)
            
            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            model.train()
            for epoch in range(config["epochs"]):
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
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Simple cross-validation (train on subset)
            cv_scores = []
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds.value, shuffle=True, random_state=random_state.value)
            for train_idx, val_idx in kf.split(X_train_scaled):
                X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                
                X_cv_train_t = torch.FloatTensor(X_cv_train).to(device)
                y_cv_train_t = torch.LongTensor(y_cv_train).to(device)
                X_cv_val_t = torch.FloatTensor(X_cv_val).to(device)
                
                cv_model = config["class"](n_features, config["hidden"], n_classes).to(device)
                cv_optimizer = optim.Adam(cv_model.parameters(), lr=0.001)
                
                cv_model.train()
                for _ in range(100):  # Quick training
                    cv_optimizer.zero_grad()
                    outputs = cv_model(X_cv_train_t)
                    loss = criterion(outputs, y_cv_train_t)
                    loss.backward()
                    cv_optimizer.step()
                
                cv_model.eval()
                with torch.no_grad():
                    cv_outputs = cv_model(X_cv_val_t)
                    _, cv_pred = torch.max(cv_outputs, 1)
                    cv_pred = cv_pred.cpu().numpy()
                    cv_scores.append(accuracy_score(y_cv_val, cv_pred))
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Log to MLflow
            mlflow.set_tag("env", "marimo-flow")
            mlflow.set_tag("framework", "pina-pytorch")
            mlflow.set_tag("model_class", config["class"].__name__)
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("hidden_size", config["hidden"])
            mlflow.log_param("epochs", config["epochs"])
            mlflow.log_param("dataset", dataset_info['name'])
            mlflow.log_param("test_size", test_size.value)
            mlflow.log_param("cv_folds", cv_folds.value)
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("cv_accuracy_mean", cv_mean)
            mlflow.log_metric("cv_accuracy_std", cv_std)
            
            # Log model
            mlflow.pytorch.log_model(
                model, "model",
                registered_model_name=f"{model_name.lower().replace(' ', '_')}_model"
            )
            
            # Store results
            model_results.append({
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "CV Mean": cv_mean,
                "CV Std": cv_std,
                "Run ID": run.info.run_id
            })
            
            models[model_name] = model
    
    f"✅ Trained {len(model_configs)} models on {dataset_info['name']} dataset"
    return X_test, X_train, model_results, models, y_pred, y_test, y_train


@app.cell
def __(go, mo, pl, px, model_results):
    """Results Comparison with Plotly"""
    # Metrics comparison
    metrics_data = []
    for result in model_results:
        for metric_name in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            metrics_data.append({
                "Model": result["Model"],
                "Metric": metric_name,
                "Score": result[metric_name]
            })
    
    metrics_df = pl.DataFrame(metrics_data)
    
    # Create grouped bar chart with Plotly (convert to pandas for Plotly)
    metrics_fig = px.bar(
        metrics_df.to_pandas(),
        x="Model",
        y="Score",
        color="Metric",
        facet_col="Metric",
        title="Model Performance Comparison",
        labels={"Score": "Score", "Model": "Model"},
        color_discrete_sequence=px.colors.qualitative.Set3,
        height=300,
    )
    metrics_fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        xaxis=dict(title=""),
    )
    metrics_fig.update_xaxes(matches=None, showticklabels=True)
    metrics_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    # Cross-validation comparison with error bars
    cv_data = []
    for result in model_results:
        cv_data.append({
            "Model": result["Model"],
            "CV Mean": result["CV Mean"],
            "CV Std": result["CV Std"],
            "Lower": result["CV Mean"] - result["CV Std"],
            "Upper": result["CV Mean"] + result["CV Std"]
        })
    
    cv_df = pl.DataFrame(cv_data)
    
    cv_fig = go.Figure()
    # Use Polars iter_rows instead of iterrows
    for row in cv_df.iter_rows(named=True):
        cv_fig.add_trace(go.Scatter(
            x=[row["CV Mean"], row["CV Mean"]],
            y=[row["Model"], row["Model"]],
            mode="markers",
            marker=dict(size=10, color="red"),
            showlegend=False,
        ))
        cv_fig.add_trace(go.Scatter(
            x=[row["Lower"], row["Upper"]],
            y=[row["Model"], row["Model"]],
            mode="lines",
            line=dict(width=2, color="gray"),
            showlegend=False,
        ))
    
    cv_fig.update_layout(
        title="Cross-Validation Results",
        template="plotly_white",
        xaxis_title="Cross-Validation Accuracy",
        yaxis_title="Model",
        height=300,
        font=dict(size=12),
    )
    
    # Results table
    results_table = mo.ui.table(
        data=model_results,
        selection=None
    )
    
    mo.md(f"""
    ## 🏆 Model Comparison Results
    
    ### Performance Metrics
    {mo.ui.plotly_chart(metrics_fig)}
    
    ### Cross-Validation Results
    {mo.ui.plotly_chart(cv_fig)}
    
    ### Detailed Results
    {results_table}
    """)
    return cv_data, cv_df, cv_fig, metrics_data, metrics_df, metrics_fig, results_table


@app.cell
def __(model_results, mo):
    """Best Model Analysis"""
    # Find best model by accuracy
    best_model = max(model_results, key=lambda x: x["Accuracy"])
    
    # Ranking by different metrics
    rankings = {}
    for metric_type in ["Accuracy", "Precision", "Recall", "F1-Score", "CV Mean"]:
        sorted_models = sorted(model_results, key=lambda x: x[metric_type], reverse=True)
        rankings[metric_type] = [model["Model"] for model in sorted_models]
    
    mo.md(f"""
    ## 🥇 Best Model Analysis
    
    ### Overall Winner (by Accuracy)
    **{best_model["Model"]}** with {best_model["Accuracy"]:.4f} accuracy
    
    ### Rankings by Metric:
    - **Accuracy**: {" > ".join(rankings["Accuracy"])}
    - **Precision**: {" > ".join(rankings["Precision"])}
    - **Recall**: {" > ".join(rankings["Recall"])}
    - **F1-Score**: {" > ".join(rankings["F1-Score"])}
    - **CV Mean**: {" > ".join(rankings["CV Mean"])}
    
    ### Best Model Details:
    - **Accuracy**: {best_model["Accuracy"]:.4f}
    - **Precision**: {best_model["Precision"]:.4f}
    - **Recall**: {best_model["Recall"]:.4f}
    - **F1-Score**: {best_model["F1-Score"]:.4f}
    - **CV Mean ± Std**: {best_model["CV Mean"]:.4f} ± {best_model["CV Std"]:.4f}
    - **Run ID**: `{best_model["Run ID"]}`
    """)
    return best_model, rankings


@app.cell
def __(dataset_choice, experiment_name, mlflow_uri, mo):
    """Next Steps"""
    mo.md(f"""
    ## 🔗 Next Steps
    
    1. **View Results**: [Open MLflow UI]({mlflow_uri.value})
    2. **Try Different Dataset**: Change dataset above to see how models perform
    3. **Hyperparameter Tuning**: Use the hyperparameter optimization notebook
    
    ### 🎯 Experiment Ideas:
    - Compare performance across different datasets
    - Adjust cross-validation folds
    - Try different test/train splits
    - Analyze feature importance for best model
    """)
    return


if __name__ == "__main__":
    app.run() 