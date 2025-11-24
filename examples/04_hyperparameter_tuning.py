# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "mlflow>=2.17.0",
#     "numpy>=1.26.4",
#     "optuna>=3.0.0",
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
    """Hyperparameter Tuning with Optuna + MLflow (PINA/PyTorch)"""

    import warnings

    import altair as alt
    import marimo as mo
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import optuna
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    
    # Set modern Plotly theme
    px.defaults.template = "plotly_white"

    warnings.filterwarnings("ignore")

    mo.md("""
    # 🔬 Hyperparameter Tuning (PINA/PyTorch)

    Automated hyperparameter optimization using Optuna with MLflow tracking.

    Builds on previous examples:
    - Uses PINA/PyTorch models from 02_basic_ml_workflow.py and 03_model_comparison.py
    - Tracks all trials in MLflow
    - Visualizes optimization progress

    Includes:
    - Optuna optimization with Bayesian search
    - MLflow integration for trial tracking
    - Progress visualization
    - Parameter importance analysis
    """)
    return (
        accuracy_score,
        alt,
        f1_score,
        go,
        load_wine,
        mlflow,
        mo,
        nn,
        np,
        optim,
        optuna,
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

    experiment_name = mo.ui.text(
        label="Experiment Name",
        value="hyperparameter-tuning",
        placeholder="Enter experiment name",
    )

    n_trials = mo.ui.slider(
        start=10, stop=100, step=5, value=20, label="Number of trials"
    )

    optimization_direction = mo.ui.dropdown(
        options=["maximize", "minimize"],
        value="maximize",
        label="Optimization direction",
    )

    mo.md(f"""
    ## ⚙️ Configuration

    {mlflow_uri}
    {experiment_name}
    {n_trials}
    {optimization_direction}
    """)
    return experiment_name, mlflow_uri, optimization_direction, n_trials


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

    f"MLflow initialized: {experiment_name.value}"
    return experiment_id,


@app.cell
@mo.cache
def _(load_wine, pl):
    """Load Dataset"""

    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Create DataFrame
    df = pl.DataFrame(
        {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
    )

    return X, data, df, feature_names, y


@app.cell
def _(X, train_test_split, y):
    """Split Data"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    """Optuna Study Configuration"""

    study_name = mo.ui.text(
        label="Study Name",
        value="rf_optimization",
        placeholder="optuna_study",
    )

    sampler_type = mo.ui.dropdown(
        options=["TPE", "Random", "CMA-ES"],
        value="TPE",
        label="Sampler Type",
    )

    mo.md(f"""
    ## 🔬 Optuna Study Configuration

    {study_name}
    {sampler_type}
    """)
    return sampler_type, study_name


@app.cell
def _(sampler_type):
    """Create Optuna Sampler"""

    if sampler_type.value == "TPE":
        sampler = optuna.samplers.TPESampler(seed=42)
    elif sampler_type.value == "Random":
        sampler = optuna.samplers.RandomSampler(seed=42)
    else:  # CMA-ES
        sampler = optuna.samplers.CmaEsSampler(seed=42)

    return (sampler,)


@app.cell
def _(
    X_test,
    X_train,
    accuracy_score,
    experiment_id,
    mlflow,
    nn,
    np,
    optim,
    optimization_direction,
    optuna,
    sampler,
    StandardScaler,
    study_name,
    torch,
    y_test,
    y_train,
):
    """Run Optuna Optimization with MLflow Tracking (PINA/PyTorch)"""

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

    # MLflow callback for tracking trials
    class MLflowCallback:
        def __init__(self, experiment_id):
            self.experiment_id = experiment_id

        def __call__(self, study, trial):
            with mlflow.start_run(
                experiment_id=self.experiment_id, run_name=f"trial_{trial.number}"
            ) as run:
                # Log parameters
                for param_name, param_value in trial.params.items():
                    mlflow.log_param(param_name, param_value)

                # Log metrics
                mlflow.log_metric("value", trial.value)
                mlflow.log_metric("trial_number", trial.number)

                # Log state
                mlflow.set_tag("trial_state", trial.state.name)
                mlflow.set_tag("study_name", study.study_name)
                mlflow.set_tag("framework", "pina-pytorch")

    # Define neural network model
    class TunableNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate, num_classes):
            super(TunableNN, self).__init__()
            layers = []
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.Linear(hidden_size, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        epochs = trial.suggest_int("epochs", 50, 300, step=50)

        # Create model
        model = TunableNN(n_features, hidden_size, num_layers, dropout_rate, n_classes).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
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
            test_accuracy = accuracy_score(y_test, y_pred)
        
        # Log test metrics to trial user attributes
        trial.set_user_attr("test_accuracy", test_accuracy)

        return test_accuracy

    # Create study
    study = optuna.create_study(
        direction=optimization_direction.value,
        sampler=sampler,
        study_name=study_name.value,
    )

    # Run optimization with MLflow callback
    mlflow_callback = MLflowCallback(experiment_id)
    study.optimize(
        objective, n_trials=n_trials.value, callbacks=[mlflow_callback], show_progress_bar=True
    )

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    # Train best model
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    best_y_pred = best_model.predict(X_test)
    best_test_accuracy = accuracy_score(y_test, best_y_pred)

    # Log best model to MLflow
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="best_model"
    ) as best_run:
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_accuracy", best_value)
        mlflow.log_metric("test_accuracy", best_test_accuracy)
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("optimization", "optuna")
        mlflow.sklearn.log_model(
            best_model, "model", registered_model_name="optimized_rf_model"
        )

    f"Optimization complete. Best CV accuracy: {best_value:.4f}, Test accuracy: {best_test_accuracy:.4f}"
    return (
        best_model,
        best_params,
        best_test_accuracy,
        best_trial,
        best_value,
        best_y_pred,
        mlflow_callback,
        objective,
        study,
    )


@app.cell
def _(mo, pl, px, study):
    """Visualize Optimization Progress with Plotly"""

    # Extract trial data
    trials_data = []
    for trial in study.trials:
        trials_data.append(
            {
                "Trial": trial.number,
                "Value": trial.value if trial.value is not None else 0,
                "State": trial.state.name,
            }
        )

    df_trials = pl.DataFrame(trials_data)

    # Progress chart with Plotly
    progress_fig = px.line(
        df_trials.to_pandas(),
        x="Trial",
        y="Value",
        color="State",
        title="Optimization Progress",
        labels={"Trial": "Trial Number", "Value": "CV Accuracy"},
        height=300,
    )
    progress_fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        yaxis=dict(range=[0, 1]),
    )
    progress_fig.update_traces(mode="lines+markers", marker=dict(size=4))

    # Best value over time
    best_values = []
    current_best = None
    for trial in study.trials:
        if trial.value is not None:
            if current_best is None or trial.value > current_best:
                current_best = trial.value
            best_values.append({"Trial": trial.number, "Best Value": current_best})

    df_best = pl.DataFrame(best_values)
    best_fig = px.line(
        df_best.to_pandas(),
        x="Trial",
        y="Best Value",
        title="Best Value Over Time",
        labels={"Trial": "Trial Number", "Best Value": "Best CV Accuracy"},
        color_discrete_sequence=["red"],
        height=300,
    )
    best_fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        showlegend=False,
    )
    best_fig.update_traces(line=dict(width=3))

    mo.md(f"""
    ## Optimization Progress

    ### Trial Results
    {mo.ui.plotly_chart(progress_fig)}

    ### Best Value Evolution
    {mo.ui.plotly_chart(best_fig)}
    """)
    return best_fig, best_values, df_best, df_trials, progress_fig, trials_data


@app.cell
def _(best_params, best_trial, best_value, best_test_accuracy, mo, optuna, pl, px, study):
    """Best Model Summary"""

    # Parameter importance
    importance = optuna.importance.get_param_importances(study)

    importance_data = [
        {"Parameter": param, "Importance": imp} for param, imp in importance.items()
    ]
    importance_df = pl.DataFrame(importance_data).sort("Importance", descending=True)

    importance_fig = px.bar(
        importance_df.to_pandas(),
        x="Importance",
        y="Parameter",
        orientation="h",
        title="Parameter Importance",
        color="Importance",
        color_continuous_scale="viridis",
        height=300,
    )
    importance_fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        yaxis={"categoryorder": "total ascending"},
    )

    mo.md(f"""
    ## Best Model Summary

    ### Performance
    - CV Accuracy: {best_value:.4f}
    - Test Accuracy: {best_test_accuracy:.4f}
    - Trial Number: {best_trial.number}

    ### Best Hyperparameters
    {mo.ui.table(pl.DataFrame([{"Parameter": k, "Value": v} for k, v in best_params.items()]))}

    ### Parameter Importance
    {mo.ui.plotly_chart(importance_fig)}
    """)
    return importance, importance_fig, importance_data, importance_df


@app.cell
def _(experiment_name, mlflow_uri, mo):
    """Next Steps"""

    mo.md(f"""
    ## Next Steps

    1. View Results: [Open MLflow UI]({mlflow_uri.value}) to see all trials
    2. Model Registry: Use 05_model_registry.py to register and deploy the best model
    3. Production Pipeline: Use 06_production_pipeline.py for end-to-end deployment

    Suggestions:
    - Increase number of trials for better results
    - Try different samplers (TPE vs Random)
    - Compare optimization across different datasets
    - Analyze parameter importance to understand model behavior
    """)
    return


if __name__ == "__main__":
    app.run()

