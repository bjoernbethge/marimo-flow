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
        # MLflow: Optuna Integration
        
        This snippet demonstrates how to integrate Optuna hyperparameter optimization
        with MLflow tracking using a custom callback to log each trial.
        """
    )
    return


@app.cell
def _():
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import optuna
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # Setup MLflow
    from pathlib import Path
    tracking_path = Path("./data/experiments")
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_path.resolve()}")
    
    return (
        Path,
        accuracy_score,
        make_classification,
        mlflow,
        np,
        optuna,
        optim,
        torch,
        train_test_split,
    )


@app.cell
def _(make_classification, train_test_split):
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X, X_test, X_train, y, y_test, y_train


@app.cell
def _(mlflow):
    # Create or get experiment
    experiment_name = "optuna_optimization"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    
    return experiment, experiment_id, experiment_name


@app.cell
def _(experiment_id, mlflow):
    # MLflow callback for Optuna trials
    class MLflowCallback:
        def __init__(self, experiment_id):
            self.experiment_id = experiment_id
        
        def __call__(self, study, trial):
            with mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=f"trial_{trial.number}"
            ) as run:
                # Log all trial parameters
                for param_name, param_value in trial.params.items():
                    mlflow.log_param(param_name, param_value)
                
                # Log trial metrics
                mlflow.log_metric("value", trial.value)
                mlflow.log_metric("trial_number", trial.number)
                
                # Log trial state and study info
                mlflow.set_tag("trial_state", trial.state.name)
                mlflow.set_tag("study_name", study.study_name)
                mlflow.set_tag("optimization", "optuna")
    
    return MLflowCallback


@app.cell
def _(X_train, nn, np, torch, y_train):
    # Define model architecture
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    class TunableModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout_rate, num_classes):
            super().__init__()
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
    
    return n_classes, n_features, TunableModel


@app.cell
def _(MLflowCallback, TunableModel, X_test, X_train, accuracy_score, experiment_id, nn, n_classes, n_features, optuna, optim, torch, y_test, y_train):
    # Define objective function for Optuna
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    def objective(trial):
        # Suggest hyperparameters
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        epochs = trial.suggest_int("epochs", 10, 50, step=10)
        
        # Create model
        model = TunableModel(n_features, hidden_size, num_layers, dropout_rate, n_classes).to(device)
        
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
        
        return test_accuracy
    
    # Create study and run optimization
    study = optuna.create_study(direction="maximize", study_name="neural_network_tuning")
    
    mlflow_callback = MLflowCallback(experiment_id)
    study.optimize(
        objective,
        n_trials=5,  # Small number for demo
        callbacks=[mlflow_callback],
        show_progress_bar=True
    )
    
    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    
    return (
        best_params,
        best_trial,
        best_value,
        device,
        mlflow_callback,
        objective,
        study,
        X_test_tensor,
        X_train_tensor,
        y_train_tensor,
    )


if __name__ == "__main__":
    app.run()
