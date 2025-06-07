import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Model Training with MLflow""")
    return

@app.cell
def _(mo):
    """Model Configuration"""
    model_type = mo.ui.dropdown(
        options=["RandomForest", "GradientBoosting", "LogisticRegression"],
        value="RandomForest",
        label="🤖 Model Type"
    )
    
    n_estimators = mo.ui.slider(
        start=10, stop=200, step=10, value=100,
        label="🌳 Number of Estimators"
    )
    
    test_size = mo.ui.slider(
        start=0.1, stop=0.5, step=0.05, value=0.2,
        label="📊 Test Size"
    )
    
    mo.md(f"""
    ## ⚙️ Model Configuration
    {model_type}
    {n_estimators}
    {test_size}
    """)
    return model_type, n_estimators, test_size

@app.cell
def _(model_type, n_estimators, test_size):
    """Prepare Sample Dataset"""
    import numpy as np
    import polars as pl
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    # Generate sample classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    # Convert to DataFrame for better handling
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pl.DataFrame(X, schema=feature_names)
    df = df.with_columns(pl.Series("target", y))
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=42
    )
    
    print(f"✅ Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X, X_test, X_train, df, feature_names, np, pl, train_test_split, y, y_test, y_train

@app.cell
def _(X_test, X_train, model_type, n_estimators, y_test, y_train):
    """Train Model with MLflow Tracking"""
    import mlflow
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # Model selection
    if model_type.value == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=n_estimators.value,
            random_state=42
        )
    elif model_type.value == "GradientBoosting":
        model = GradientBoostingClassifier(
            n_estimators=n_estimators.value,
            random_state=42
        )
    else:  # LogisticRegression
        model = LogisticRegression(random_state=42)
    
    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"{model_type.value}_experiment") as run:
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Log parameters
        mlflow.log_param("model_type", model_type.value)
        if hasattr(model, 'n_estimators'):
            mlflow.log_param("n_estimators", n_estimators.value)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        run_id = run.info.run_id
    
    print(f"✅ Model trained: {model_type.value}")
    print(f"📈 Train Accuracy: {train_accuracy:.4f}")
    print(f"📊 Test Accuracy: {test_accuracy:.4f}")
    print(f"🔗 MLflow Run ID: {run_id}")
    
    return (
        LogisticRegression,
        RandomForestClassifier,
        accuracy_score,
        classification_report,
        mlflow,
        model,
        run_id,
        test_accuracy,
        train_accuracy,
        y_pred_test,
        y_pred_train,
    )

@app.cell
def _(classification_report, mo, y_pred_test, y_test):
    """Model Evaluation Report"""
    report = classification_report(y_test, y_pred_test, output_dict=True)
    
    mo.md(f"""
    ## 📋 Model Performance Report
    
    ### Classification Report
    ```
    {classification_report(y_test, y_pred_test)}
    ```
    
    ### Key Metrics
    - **Accuracy**: {report['accuracy']:.4f}
    - **Macro Avg F1**: {report['macro avg']['f1-score']:.4f}
    - **Weighted Avg F1**: {report['weighted avg']['f1-score']:.4f}
    """)
    return (report,)

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run() 