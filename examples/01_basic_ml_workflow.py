import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def __():
    """Basic ML Workflow with Marimo Flow"""
    import marimo as mo
    import mlflow
    import mlflow.sklearn
    import polars as pl
    import altair as alt
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.datasets import make_classification
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # 🚀 Basic ML Workflow - Marimo Flow
    
    **Simple ML experiment with MLflow tracking**
    
    This notebook demonstrates:
    - 📊 **Data Generation**: Synthetic classification dataset
    - 🎯 **Model Training**: Random Forest classifier
    - 🔬 **MLflow Tracking**: Experiment logging
    - 📈 **Visualization**: Results and metrics
    """)
    return (
        RandomForestClassifier,
        accuracy_score,
        alt,
        classification_report,
        make_classification,
        mlflow,
        mo,
        np,
        pl,
        train_test_split,
        warnings,
    )


@app.cell
def __(mo):
    """MLflow Configuration"""
    mlflow_uri = mo.ui.text(
        label="🔗 MLflow Tracking URI",
        value="http://localhost:5000",
        placeholder="Enter MLflow server URL"
    )
    
    experiment_name = mo.ui.text(
        label="🧪 Experiment Name", 
        value="basic-ml-workflow",
        placeholder="Enter experiment name"
    )
    
    mo.md(f"""
    ## ⚙️ MLflow Configuration
    {mlflow_uri}
    {experiment_name}
    """)
    return experiment_name, mlflow_uri


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
def __(mo):
    """Dataset Parameters"""
    n_samples = mo.ui.slider(
        start=500, stop=2000, step=100, value=1000,
        label="Number of samples"
    )
    
    n_features = mo.ui.slider(
        start=10, stop=50, step=5, value=20,
        label="Number of features"
    )
    
    random_state = mo.ui.number(
        start=0, stop=100, step=1, value=42,
        label="Random state"
    )
    
    mo.md(f"""
    ## 📊 Dataset Configuration
    {n_samples}
    {n_features}
    {random_state}
    """)
    return n_features, n_samples, random_state


@app.cell
def __(make_classification, n_features, n_samples, pl, random_state):
    """Generate Dataset"""
    X, y = make_classification(
        n_samples=n_samples.value,
        n_features=n_features.value,
        n_classes=2,
        n_redundant=2,
        n_informative=n_features.value - 2,
        random_state=random_state.value,
        n_clusters_per_class=1
    )
    
    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features.value)]
    df = pl.DataFrame({
        **{name: X[:, i] for i, name in enumerate(feature_names)},
        "target": y
    })
    
    f"📊 Dataset created: {len(df)} samples × {n_features.value} features"
    return X, df, feature_names, y


@app.cell
def __(alt, df, mo):
    """Data Visualization"""
    df_pandas = df.to_pandas()
    
    # Class distribution
    class_dist = alt.Chart(df_pandas).mark_bar().encode(
        x=alt.X('count()', title='Count'),
        y=alt.Y('target:O', title='Class'),
        color=alt.Color('target:N', scale=alt.Scale(scheme='category10'))
    ).properties(title="Class Distribution")
    
    # Feature scatter plot
    scatter = alt.Chart(df_pandas).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('feature_0:Q', title='Feature 0'),
        y=alt.Y('feature_1:Q', title='Feature 1'),
        color=alt.Color('target:N', scale=alt.Scale(scheme='category10'))
    ).interactive().properties(title="Feature Space")
    
    mo.md(f"""
    ## 📈 Data Exploration
    
    ### Class Distribution
    {mo.ui.altair_chart(class_dist)}
    
    ### Feature Space
    {mo.ui.altair_chart(scatter)}
    """)
    return class_dist, df_pandas, scatter


@app.cell
def __(mo):
    """Model Parameters"""
    n_estimators = mo.ui.slider(
        start=10, stop=200, step=10, value=100,
        label="Number of estimators"
    )
    
    max_depth = mo.ui.slider(
        start=3, stop=20, step=1, value=10,
        label="Max depth"
    )
    
    test_size = mo.ui.slider(
        start=0.1, stop=0.4, step=0.05, value=0.2,
        label="Test size"
    )
    
    mo.md(f"""
    ## 🎯 Model Configuration
    {n_estimators}
    {max_depth}
    {test_size}
    """)
    return max_depth, n_estimators, test_size


@app.cell
def __(
    RandomForestClassifier,
    X,
    accuracy_score,
    classification_report,
    max_depth,
    mlflow,
    n_estimators,
    random_state,
    test_size,
    train_test_split,
    y,
):
    """Train Model with MLflow"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=random_state.value
    )
    
    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"rf_n{n_estimators.value}_d{max_depth.value}") as run:
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=n_estimators.value,
            max_depth=max_depth.value,
            random_state=random_state.value,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators.value)
        mlflow.log_param("max_depth", max_depth.value)
        mlflow.log_param("test_size", test_size.value)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(
            model, "model",
            registered_model_name="basic_rf_model"
        )
        
        run_id = run.info.run_id
    
    training_results = {
        "accuracy": accuracy,
        "run_id": run_id,
        "model": model,
        "y_pred": y_pred
    }
    
    f"✅ Model trained! Accuracy: {accuracy:.4f}"
    return X_test, X_train, model, run_id, training_results, y_pred, y_test, y_train


@app.cell
def __(alt, mo, np, training_results, y_pred, y_test):
    """Results Visualization"""
    from sklearn.metrics import confusion_matrix
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_data = []
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            cm_data.append({
                'True': f'Class {i}',
                'Predicted': f'Class {j}',
                'Count': int(cm[i, j])
            })
    
    confusion_chart = alt.Chart(alt.InlineData(values=cm_data)).mark_rect().encode(
        x=alt.X('Predicted:N'),
        y=alt.Y('True:N'),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['True:N', 'Predicted:N', 'Count:Q']
    ).properties(title="Confusion Matrix")
    
    # Feature importance
    importance_data = [
        {'Feature': f'Feature {i}', 'Importance': imp}
        for i, imp in enumerate(training_results["model"].feature_importances_)
    ]
    importance_data = sorted(importance_data, key=lambda x: x['Importance'], reverse=True)[:10]
    
    importance_chart = alt.Chart(alt.InlineData(values=importance_data)).mark_bar().encode(
        x=alt.X('Importance:Q'),
        y=alt.Y('Feature:N', sort='-x'),
        color=alt.Color('Importance:Q', scale=alt.Scale(scheme='viridis'))
    ).properties(title="Top 10 Feature Importance")
    
    mo.md(f"""
    ## 📊 Results
    
    **Accuracy: {training_results["accuracy"]:.4f}**
    
    ### Confusion Matrix
    {mo.ui.altair_chart(confusion_chart)}
    
    ### Feature Importance
    {mo.ui.altair_chart(importance_chart)}
    
    **Run ID**: `{training_results["run_id"]}`
    """)
    return cm, cm_data, confusion_chart, confusion_matrix, importance_chart, importance_data


@app.cell
def __(experiment_name, mlflow_uri, mo):
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