import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def __():
    """Model Comparison with Marimo Flow"""
    import marimo as mo
    import mlflow
    import mlflow.sklearn
    import polars as pl
    import altair as alt
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.datasets import load_wine, load_breast_cancer, load_iris
    import warnings
    warnings.filterwarnings('ignore')

    mo.md("""
    # 🏆 Model Comparison - Marimo Flow
    
    **Compare multiple ML models with MLflow tracking**
    
    This notebook demonstrates:
    - 🎯 **Multiple Models**: RF, GB, LR, SVM
    - 📊 **Real Datasets**: Wine, Breast Cancer, Iris
    - 🔬 **Cross-Validation**: Robust evaluation
    - 📈 **Comparison**: Side-by-side metrics
    """)
    return (
        GradientBoostingClassifier,
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        accuracy_score,
        alt,
        cross_val_score,
        f1_score,
        load_breast_cancer,
        load_iris,
        load_wine,
        mlflow,
        mo,
        np,
        pl,
        precision_score,
        recall_score,
        train_test_split,
        warnings,
    )


@app.cell
def __(mo):
    """Configuration"""
    mlflow_uri = mo.ui.text(
        label="🔗 MLflow Tracking URI",
        value="http://localhost:5000"
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
    
    f"📊 Dataset loaded: {dataset_info['name']} ({dataset_info['n_samples']} samples, {dataset_info['n_features']} features, {dataset_info['n_classes']} classes)"
    return X, data, dataset_info, df, feature_names, target_names, y


@app.cell
def __(alt, df, mo, target_names):
    """Dataset Visualization"""
    df_pandas = df.to_pandas()
    
    # Class distribution
    class_counts = df_pandas['target'].value_counts().reset_index()
    class_counts['target_name'] = [target_names[i] for i in class_counts['target']]
    
    class_dist = alt.Chart(class_counts).mark_bar().encode(
        x=alt.X('count', title='Count'),
        y=alt.Y('target_name:N', title='Class'),
        color=alt.Color('target:N', scale=alt.Scale(scheme='category10'))
    ).properties(title="Class Distribution")
    
    # Feature correlation heatmap (first 10 features)
    corr_data = df_pandas.iloc[:, :min(10, len(df_pandas.columns)-1)].corr()
    corr_list = []
    for i, col1 in enumerate(corr_data.columns):
        for j, col2 in enumerate(corr_data.columns):
            corr_list.append({
                'Feature1': col1,
                'Feature2': col2,
                'Correlation': corr_data.iloc[i, j]
            })
    
    heatmap = alt.Chart(alt.InlineData(values=corr_list)).mark_rect().encode(
        x=alt.X('Feature1:N', title=''),
        y=alt.Y('Feature2:N', title=''),
        color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domain=[-1, 1])),
        tooltip=['Feature1:N', 'Feature2:N', 'Correlation:Q']
    ).properties(title="Feature Correlation (Top 10)")
    
    mo.md(f"""
    ## 📈 Dataset Overview
    
    ### Class Distribution
    {mo.ui.altair_chart(class_dist)}
    
    ### Feature Correlation
    {mo.ui.altair_chart(heatmap)}
    """)
    return class_counts, class_dist, corr_data, corr_list, df_pandas, heatmap


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
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    X,
    accuracy_score,
    cross_val_score,
    cv_folds,
    dataset_info,
    f1_score,
    mlflow,
    precision_score,
    random_state,
    recall_score,
    test_size,
    train_test_split,
    y,
):
    """Train Multiple Models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=random_state.value, stratify=y
    )
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=random_state.value, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=random_state.value
        ),
        "Logistic Regression": LogisticRegression(
            random_state=random_state.value, max_iter=1000
        ),
        "SVM": SVC(
            random_state=random_state.value, probability=True
        )
    }
    
    # Train and evaluate models
    model_results = []
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{dataset_info['name']}") as run:
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds.value, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Log to MLflow
            mlflow.log_param("model_type", model_name)
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
            mlflow.sklearn.log_model(
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
    
    f"✅ Trained {len(models)} models on {dataset_info['name']} dataset"
    return X_test, X_train, model_results, models, y_pred, y_test, y_train


@app.cell
def __(alt, mo, model_results):
    """Results Comparison"""
    # Metrics comparison
    metrics_data = []
    for result in model_results:
        for metric_name in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            metrics_data.append({
                "Model": result["Model"],
                "Metric": metric_name,
                "Score": result[metric_name]
            })
    
    metrics_chart = alt.Chart(alt.InlineData(values=metrics_data)).mark_bar().encode(
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Model:N'),
        color=alt.Color('Metric:N', scale=alt.Scale(scheme='category10')),
        column=alt.Column('Metric:N')
    ).resolve_scale(x='independent').properties(title="Model Performance Comparison")
    
    # Cross-validation comparison
    cv_data = []
    for result in model_results:
        cv_data.append({
            "Model": result["Model"],
            "CV Mean": result["CV Mean"],
            "CV Std": result["CV Std"],
            "Lower": result["CV Mean"] - result["CV Std"],
            "Upper": result["CV Mean"] + result["CV Std"]
        })
    
    cv_chart = alt.Chart(alt.InlineData(values=cv_data)).mark_errorbar().encode(
        x=alt.X('Lower:Q', title='Cross-Validation Accuracy'),
        x2='Upper:Q',
        y=alt.Y('Model:N')
    ) + alt.Chart(alt.InlineData(values=cv_data)).mark_circle(size=100).encode(
        x=alt.X('CV Mean:Q'),
        y=alt.Y('Model:N'),
        color=alt.value('red')
    )
    
    # Results table
    results_table = mo.ui.table(
        data=model_results,
        selection=None
    )
    
    mo.md(f"""
    ## 🏆 Model Comparison Results
    
    ### Performance Metrics
    {mo.ui.altair_chart(metrics_chart)}
    
    ### Cross-Validation Results
    {mo.ui.altair_chart(cv_chart)}
    
    ### Detailed Results
    {results_table}
    """)
    return cv_chart, cv_data, metrics_chart, metrics_data, results_table


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