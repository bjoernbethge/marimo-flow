import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interactive Parameter Exploration""")
    return


@app.cell
def _():
    """Import required libraries"""
    import altair as alt
    import numpy as np
    import polars as pl
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, auc, roc_curve
    from sklearn.model_selection import train_test_split

    return (
        RandomForestClassifier,
        accuracy_score,
        auc,
        make_classification,
        np,
        alt,
        pl,
        roc_curve,
        train_test_split,
    )


@app.cell
def _(mo):
    """Dataset parameters"""
    mo.md("## 📊 Dataset Configuration")
    return


@app.cell
def _(mo):
    """Dataset parameter controls"""
    n_samples = mo.ui.slider(
        start=100, stop=2000, step=100, value=1000,
        label="Number of samples"
    )

    n_features = mo.ui.slider(
        start=2, stop=20, step=1, value=10,
        label="Number of features"
    )

    n_informative = mo.ui.slider(
        start=1, stop=10, step=1, value=5,
        label="Informative features"
    )

    noise = mo.ui.slider(
        start=0.0, stop=1.0, step=0.1, value=0.1,
        label="Noise level"
    )

    return n_features, n_informative, n_samples, noise


@app.cell
def _(mo, n_features, n_informative, n_samples, noise):
    """Display dataset parameters"""
    mo.md(f"""
    ### Dataset Parameters
    {n_samples} {n_features} {n_informative} {noise}
    """)
    return


@app.cell
def _(make_classification, n_features, n_informative, n_samples, noise):
    """Generate dataset"""
    X, y = make_classification(
        n_samples=n_samples.value,
        n_features=n_features.value,
        n_informative=min(n_informative.value, n_features.value),
        n_redundant=max(0, n_features.value - n_informative.value - 1),
        flip_y=noise.value,
        random_state=42
    )

    print(f"✅ Dataset generated: {X.shape}")

    return X, y


@app.cell
def _(mo):
    """Model parameters"""
    mo.md("## 🤖 RandomForestClassifier Model Configuration")
    return


@app.cell
def _(mo):
    """Model parameter controls"""
    n_estimators = mo.ui.slider(
        start=10, stop=200, step=10, value=100,
        label="Number of trees"
    )

    max_depth = mo.ui.slider(
        start=1, stop=20, step=1, value=5,
        label="Maximum depth"
    )

    min_samples_split = mo.ui.slider(
        start=2, stop=20, step=1, value=2,
        label="Min samples split"
    )

    criterion = mo.ui.dropdown(
        options=["gini", "entropy"],
        value="gini",
        label="Split criterion"
    )

    return criterion, max_depth, min_samples_split, n_estimators


@app.cell
def _(criterion, max_depth, min_samples_split, mo, n_estimators):
    """Display model parameters"""
    mo.md(f"""
    ### Model Parameters
    {n_estimators} {max_depth} {min_samples_split} {criterion}
    """)
    return


@app.cell
def _(mo):
    """Training control"""
    mo.md("## 🎯 Training")
    return


@app.cell
def _(mo):
    """Test size control"""
    test_size = mo.ui.slider(
        start=0.1, stop=0.5, step=0.05, value=0.2,
        label="Test set size"
    )

    return (test_size,)


@app.cell
def _(mo, test_size):
    """Train button"""
    train_button = mo.ui.run_button(
        label=f"🚀 Train Model (test size: {test_size.value})"
    )

    mo.md(f"{train_button}")

    return (train_button,)


@app.cell
def _(X, mo, test_size, train_button, train_test_split, y):
    """Split data"""
    mo.stop(not train_button.value, "Click 'Train Model' to proceed")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size.value, random_state=42, stratify=y
    )

    print(f"✅ Data split: Train {len(X_train)}, Test {len(X_test)}")

    return X_test, X_train, y_test, y_train


@app.cell
def _(
    RandomForestClassifier,
    X_test,
    X_train,
    accuracy_score,
    criterion,
    max_depth,
    min_samples_split,
    n_estimators,
    y_test,
    y_train,
):
    """Train model"""
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=n_estimators.value,
        max_depth=max_depth.value,
        min_samples_split=min_samples_split.value,
        criterion=criterion.value,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Probabilities for ROC
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Accuracies
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"✅ Model trained")
    print(f"📈 Train accuracy: {train_acc:.4f}")
    print(f"📊 Test accuracy: {test_acc:.4f}")

    return model, test_acc, train_acc, y_prob_test


@app.cell
def _(auc, roc_curve, y_prob_test, y_test):
    """Calculate ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc = auc(fpr, tpr)

    return fpr, roc_auc, tpr


@app.cell
def _(fpr, alt, roc_auc, test_acc, tpr, train_acc, pl):
    """Visualize results with Altair"""
    # Accuracy bar chart
    acc_df = pl.DataFrame({
        "Set": ["Train", "Test"],
        "Accuracy": [train_acc, test_acc]
    })
    acc_chart = alt.Chart(acc_df.to_pandas()).mark_bar().encode(
        x=alt.X("Set", title="Set"),
        y=alt.Y("Accuracy", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Set", scale=alt.Scale(range=["green", "orange"])),
        tooltip=["Set", "Accuracy"]
    ).properties(
        width=200,
        height=300,
        title="Model Performance"
    )

    # ROC curve
    roc_df = pl.DataFrame({
        "FPR": fpr,
        "TPR": tpr
    })
    roc_chart = alt.Chart(roc_df.to_pandas()).mark_line(color="darkorange").encode(
        x=alt.X("FPR", scale=alt.Scale(domain=[0, 1]), title="False Positive Rate"),
        y=alt.Y("TPR", scale=alt.Scale(domain=[0, 1.05]), title="True Positive Rate"),
        tooltip=["FPR", "TPR"]
    ).properties(
        width=300,
        height=300,
        title=f"ROC Curve (AUC = {roc_auc:.2f})"
    ) + alt.Chart(pl.DataFrame({"FPR": [0, 1], "TPR": [0, 1]}).to_pandas()).mark_line(color="navy", strokeDash=[5,5])

    charts = alt.hconcat(acc_chart, roc_chart).resolve_scale(color="independent")
    charts.display()
    return


@app.cell
def _(model, n_features, np):
    """Feature importance"""
    feature_importance = model.feature_importances_
    feature_names = [f"Feature {i}" for i in range(n_features.value)]

    # Sort features by importance
    indices = np.argsort(feature_importance)[::-1]

    return feature_importance, feature_names, indices


@app.cell
def _(feature_importance, feature_names, indices, alt, pl):
    """Plot feature importance with Altair"""
    fi_df = pl.DataFrame({
        "Feature": [feature_names[i] for i in indices],
        "Importance": feature_importance[indices]
    })
    fi_chart = alt.Chart(fi_df.to_pandas()).mark_bar().encode(
        x=alt.X("Feature", sort="-y", title="Features"),
        y=alt.Y("Importance", title="Importance"),
        tooltip=["Feature", "Importance"]
    ).properties(
        width=500,
        height=300,
        title="Feature Importances"
    )
    fi_chart.display()
    return


@app.cell
def _(mo, roc_auc, test_acc, train_acc):
    """Summary"""
    mo.md(f"""
    ## 📊 Results Summary

    ### Performance Metrics
    - **Training Accuracy:** {train_acc:.4f}
    - **Test Accuracy:** {test_acc:.4f}
    - **ROC AUC:** {roc_auc:.4f}

    ### Interactive Features
    This notebook demonstrates:
    - **Reactive parameters** - Change sliders to see immediate updates
    - **Conditional execution** - Use run button to control expensive operations
    - **Dynamic visualizations** - Plots update with parameter changes
    - **Feature importance** - Understand model decisions
    """)
    return


if __name__ == "__main__":
    app.run()
