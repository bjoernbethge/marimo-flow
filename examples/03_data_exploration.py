import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    """Data Exploration with Marimo Flow"""

    import marimo as mo
    import polars as pl
    import altair as alt
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_diabetes
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import warnings

    warnings.filterwarnings("ignore")

    mo.md("""
    # 🔍 Data Exploration - Marimo Flow

    **Interactive data analysis and visualization**

    This notebook demonstrates:
    - 📊 **Dataset Loading**: Multiple built-in datasets
    - 🔍 **Statistical Analysis**: Descriptive statistics
    - 📈 **Visualizations**: Interactive charts
    - 🎯 **Dimensionality Reduction**: PCA, t-SNE
    """)
    return (
        PCA,
        StandardScaler,
        TSNE,
        alt,
        load_breast_cancer,
        load_diabetes,
        load_iris,
        load_wine,
        mo,
        np,
        pd,
        pl,
    )


@app.cell
def _(mo):
    """Dataset Selection"""

    dataset_choice = mo.ui.dropdown(
        options=["wine", "breast_cancer", "iris", "diabetes"],
        value="wine",
        label="📊 Choose Dataset",
    )

    analysis_type = mo.ui.dropdown(
        options=["overview", "correlations", "distributions", "dimensionality_reduction"],
        value="overview",
        label="🔍 Analysis Type",
    )

    mo.md(f"""
    ## 📋 Configuration
    {dataset_choice}
    {analysis_type}
    """)
    return analysis_type, dataset_choice


@app.cell
def _(
    dataset_choice,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    np,
    pl,
):
    """Load and Prepare Dataset"""

    dataset_loaders = {
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "iris": load_iris,
        "diabetes": load_diabetes,
    }

    data = dataset_loaders[dataset_choice.value]()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # Handle target names (some datasets don't have them)
    if hasattr(data, "target_names") and data.target_names is not None:
        target_names = data.target_names
        is_classification = True
    else:
        target_names = ["target"]
        is_classification = False

    # Create comprehensive DataFrame
    df = pl.DataFrame(
        {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
    )

    dataset_info = {
        "name": dataset_choice.value,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "n_classes": len(np.unique(y)) if is_classification else "continuous",
        "feature_names": feature_names,
        "target_names": target_names,
        "is_classification": is_classification,
    }

    f"📊 Dataset loaded: {dataset_info['name']} ({dataset_info['n_samples']} samples, {dataset_info['n_features']} features)"
    return X, dataset_info, df, target_names, y


@app.cell
def _(df):
    """Convert to Pandas DataFrame"""

    # Always create pandas DataFrame for other cells to use
    df_pandas = df.to_pandas()
    return (df_pandas,)


@app.cell
def _(dataset_info, df_pandas, mo):
    """Dataset Overview - Always Visible"""

    # Basic statistics
    stats = df_pandas.describe()

    # Missing values
    missing_info = df_pandas.isnull().sum()

    # Data types
    dtypes_info = df_pandas.dtypes

    mo.md(f"""
    ## 📊 Dataset Overview: {dataset_info["name"].title()}

    ### Basic Information
    - **Samples**: {dataset_info["n_samples"]:,}
    - **Features**: {dataset_info["n_features"]}
    - **Type**: {"Classification" if dataset_info["is_classification"] else "Regression"}
    - **Classes/Target**: {dataset_info["n_classes"]}

    ### Quick Stats
    - **Mean Target**: {df_pandas["target"].mean():.3f}
    - **Target Range**: {df_pandas["target"].min():.1f} - {df_pandas["target"].max():.1f}
    - **Missing Values**: {missing_info.sum()} total
    - **Complete Cases**: {len(df_pandas.dropna())} ({len(df_pandas.dropna()) / len(df_pandas) * 100:.1f}%)
    """)
    return (stats,)


@app.cell
def _(analysis_type, mo, stats):
    """Detailed Statistical Summary"""

    if analysis_type.value == "overview":
        mo.md(f"""
        ### 📈 Detailed Statistical Summary
        {mo.ui.table(stats.round(3))}
        """)
    else:
        mo.md("")
    return


@app.cell
def _(alt, dataset_info, df_pandas, mo, target_names):
    """Quick Data Visualization - Always Visible"""

    # Class/Target distribution
    if dataset_info["is_classification"]:
        class_counts = df_pandas["target"].value_counts().reset_index()
        class_counts["target_name"] = [target_names[i] for i in class_counts["target"]]

        target_chart = (
            alt.Chart(class_counts)
            .mark_bar()
            .encode(
                x=alt.X("count", title="Count"),
                y=alt.Y("target_name:N", title="Class"),
                color=alt.Color("target:N", scale=alt.Scale(scheme="category10")),
            )
            .properties(title="Class Distribution", width=300, height=200)
        )
    else:
        target_chart = (
            alt.Chart(df_pandas)
            .mark_bar()
            .encode(
                x=alt.X("target:Q", bin=alt.Bin(maxbins=20)),
                y="count()",
                color=alt.value("steelblue"),
            )
            .properties(title="Target Distribution", width=300, height=200)
        )

    # Feature scatter plot (first 2 features)
    feature_cols = [col for col in df_pandas.columns if col != "target"][:2]
    if len(feature_cols) >= 2:
        feature_scatter = (
            alt.Chart(df_pandas)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(f"{feature_cols[0]}:Q"),
                y=alt.Y(f"{feature_cols[1]}:Q"),
                color=alt.Color(
                    "target:N" if dataset_info["is_classification"] else "target:Q",
                    scale=alt.Scale(
                        scheme="category10"
                        if dataset_info["is_classification"]
                        else "viridis"
                    ),
                ),
                tooltip=[f"{feature_cols[0]}:Q", f"{feature_cols[1]}:Q", "target:Q"],
            )
            .interactive()
            .properties(
                title=f"{feature_cols[0]} vs {feature_cols[1]}", width=300, height=200
            )
        )
    else:
        feature_scatter = alt.Chart().mark_text(text="Not enough features for scatter plot")

    mo.md(f"""
    ## 📈 Quick Data Visualization

    {mo.ui.altair_chart(alt.hconcat(target_chart, feature_scatter))}
    """)
    return target_chart, feature_scatter


@app.cell
def _(analysis_type, df_pandas, mo, np):
    """Feature Distributions"""

    if analysis_type.value == "distributions":
        # Select numeric columns only
        numeric_cols = df_pandas.select_dtypes(include=[np.number]).columns.tolist()
        if "target" in numeric_cols:
            numeric_cols.remove("target")

        # Feature selection for distribution
        selected_features = mo.ui.multiselect(
            options=numeric_cols[:10],  # Limit to first 10 features
            value=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
            label="Select features to visualize",
        )

        mo.md(f"""
        ## 📈 Feature Distributions
        {selected_features}
        """)
    else:
        selected_features = None
        numeric_cols = []
        mo.md("Select 'distributions' in Analysis Type to see feature distributions")
    return (selected_features,)


@app.cell
def _(alt, analysis_type, df_pandas, mo, selected_features):
    """Distribution Visualizations"""

    if (
        analysis_type.value == "distributions"
        and selected_features
        and selected_features.value
    ):
        charts = []

        for feature in selected_features.value:
            # Histogram
            hist = (
                alt.Chart(df_pandas)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=30)),
                    y="count()",
                    color=alt.value("steelblue"),
                )
                .properties(title=f"{feature} Distribution", width=200, height=150)
            )
            charts.append(hist)

        # Combine charts
        if len(charts) > 0:
            combined_chart = alt.hconcat(*charts[:4]).resolve_scale(x="independent")

            mo.md(f"""
            ### Distribution Charts
            {mo.ui.altair_chart(combined_chart)}
            """)
        else:
            mo.md("")
    else:
        mo.md("")
    return


@app.cell
def _(analysis_type, df_pandas, mo, np):
    """Correlation Analysis"""

    if analysis_type.value == "correlations":
        # Calculate correlation matrix
        numeric_df = df_pandas.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        # Find highly correlated pairs
        high_corr_pairs = []
        for row_idx in range(len(corr_matrix.columns)):
            for col_idx in range(row_idx + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[row_idx, col_idx]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append(
                        {
                            "Feature 1": corr_matrix.columns[row_idx],
                            "Feature 2": corr_matrix.columns[col_idx],
                            "Correlation": corr_val,
                        }
                    )

        mo.md(f"""
        ## 🔗 Correlation Analysis

        ### Correlation Matrix
        {mo.ui.table(corr_matrix.round(3))}

        ### Highly Correlated Features (|r| > 0.7)
        {mo.ui.table(high_corr_pairs) if high_corr_pairs else "No highly correlated feature pairs found"}
        """)
    else:
        corr_matrix = None
        high_corr_pairs = None
        mo.md("Select 'correlations' in Analysis Type to see correlation analysis")
    return (corr_matrix,)


@app.cell
def _(alt, analysis_type, corr_matrix, mo):
    """Correlation Heatmap"""

    if analysis_type.value == "correlations" and corr_matrix is not None:
        # Prepare data for heatmap
        corr_data = []
        for x_idx, col1 in enumerate(corr_matrix.columns):
            for y_idx, col2 in enumerate(corr_matrix.columns):
                corr_data.append(
                    {
                        "Feature1": col1,
                        "Feature2": col2,
                        "Correlation": corr_matrix.iloc[x_idx, y_idx],
                        "x": x_idx,
                        "y": y_idx,
                    }
                )

        # Create heatmap
        heatmap = (
            alt.Chart(alt.InlineData(values=corr_data))
            .mark_rect()
            .encode(
                x=alt.X("Feature1:N", title=""),
                y=alt.Y("Feature2:N", title=""),
                color=alt.Color(
                    "Correlation:Q",
                    scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                    legend=alt.Legend(title="Correlation"),
                ),
                tooltip=["Feature1:N", "Feature2:N", "Correlation:Q"],
            )
            .properties(title="Feature Correlation Heatmap", width=400, height=400)
        )

        mo.md(f"""
        ### Correlation Heatmap
        {mo.ui.altair_chart(heatmap)}
        """)
    else:
        mo.md("")
    return


@app.cell
def _(analysis_type, mo):
    """Dimensionality Reduction Parameters"""

    if analysis_type.value == "dimensionality_reduction":
        reduction_method = mo.ui.dropdown(
            options=["PCA", "t-SNE"], value="PCA", label="Reduction Method"
        )

        n_components = mo.ui.slider(
            start=2, stop=3, step=1, value=2, label="Number of components"
        )

        standardize = mo.ui.checkbox(value=True, label="Standardize features")

        mo.md(f"""
        ## 🎯 Dimensionality Reduction
        {reduction_method}
        {n_components}
        {standardize}
        """)
    else:
        reduction_method = None
        n_components = None
        standardize = None
        mo.md(
            "Select 'dimensionality_reduction' in Analysis Type to see dimensionality reduction"
        )
    return n_components, reduction_method, standardize


@app.cell
def _(
    PCA,
    StandardScaler,
    TSNE,
    X,
    analysis_type,
    n_components,
    reduction_method,
    standardize,
):
    """Perform Dimensionality Reduction"""

    if analysis_type.value == "dimensionality_reduction" and reduction_method:
        # Prepare data
        X_processed = X.copy()

        if standardize.value:
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)

        # Apply reduction
        if reduction_method.value == "PCA":
            reducer = PCA(n_components=n_components.value, random_state=42)
            X_reduced = reducer.fit_transform(X_processed)

            # Calculate explained variance
            explained_variance = reducer.explained_variance_ratio_
            cumulative_variance = explained_variance.cumsum()

            reduction_info = {
                "method": "PCA",
                "explained_variance": explained_variance,
                "cumulative_variance": cumulative_variance,
                "total_variance": cumulative_variance[-1],
            }
        else:  # t-SNE
            reducer = TSNE(
                n_components=n_components.value,
                random_state=42,
                perplexity=min(30, len(X) // 4),
            )
            X_reduced = reducer.fit_transform(X_processed)

            reduction_info = {
                "method": "t-SNE",
                "explained_variance": None,
                "cumulative_variance": None,
                "total_variance": None,
            }

        f"✅ {reduction_method.value} completed: {X.shape} → {X_reduced.shape}"
    else:
        X_reduced = None
        reduction_info = None
        reducer = None
    return X_reduced, reduction_info


@app.cell
def _(
    X_reduced,
    alt,
    analysis_type,
    dataset_info,
    mo,
    n_components,
    pd,
    reduction_info,
    y,
):
    """Visualize Reduced Data"""

    if analysis_type.value == "dimensionality_reduction" and X_reduced is not None:
        # Create DataFrame for visualization
        if n_components.value == 2:
            reduced_df = pd.DataFrame(
                {
                    "Component 1": X_reduced[:, 0],
                    "Component 2": X_reduced[:, 1],
                    "target": y,
                }
            )

            # Create scatter plot
            if dataset_info["is_classification"]:
                reduction_scatter = (
                    alt.Chart(reduced_df)
                    .mark_circle(size=60, opacity=0.7)
                    .encode(
                        x=alt.X("Component 1:Q"),
                        y=alt.X("Component 2:Q"),
                        color=alt.Color("target:N", scale=alt.Scale(scheme="category10")),
                        tooltip=["Component 1:Q", "Component 2:Q", "target:N"],
                    )
                    .interactive()
                    .properties(
                        title=f"{reduction_info['method']} Visualization",
                        width=500,
                        height=400,
                    )
                )
            else:
                reduction_scatter = (
                    alt.Chart(reduced_df)
                    .mark_circle(size=60, opacity=0.7)
                    .encode(
                        x=alt.X("Component 1:Q"),
                        y=alt.X("Component 2:Q"),
                        color=alt.Color("target:Q", scale=alt.Scale(scheme="viridis")),
                        tooltip=["Component 1:Q", "Component 2:Q", "target:Q"],
                    )
                    .interactive()
                    .properties(
                        title=f"{reduction_info['method']} Visualization",
                        width=500,
                        height=400,
                    )
                )

        # Show results
        if reduction_info["method"] == "PCA":
            mo.md(f"""
            ### {reduction_info["method"]} Results

            **Explained Variance:**
            - Component 1: {reduction_info["explained_variance"][0]:.3f} ({reduction_info["explained_variance"][0] * 100:.1f}%)
            - Component 2: {reduction_info["explained_variance"][1]:.3f} ({reduction_info["explained_variance"][1] * 100:.1f}%)
            - **Total**: {reduction_info["total_variance"]:.3f} ({reduction_info["total_variance"] * 100:.1f}%)

            {mo.ui.altair_chart(reduction_scatter)}
            """)
        else:
            mo.md(f"""
            ### {reduction_info["method"]} Results

            t-SNE preserves local structure and reveals clusters in the data.

            {mo.ui.altair_chart(reduction_scatter)}
            """)
    else:
        mo.md("")
    return


@app.cell
def _(analysis_type, dataset_choice, mo):
    """Summary and Next Steps"""

    mo.md(f"""
    ## 🎯 Summary

    You've explored the **{dataset_choice.value}** dataset using **{analysis_type.value}** analysis.

    ### 🔄 Try Different Analyses:
    - **Overview**: Basic statistics and data quality
    - **Distributions**: Feature histograms and patterns  
    - **Correlations**: Feature relationships and dependencies
    - **Dimensionality Reduction**: PCA and t-SNE visualizations

    ### 💡 Next Steps:
    1. Switch between different analysis types above
    2. Try different datasets to compare patterns
    3. Use insights for feature engineering in ML models
    4. Export findings for further analysis
    """)
    return


if __name__ == "__main__":
    app.run()
