# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "numpy>=1.26.4",
#     "plotly>=5.24.0",
#     "polars[async,database,pyarrow,pydantic,sqlalchemy]>=1.12.0",
#     "scikit-learn>=1.5.0",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """Data Exploration"""

    import warnings

    import altair as alt
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from pathlib import Path
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    # Set modern Plotly theme
    px.defaults.template = "plotly_white"
    px.defaults.color_continuous_scale = "viridis"

    warnings.filterwarnings("ignore")

    mo.md("""
    # 🔍 Data Exploration

    **Interactive data analysis and visualization**

    Overview:
    - 📊 **Dataset Loading**: Multiple built-in datasets
    - 🔍 **Statistical Analysis**: Descriptive statistics
    - 📈 **Visualizations**: Interactive charts
    - 🎯 **Dimensionality Reduction**: PCA, t-SNE
    """)
    return (
        PCA,
        Path,
        StandardScaler,
        TSNE,
        alt,
        go,
        load_breast_cancer,
        load_diabetes,
        load_iris,
        load_wine,
        mo,
        np,
        pl,
        px,
    )


@app.cell
def _(mo):
    """Data Source Selection"""

    data_source = mo.ui.dropdown(
        options=["builtin", "file"],
        value="builtin",
        label="📊 Data Source",
    )

    dataset_choice = mo.ui.dropdown(
        options=["wine", "breast_cancer", "iris", "diabetes"],
        value="wine",
        label="📦 Built-in Dataset",
    )

    file_upload = mo.ui.file(
        filetypes=[".csv", ".parquet", ".json"],
        max_size=100 * 1024 * 1024,
        label="📁 Upload File",
    )

    analysis_type = mo.ui.dropdown(
        options=["overview", "correlations", "distributions", "dimensionality_reduction"],
        value="overview",
        label="🔍 Analysis Type",
    )

    mo.md(f"""
    ## 📋 Configuration
    
    {data_source}
    
    {dataset_choice if data_source.value == "builtin" else file_upload}
    
    {analysis_type}
    """)
    return analysis_type, data_source, dataset_choice, file_upload


@app.cell
def _(
    Path,
    data_source,
    dataset_choice,
    file_upload,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    mo,
    np,
    pl,
):
    """Load and Prepare Dataset"""

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
            # Use last column as target
            target_col = df.columns[-1]
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col]
        X = df.select(feature_cols).to_numpy()
        y = df[target_col].to_numpy()
        feature_names = feature_cols
        
        # Determine if classification
        unique_targets = len(np.unique(y))
        is_classification = unique_targets < 20 and unique_targets > 1
        
        if is_classification:
            target_names = [str(i) for i in np.unique(y)]
        else:
            target_names = ["target"]
        
        dataset_info = {
            "name": file_path.stem,
            "n_samples": len(X),
            "n_features": len(feature_names),
            "n_classes": unique_targets if is_classification else "continuous",
            "feature_names": feature_names,
            "target_names": target_names,
            "is_classification": is_classification,
        }
    else:
        # Load built-in dataset
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

        dataset_info = {
            "name": dataset_choice.value,
            "n_samples": len(X),
            "n_features": len(feature_names),
            "n_classes": len(np.unique(y)) if is_classification else "continuous",
            "feature_names": feature_names,
            "target_names": target_names,
            "is_classification": is_classification,
        }
    
    # Create comprehensive DataFrame
    df = pl.DataFrame(
        {**{name: X[:, i] for i, name in enumerate(feature_names)}, "target": y}
    )

    return X, dataset_info, df, target_names, y


@app.cell
def _(df):
    """DataFrame ready for use"""
    # Polars DataFrame - no conversion needed
    return


@app.cell
def _(dataset_info, df, mo, pl):
    """Dataset Overview - Always Visible"""

    # Basic statistics (Polars)
    stats = df.describe()

    # Missing values
    missing_count = df.null_count().sum_horizontal().item()

    # Target stats
    target_mean = df["target"].mean()
    target_min = df["target"].min()
    target_max = df["target"].max()
    complete_cases = len(df.drop_nulls())
    total_cases = len(df)

    dataset_overview = mo.md(f"""
    ## 📊 Dataset Overview: {dataset_info["name"].title()}

    ### Basic Information
    - **Samples**: {dataset_info["n_samples"]:,}
    - **Features**: {dataset_info["n_features"]}
    - **Type**: {"Classification" if dataset_info["is_classification"] else "Regression"}
    - **Classes/Target**: {dataset_info["n_classes"]}

    ### Quick Stats
    - **Mean Target**: {target_mean:.3f}
    - **Target Range**: {target_min:.1f} - {target_max:.1f}
    - **Missing Values**: {missing_count} total
    - **Complete Cases**: {complete_cases} ({complete_cases / total_cases * 100:.1f}%)
    """)
    mo.sidebar([dataset_overview])
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
def _(dataset_info, df, go, mo, px, target_names, pl):
    """Quick Data Visualization - Always Visible"""

    # Class/Target distribution with Plotly (Polars)
    if dataset_info["is_classification"]:
        class_counts = df["target"].value_counts().sort("target")
        class_counts_pd = class_counts.to_pandas()
        class_counts_pd["target_name"] = [target_names[i] for i in class_counts_pd["target"]]

        target_fig = px.bar(
            class_counts_pd,
            x="count",
            y="target_name",
            color="target",
            title="Class Distribution",
            labels={"count": "Count", "target_name": "Class"},
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=300,
        )
        target_fig.update_layout(
            template="plotly_white",
            showlegend=False,
            font=dict(size=12),
        )
    else:
        target_fig = px.histogram(
            df.to_pandas(),
            x="target",
            nbins=30,
            title="Target Distribution",
            labels={"target": "Target Value", "count": "Frequency"},
            color_discrete_sequence=["#636EFA"],
            height=300,
        )
        target_fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
        )

    # Feature scatter plot (first 2 features) with Plotly
    feature_cols = [col for col in df.columns if col != "target"][:2]
    if len(feature_cols) >= 2:
        color_col = "target"
        if dataset_info["is_classification"]:
            scatter_fig = px.scatter(
                df.to_pandas(),
                x=feature_cols[0],
                y=feature_cols[1],
                color=color_col,
                title=f"{feature_cols[0]} vs {feature_cols[1]}",
                labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]},
                color_discrete_sequence=px.colors.qualitative.Set3,
                height=300,
                opacity=0.7,
            )
        else:
            scatter_fig = px.scatter(
                df.to_pandas(),
                x=feature_cols[0],
                y=feature_cols[1],
                color=color_col,
                title=f"{feature_cols[0]} vs {feature_cols[1]}",
                labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]},
                color_continuous_scale="viridis",
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
        ## 📈 Quick Data Visualization

        {mo.ui.plotly_chart(target_fig)}
        {mo.ui.plotly_chart(scatter_fig)}
        """)
    else:
        mo.md(f"""
        ## 📈 Quick Data Visualization

        {mo.ui.plotly_chart(target_fig)}
        """)
    return scatter_fig, target_fig


@app.cell
def _(analysis_type, df, mo, pl):
    """Feature Distributions"""

    if analysis_type.value == "distributions":
        # Select numeric columns only (Polars)
        numeric_cols = [
            col for col in df.columns 
            if col != "target" and df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]
        ]
        
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
def _(analysis_type, df, mo, px, selected_features):
    """Distribution Visualizations with Plotly"""

    if (
        analysis_type.value == "distributions"
        and selected_features
        and selected_features.value
    ):
        from plotly.subplots import make_subplots
        
        n_features = len(selected_features.value)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=selected_features.value[:4],
            vertical_spacing=0.15,
        )
        
        for idx, feature in enumerate(selected_features.value[:4]):
            row = (idx // n_cols) + 1
            col = (idx % n_cols) + 1
            
            hist_data = df[feature].to_numpy()
            fig.add_trace(
                go.Histogram(
                    x=hist_data,
                    name=feature,
                    marker_color=px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)],
                    opacity=0.7,
                ),
                row=row,
                col=col,
            )
        
        fig.update_layout(
            template="plotly_white",
            height=150 * n_rows + 100,
            showlegend=False,
            font=dict(size=11),
        )
        fig.update_xaxes(title_text="Value")
        fig.update_yaxes(title_text="Frequency")

        mo.md(f"""
        ### Distribution Charts
        {mo.ui.plotly_chart(fig)}
        """)
    else:
        mo.md("")
    return


@app.cell
def _(analysis_type, df, mo, pl):
    """Correlation Analysis"""

    if analysis_type.value == "correlations":
        # Calculate correlation matrix (Polars)
        numeric_cols = [
            col for col in df.columns 
            if df[col].dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]
        ]
        numeric_df = df.select(numeric_cols)
        corr_matrix = numeric_df.corr()

        # Find highly correlated pairs (Polars)
        high_corr_pairs = []
        corr_matrix_pd = corr_matrix.to_pandas()  # Convert for easier iteration
        for row_idx in range(len(corr_matrix_pd.columns)):
            for col_idx in range(row_idx + 1, len(corr_matrix_pd.columns)):
                corr_val = corr_matrix_pd.iloc[row_idx, col_idx]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append(
                        {
                            "Feature 1": corr_matrix_pd.columns[row_idx],
                            "Feature 2": corr_matrix_pd.columns[col_idx],
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
def _(analysis_type, corr_matrix, mo, px):
    """Correlation Heatmap"""

    import plotly.graph_objects as go

    if analysis_type.value == "correlations" and corr_matrix is not None:
        # Create modern Plotly heatmap (convert Polars to numpy)
        corr_matrix_np = corr_matrix.to_numpy()
        corr_cols = corr_matrix.columns
        
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix_np,
                x=corr_cols,
                y=corr_cols,
                colorscale="RdBu",
                zmid=0,
                text=corr_matrix_np.round(2),
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation"),
            )
        )
        
        heatmap_fig.update_layout(
            title="Feature Correlation Heatmap",
            template="plotly_white",
            width=600,
            height=600,
            font=dict(size=11),
            xaxis=dict(side="bottom"),
        )

        mo.md(f"""
        ### Correlation Heatmap
        {mo.ui.plotly_chart(heatmap_fig)}
        """)
    else:
        mo.md("")
    return heatmap_fig,


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
    pl,
    reduction_info,
    y,
):
    """Visualize Reduced Data"""

    if analysis_type.value == "dimensionality_reduction" and X_reduced is not None:
        # Create DataFrame for visualization
        if n_components.value == 2:
            reduced_df = pl.DataFrame(
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
