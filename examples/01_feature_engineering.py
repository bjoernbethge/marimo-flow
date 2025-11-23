# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "numpy>=1.26.4",
#     "plotly>=5.24.0",
#     "polars>=1.12.0",
#     "scikit-learn>=1.5.0",
#     "scipy>=1.13.0",
# ]
# ///
import marimo
import marimo as mo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports and Setup"""

    import altair as alt
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    import warnings
    from pathlib import Path
    from scipy import stats
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
    
    # Set modern Plotly theme
    px.defaults.template = "plotly_white"
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.feature_selection import (
        SelectKBest,
        f_classif,
        f_regression,
        mutual_info_regression,
    )
    from sklearn.preprocessing import (
        MinMaxScaler,
        PolynomialFeatures,
        PowerTransformer,
        RobustScaler,
        StandardScaler,
    )

    # Enable Altair for large datasets
    alt.data_transformers.enable("default", max_rows=None)
    warnings.filterwarnings("ignore")

    mo.md("""
    # 🛠️ Feature Engineering

    Interactive feature engineering pipeline with **Altair** visualizations
    """)
    return (
        Path,
        MinMaxScaler,
        PCA,
        PolynomialFeatures,
        PowerTransformer,
        RandomForestClassifier,
        RandomForestRegressor,
        RobustScaler,
        SelectKBest,
        StandardScaler,
        alt,
        f_classif,
        f_regression,
        go,
        load_breast_cancer,
        load_diabetes,
        load_iris,
        load_wine,
        mo,
        mutual_info_regression,
        np,
        pl,
        px,
        stats,
    )


@app.cell
def _(alt, pl):
    """Helper Functions"""


    def create_distribution_chart(df, feature, width=180, height=120):
        """Create a distribution chart for a single feature"""
        return (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=20)),
                y=alt.Y("count()"),
                tooltip=[alt.Tooltip(f"{feature}:Q", format=".2f"), "count()"],
            )
            .properties(width=width, height=height, title=feature[:20])
        )


    def create_comparison_boxplot(df_original, df_transformed, feature):
        """Create before/after boxplot comparison"""
        data = pl.concat(
            [
                pl.DataFrame(
                    {"value": df_original[feature], "type": "Original", "feature": feature}
                ),
                pl.DataFrame(
                    {
                        "value": df_transformed[feature],
                        "type": "Transformed",
                        "feature": feature,
                    }
                ),
            ]
        )

        return (
            alt.Chart(data)
            .mark_boxplot()
            .encode(
                x=alt.X("type:N", title=None),
                y=alt.Y("value:Q", title="Value"),
                color="type:N",
            )
            .properties(width=150, height=200, title=feature[:15])
        )


    def create_importance_chart(scores_df, top_n=15):
        """Create feature importance bar chart"""
        return (
            alt.Chart(scores_df.head(top_n))
            .mark_bar()
            .encode(
                x=alt.X("Score:Q", title="Importance Score"),
                y=alt.Y("Feature:N", sort="-x"),
                color=alt.Color(
                    "Selected:N",
                    scale=alt.Scale(domain=[True, False], range=["#2ca02c", "#d62728"]),
                    legend=alt.Legend(title="Selected"),
                ),
                tooltip=["Feature", alt.Tooltip("Score:Q", format=".3f"), "Rank:O"],
            )
            .properties(width=400, height=350, title="Feature Importance")
        )
    return (
        create_comparison_boxplot,
        create_distribution_chart,
        create_importance_chart,
    )


@app.cell
def _(mo):
    """Data Source Selection"""

    data_source = mo.ui.dropdown(
        options=["builtin", "file"],
        value="builtin",
        label="📊 Data Source",
    )

    dataset_selector = mo.ui.dropdown(
        options=["wine", "breast_cancer", "iris", "diabetes"],
        value="wine",
        label="📦 Built-in Dataset",
    )

    file_upload = mo.ui.file(
        filetypes=[".csv", ".parquet", ".json"],
        max_size=100 * 1024 * 1024,
        label="📁 Upload File",
    )

    mo.md(f"""
    ## 📋 Data Source
    
    {data_source}
    
    {dataset_selector if data_source.value == "builtin" else file_upload}
    """)
    return data_source, dataset_selector, file_upload


@app.cell(hide_code=True)
def _(
    Path,
    data_source,
    dataset_selector,
    file_upload,
    load_breast_cancer,
    load_diabetes,
    load_iris,
    load_wine,
    mo,
    np,
    pl,
):
    """Load Dataset - Cached for performance"""

    if data_source.value == "file" and file_upload.value:
        # Load from file
        file_path = Path(file_upload.value)
        
        if file_path.suffix == ".csv":
            df_original = pl.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            df_original = pl.read_parquet(file_path)
        elif file_path.suffix == ".json":
            df_original = pl.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Try to identify target column
        target_col = None
        for col in ["target", "label", "y", "class"]:
            if col in df_original.columns:
                target_col = col
                break
        
        if target_col is None:
            target_col = df_original.columns[-1]
        
        # Separate features and target
        feature_cols = [c for c in df_original.columns if c != target_col]
        X_raw = df_original.select(feature_cols).to_numpy()
        y_raw = df_original[target_col].to_numpy()
        feature_names = feature_cols
        
        # Create loaded_data-like object
        class DataObj:
            def __init__(self):
                self.data = X_raw
                self.target = y_raw
                self.feature_names = feature_names
        
        loaded_data = DataObj()
        is_classification_task = len(np.unique(y_raw)) < 20
        
    else:
        # Load built-in dataset
        dataset_loaders = {
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
            "iris": load_iris,
            "diabetes": load_diabetes,
        }

        loaded_data = dataset_loaders[dataset_selector.value]()
        X_raw, y_raw = loaded_data.data, loaded_data.target

        df_original = pl.DataFrame(X_raw, schema=loaded_data.feature_names)
        df_original = df_original.with_columns(pl.Series("target", y_raw))

        # Check if classification
        is_classification_task = len(np.unique(y_raw)) < 20

    info = mo.md(f"""
    **Dataset:** {dataset_selector.value if data_source.value == "builtin" else file_path.stem}  
    **Shape:** {df_original.shape[0]} samples × {df_original.shape[1] - 1} features  
    **Type:** {"Classification" if is_classification_task else "Regression"}
    """)
    return X_raw, df_original, info, is_classification_task, loaded_data, y_raw


@app.cell
def _(df_original, mo):
    mo.vstack(
        [mo.md("## 📋 Data Overview"), mo.ui.table(df_original.head(10), selection=None)]
    )
    return


@app.cell
def _(df_original, go, mo, px):
    """Feature Distribution Visualization with Plotly"""

    from plotly.subplots import make_subplots

    mo.md("## 📊 Feature Distributions")

    # Select first 6 features
    dist_features = df_original.columns[:6]
    if "target" in dist_features:
        dist_features = [f for f in dist_features if f != "target"][:6]

    n_features = len(dist_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=dist_features[:6],
        vertical_spacing=0.15,
    )

    for idx, feature in enumerate(dist_features[:6]):
        row = (idx // n_cols) + 1
        col = (idx % n_cols) + 1

        hist_data = df_original[feature].to_numpy()
        fig.add_trace(
            go.Histogram(
                x=hist_data,
                name=feature,
                marker_color=px.colors.qualitative.Set3[idx % len(px.colors.qualitative.Set3)],
                opacity=0.7,
                nbinsx=20,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template="plotly_white",
        height=200 * n_rows + 100,
        showlegend=False,
        font=dict(size=11),
        title_text="Feature Distributions",
    )
    fig.update_xaxes(title_text="Value")
    fig.update_yaxes(title_text="Frequency")

    mo.ui.plotly_chart(fig)
    return fig,


@app.cell
def _(mo):
    """Engineering Method Selection"""

    mo.md("""
    ## ⚙️ Feature Engineering

    Choose your transformation strategy:
    """)

    method_tabs = mo.ui.tabs(
        {
            "📏 Scaling": "scale",
            "🔄 Transformations": "transform",
            "➕ Create Features": "create",
            "🎯 Select Features": "select",
        }
    )

    method_tabs
    return (method_tabs,)


@app.cell
def _(method_tabs, mo):
    """Scaling Configuration"""

    if method_tabs.value == "scale":
        scaling_method = mo.ui.dropdown(
            options=["StandardScaler", "MinMaxScaler", "RobustScaler"],
            value="StandardScaler",
            label="Scaling Method",
        )

        mo.vstack(
            [
                mo.md("""
            ### Feature Scaling

            - **StandardScaler**: Mean=0, Std=1 (good for normal distributions)
            - **MinMaxScaler**: Scale to [0,1] range  
            - **RobustScaler**: Uses median/IQR (robust to outliers)
            """),
                scaling_method,
            ]
        )
    else:
        scaling_method = mo.ui.dropdown(options=["StandardScaler"], value="StandardScaler")
    return (scaling_method,)


@app.cell
def _(
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
    X_raw,
    loaded_data,
    method_tabs,
    mo,
    pl,
    scaling_method,
    y_raw,
):
    """Apply Scaling"""

    if method_tabs.value == "scale":
        scaler_map = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
        }

        selected_scaler = scaler_map[scaling_method.value]
        X_scaled_data = selected_scaler.fit_transform(X_raw)

        df_after_scaling = pl.DataFrame(X_scaled_data, schema=loaded_data.feature_names)
        df_after_scaling = df_after_scaling.with_column("target", y_raw)

        mo.md(f"✅ Applied **{scaling_method.value}**")
    else:
        df_after_scaling = None
        X_scaled_data = X_raw
    return (df_after_scaling,)


@app.cell
def _(
    df_after_scaling,
    df_original,
    go,
    method_tabs,
    mo,
    px,
):
    """Visualize Scaling Results with Plotly"""

    if method_tabs.value == "scale" and df_after_scaling is not None:
        from plotly.subplots import make_subplots

        # Compare first 3 features
        scale_features = [f for f in df_original.columns if f != "target"][:3]

        fig = make_subplots(
            rows=1,
            cols=len(scale_features),
            subplot_titles=scale_features,
            horizontal_spacing=0.1,
        )

        for idx, feat in enumerate(scale_features):
            original_data = df_original[feat].to_numpy()
            transformed_data = df_after_scaling[feat].to_numpy()

            fig.add_trace(
                go.Box(
                    y=original_data,
                    name="Original",
                    marker_color=px.colors.qualitative.Set3[0],
                    boxmean="sd",
                ),
                row=1,
                col=idx + 1,
            )

            fig.add_trace(
                go.Box(
                    y=transformed_data,
                    name="Transformed",
                    marker_color=px.colors.qualitative.Set3[1],
                    boxmean="sd",
                ),
                row=1,
                col=idx + 1,
            )

        fig.update_layout(
            template="plotly_white",
            height=300,
            showlegend=False,
            font=dict(size=11),
            title_text="Before vs After Scaling",
        )

        mo.ui.plotly_chart(fig)
    return


@app.cell
def _(df_original, method_tabs, mo):
    """Transformation Configuration"""

    if method_tabs.value == "transform":
        transform_method = mo.ui.dropdown(
            options=["log", "sqrt", "box-cox", "yeo-johnson"],
            value="yeo-johnson",
            label="Transformation Method",
        )

        transform_features = mo.ui.multiselect(
            options=list(df_original.columns[:-1])[:10],
            value=[df_original.columns[0]],
            label="Features to Transform",
        )

        mo.vstack(
            [
                mo.md("""
            ### Power Transformations

            Reduce skewness and normalize distributions:
            - **Log**: For right-skewed positive data
            - **Sqrt**: Moderate skew reduction
            - **Box-Cox**: Automatic optimization (positive values)
            - **Yeo-Johnson**: Like Box-Cox but handles negatives
            """),
                transform_method,
                transform_features,
            ]
        )
    else:
        transform_method = None
        transform_features = None
    return transform_features, transform_method


@app.cell
def _(
    PowerTransformer,
    X_raw,
    df_original,
    loaded_data,
    method_tabs,
    mo,
    np,
    pl,
    stats,
    transform_features,
    transform_method,
    y_raw,
):
    """Apply Transformations"""

    if method_tabs.value == "transform" and transform_features and transform_features.value:
        X_transformed_data = X_raw.copy()
        transformation_results = []

        for feat_name in transform_features.value:
            feat_idx = list(loaded_data.feature_names).index(feat_name)
            original_values = X_raw[:, feat_idx]

            # Apply transformation
            if transform_method.value == "log":
                if original_values.min() <= 0:
                    transformed_values = np.log1p(
                        original_values - original_values.min() + 1
                    )
                else:
                    transformed_values = np.log1p(original_values)
            elif transform_method.value == "sqrt":
                if original_values.min() < 0:
                    transformed_values = np.sqrt(original_values - original_values.min())
                else:
                    transformed_values = np.sqrt(original_values)
            elif transform_method.value in ["box-cox", "yeo-johnson"]:
                power_transformer = PowerTransformer(method=transform_method.value)
                transformed_values = power_transformer.fit_transform(
                    original_values.reshape(-1, 1)
                ).ravel()

            X_transformed_data[:, feat_idx] = transformed_values

            # Calculate improvement
            transformation_results.append(
                {
                    "Feature": feat_name,
                    "Original Skew": stats.skew(original_values),
                    "New Skew": stats.skew(transformed_values),
                    "Improvement": abs(stats.skew(original_values))
                    - abs(stats.skew(transformed_values)),
                }
            )

        df_after_transform = pl.DataFrame(
            X_transformed_data, schema=loaded_data.feature_names
        )
        df_after_transform = df_after_transform.with_columns(pl.Series("target", y_raw))

        transform_results_df = pl.DataFrame(transformation_results).round(3)
        mo.vstack(
            [
                mo.md(f"✅ Applied **{transform_method.value}** transformation"),
                mo.ui.table(transform_results_df),
            ]
        )
    else:
        df_after_transform = df_original.copy()
        X_transformed_data = X_raw
    return (df_after_transform,)


@app.cell
def _(
    df_after_transform,
    df_original,
    go,
    method_tabs,
    mo,
    px,
    transform_features,
):
    """Visualize Transformation Results with Plotly"""

    if method_tabs.value == "transform" and transform_features and transform_features.value:
        selected_feature = transform_features.value[0]

        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Before", "After"],
            horizontal_spacing=0.15,
        )

        # Original distribution
        original_data = df_original[selected_feature].to_numpy()
        fig.add_trace(
            go.Histogram(
                x=original_data,
                name="Original",
                marker_color=px.colors.qualitative.Set3[0],
                opacity=0.7,
                nbinsx=30,
            ),
            row=1,
            col=1,
        )

        # Transformed distribution
        transformed_data = df_after_transform[selected_feature].to_numpy()
        fig.add_trace(
            go.Histogram(
                x=transformed_data,
                name="Transformed",
                marker_color=px.colors.qualitative.Set3[1],
                opacity=0.7,
                nbinsx=30,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            template="plotly_white",
            height=300,
            showlegend=False,
            font=dict(size=12),
            title_text=f"Transformation: {selected_feature}",
        )
        fig.update_xaxes(title_text="Value")
        fig.update_yaxes(title_text="Frequency")

        mo.ui.plotly_chart(fig)
    return


@app.cell
def _(method_tabs, mo):
    """Feature Creation Configuration"""

    if method_tabs.value == "create":
        creation_method = mo.ui.dropdown(
            options=["polynomial", "interactions"],
            value="polynomial",
            label="Creation Method",
        )

        polynomial_degree = (
            mo.ui.slider(start=2, stop=3, value=2, label="Polynomial Degree")
            if creation_method.value == "polynomial"
            else None
        )

        mo.vstack(
            [
                mo.md("""
            ### Create New Features

            - **Polynomial**: Add polynomial terms (x², x³, xy, etc.)
            - **Interactions**: Multiply feature pairs
            """),
                creation_method,
                polynomial_degree if polynomial_degree else mo.md(""),
            ]
        )
    else:
        creation_method = None
        polynomial_degree = None
    return creation_method, polynomial_degree


@app.cell
def _(
    PolynomialFeatures,
    X_raw,
    creation_method,
    df_original,
    loaded_data,
    method_tabs,
    mo,
    pl,
    polynomial_degree,
    y_raw,
):
    """Apply Feature Creation"""

    if method_tabs.value == "create" and creation_method:
        if creation_method.value == "polynomial" and polynomial_degree:
            # Use first 3 features for polynomial
            poly_transformer = PolynomialFeatures(
                degree=polynomial_degree.value, include_bias=False
            )
            X_poly_features = poly_transformer.fit_transform(X_raw[:, :3])

            poly_feature_names = poly_transformer.get_feature_names_out(
                loaded_data.feature_names[:3]
            )
            df_with_poly = pl.DataFrame(X_poly_features, schema=poly_feature_names)
            df_with_poly = df_with_poly.with_columns(pl.Series("target", y_raw))

            mo.md(f"""
            ✅ Created **{len(poly_feature_names)}** polynomial features from first 3 features

            New features include: {", ".join(poly_feature_names[3:8])}...
            """)
        else:
            df_with_poly = df_original.copy()
    else:
        df_with_poly = df_original.copy()
    return (df_with_poly,)


@app.cell
def _(df_with_poly, method_tabs, mo):
    """Show Created Features Table"""

    if method_tabs.value == "create":
        mo.ui.table(df_with_poly.head(10), selection=None)
    return


@app.cell
def _(loaded_data, method_tabs, mo):
    """Feature Selection Configuration"""

    if method_tabs.value == "select":
        n_features_select = mo.ui.slider(
            start=1,
            stop=min(15, len(loaded_data.feature_names)),
            value=min(5, len(loaded_data.feature_names)),
            step=1,
            label="Number of Features",
        )

        selection_algorithm = mo.ui.dropdown(
            options=["f_score", "mutual_info", "random_forest"],
            value="f_score",
            label="Selection Method",
        )

        mo.vstack(
            [
                mo.md("""
            ### Feature Selection

            Select the most informative features:
            - **F-Score**: Statistical test (fast)
            - **Mutual Info**: Captures non-linear relationships
            - **Random Forest**: Tree-based importance
            """),
                selection_algorithm,
                n_features_select,
            ]
        )
    else:
        n_features_select = None
        selection_algorithm = None
    return n_features_select, selection_algorithm


@app.cell
def _(
    RandomForestClassifier,
    RandomForestRegressor,
    SelectKBest,
    X_raw,
    f_classif,
    f_regression,
    is_classification_task,
    loaded_data,
    method_tabs,
    mo,
    mutual_info_regression,
    n_features_select,
    pl,
    selection_algorithm,
    y_raw,
):
    """Apply Feature Selection"""

    if method_tabs.value == "select" and selection_algorithm and n_features_select:
        if selection_algorithm.value == "f_score":
            if is_classification_task:
                feature_selector = SelectKBest(
                    score_func=f_classif, k=n_features_select.value
                )
            else:
                feature_selector = SelectKBest(
                    score_func=f_regression, k=n_features_select.value
                )
            feature_selector.fit(X_raw, y_raw)
            feature_scores = feature_selector.scores_

        elif selection_algorithm.value == "mutual_info":
            feature_selector = SelectKBest(
                score_func=mutual_info_regression, k=n_features_select.value
            )
            feature_selector.fit(X_raw, y_raw)
            feature_scores = feature_selector.scores_

        elif selection_algorithm.value == "random_forest":
            if is_classification_task:
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_raw, y_raw)
            feature_scores = rf_model.feature_importances_

            # Create selector for consistency
            feature_selector = SelectKBest(k=n_features_select.value)
            feature_selector.scores_ = feature_scores
            feature_selector.fit(X_raw, y_raw)

        # Create results dataframe
        selection_results_df = pl.DataFrame(
            {
                "Feature": loaded_data.feature_names,
                "Score": feature_scores,
                "Rank": pl.Series(feature_scores).rank(ascending=False).cast(pl.Int64),
                "Selected": feature_selector.get_support(),
            }
        ).sort_by("Score", descending=True)

        selected_feature_list = selection_results_df[selection_results_df["Selected"]][
            "Feature"
        ].to_list()

        mo.md(
            f"✅ Selected **{len(selected_feature_list)}** features using **{selection_algorithm.value}**"
        )
    else:
        selection_results_df = None
        selected_feature_list = []
    return (selection_results_df,)


@app.cell
def _(method_tabs, mo, pl, px, selection_results_df):
    """Visualize Feature Selection Results with Plotly"""

    if method_tabs.value == "select" and selection_results_df is not None:
        # Create feature importance chart with Plotly
        top_features = selection_results_df.head(15)
        
        importance_fig = px.bar(
            top_features.to_pandas(),
            x="Score",
            y="Feature",
            orientation="h",
            color="Selected",
            title="Feature Importance",
            labels={"Score": "Importance Score", "Feature": "Feature"},
            color_discrete_map={True: "#2ca02c", False: "#d62728"},
            height=400,
        )
        importance_fig.update_layout(
            template="plotly_white",
            font=dict(size=12),
            yaxis={"categoryorder": "total ascending"},
        )

        mo.ui.plotly_chart(importance_fig)
    return


@app.cell
def _(method_tabs, mo, selection_results_df):
    """Feature Selection Results Table"""

    if method_tabs.value == "select" and selection_results_df is not None:
        mo.md("### Selected Features")
        mo.ui.table(
            selection_results_df[selection_results_df["Selected"]][
                ["Feature", "Score", "Rank"]
            ].round(3)
        )
    return


@app.cell
def _(df_original, mo, pl, px):
    """Target Correlation Analysis with Plotly"""

    # Calculate correlation with target (Polars)
    feature_cols = [col for col in df_original.columns if col != "target"]
    correlations = []
    
    for feat in feature_cols:
        corr_val = df_original.select([
            pl.corr(feat, "target")
        ]).item()
        correlations.append({
            "Feature": feat,
            "Correlation": corr_val
        })
    
    correlation_df = pl.DataFrame(correlations).sort(
        pl.col("Correlation").abs(), descending=True
    ).head(10)

    # Create correlation chart with Plotly
    target_corr_fig = px.bar(
        correlation_df.to_pandas(),
        x="Correlation",
        y="Feature",
        orientation="h",
        title="Top 10 Feature Correlations with Target",
        labels={"Correlation": "Correlation", "Feature": "Feature"},
        color="Correlation",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        height=300,
    )
    target_corr_fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        yaxis={"categoryorder": "total ascending"},
        xaxis=dict(range=[-1, 1]),
    )

    mo.vstack([
        mo.md("## 🔗 Feature Correlations with Target"),
        mo.ui.plotly_chart(target_corr_fig)
    ])
    return correlation_df, target_corr_fig


@app.cell(hide_code=True)
def _(PCA, X_raw, loaded_data, mo, pl, px):
    """PCA Analysis with Plotly"""

    # Apply PCA
    pca_transformer = PCA(n_components=2)
    X_pca_transformed = pca_transformer.fit_transform(X_raw)

    # Create PCA dataframe (Polars)
    pca_results_df = pl.DataFrame(
        {
            "PC1": X_pca_transformed[:, 0],
            "PC2": X_pca_transformed[:, 1],
            "target": loaded_data.target,
        }
    )

    # Create scatter plot with Plotly
    pca_fig = px.scatter(
        pca_results_df.to_pandas(),
        x="PC1",
        y="PC2",
        color="target",
        title=f"PCA: First Two Components (PC1: {pca_transformer.explained_variance_ratio_[0]:.1%}, PC2: {pca_transformer.explained_variance_ratio_[1]:.1%})",
        labels={
            "PC1": f"PC1 ({pca_transformer.explained_variance_ratio_[0]:.1%} var)",
            "PC2": f"PC2 ({pca_transformer.explained_variance_ratio_[1]:.1%} var)",
        },
        color_continuous_scale="viridis",
        height=400,
        opacity=0.7,
    )
    pca_fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
    )
    pca_fig.update_traces(marker=dict(size=5))

    mo.vstack(
        [
            mo.md("## 🎯 PCA Analysis"),
            mo.md(f"""
        **Explained Variance:**
        - PC1: {pca_transformer.explained_variance_ratio_[0]:.1%}
        - PC2: {pca_transformer.explained_variance_ratio_[1]:.1%}
        - Total: {sum(pca_transformer.explained_variance_ratio_[:2]):.1%}
        """),
            mo.ui.plotly_chart(pca_fig),
        ]
    )
    return pca_fig,


@app.cell
def _(df_original, mo):
    """Interactive Data Transformer"""

    interactive_transformer = mo.vstack(
        [
            mo.md("""
        ## 🔧 Interactive Data Transformer

        Transform your data interactively - no coding required!
        """),
            mo.ui.dataframe(df_original),
        ]
    )
    interactive_transformer
    return


@app.cell
def _(mo):
    """Summary"""

    mo.md("""
    ## 📝 Summary

    This notebook provides interactive feature engineering tools:

    1. **📊 Data Exploration** - Interactive visualizations with Altair
    2. **📏 Feature Scaling** - Normalize features for ML algorithms
    3. **🔄 Transformations** - Reduce skewness and normalize distributions
    4. **➕ Feature Creation** - Generate polynomial and interaction features
    5. **🎯 Feature Selection** - Select most informative features
    6. **🔗 Correlation Analysis** - Understand feature relationships
    7. **📉 PCA** - Dimensionality reduction and visualization

    All charts are **interactive** - try brushing and selecting data points!
    """)
    return


if __name__ == "__main__":
    app.run()
