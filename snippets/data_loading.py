import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Loading with Polars""")
    return


@app.cell
def _():
    """Import required libraries"""
    import polars as pl
    import numpy as np
    from pathlib import Path

    return np, pl


@app.cell
def _(mo):
    """Configure data source"""
    data_path = mo.ui.file(
        filetypes = [".csv", ".parquet", ".json"],
        max_size = 1000*1024*1024
    )

    data_path
    return (data_path,)


app._unparsable_cell(
    r"""
    \"\"\"Load dataset based on file type\"\"\"
    def load_data(path):
        \"\"\"Load data with appropriate method\"\"\"
        if  == \"CSV\":
            return pl.read_csv(path)
        elif ftype == \"Parquet\":
            return pl.read_parquet(path)
        elif ftype == \"JSON\":
            return pl.read_json(path)
        else:
            raise ValueError(f\"Unsupported file type: {ftype}\")
    """,
    name="_"
)


@app.cell
def _(data_path, file_type, load_data, np, pl):


    # Try to load data or create sample
    try:
        df = load_data(data_path.value, file_type.value)
        data_source = f"Loaded from: {data_path.value}"
    except Exception as e:
        # Create sample dataset on error
        np.random.seed(42)
        df = pl.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "feature_3": np.random.randn(100) * 10 + 50,
            "category": np.random.choice(["A", "B", "C"], 100),
            "target": np.random.choice([0, 1], 100)
        })
        data_source = f"Generated sample dataset (Error: {str(e)})"

    print(f"✅ {data_source}")

    return (df,)


@app.cell
def _(df):
    """Extract dataset information"""
    dataset_info = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "columns": df.columns,
        "dtypes": dict(zip(df.columns, df.dtypes)),
        "null_counts": df.null_count().to_dicts()[0] if len(df) > 0 else {}
    }

    print(f"📊 Shape: {dataset_info['n_samples']:,} × {dataset_info['n_features']}")

    return


@app.cell
def _(df, pl):
    """Compute basic statistics"""
    # Get numeric columns
    numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]]

    if numeric_cols:
        stats_df = df.select(numeric_cols).describe()
    else:
        stats_df = pl.DataFrame({"message": ["No numeric columns found"]})

    return (stats_df,)


@app.cell
def _(df, mo, pl, stats_df):
    """Display data summary"""
    mo.md(f"""
    ## 📈 Dataset Overview

    ### Shape
    - **Rows:** {df.shape[0]:,}
    - **Columns:** {df.shape[1]}

    ### Column Names
    {', '.join(f'`{col}`' for col in df.columns)}

    ### Data Types
    {mo.ui.table(
        pl.DataFrame({
            "Column": df.columns,
            "Type": [str(dt) for dt in df.dtypes]
        })
    )}

    ### Numeric Statistics
    {mo.ui.table(stats_df)}
    """)
    return


@app.cell
def _(df):
    """Preview first few rows"""
    preview_df = df.head(10)
    return (preview_df,)


@app.cell
def _(mo, preview_df):
    """Display data preview"""
    mo.md(f"""
    ### 🔍 Data Preview (First 10 rows)
    {mo.ui.table(preview_df)}
    """)
    return


if __name__ == "__main__":
    app.run()
