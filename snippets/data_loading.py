import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Data Loading with Polars""")
    return

@app.cell
def _(mo):
    """Data Source Configuration"""
    data_path = mo.ui.text(
        label="📁 Data Path",
        value="data/sample.csv"
    )
    
    mo.md(f"""
    ## 📊 Data Loading Configuration
    {data_path}
    """)
    return (data_path,)

@app.cell
def _(data_path):
    """Load and Analyze Dataset"""
    import polars as pl
    import numpy as np
    
    # Create sample data if file doesn't exist
    try:
        df = pl.read_csv(data_path.value)
        source = f"Loaded from: {data_path.value}"
    except:
        # Create sample dataset
        np.random.seed(42)
        df = pl.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
            "target": np.random.choice([0, 1], 100)
        })
        source = "Generated sample dataset"
    
    # Dataset info
    dataset_info = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "columns": df.columns,
        "dtypes": df.dtypes
    }
    
    print(f"✅ {source}")
    print(f"📊 Shape: {dataset_info['n_samples']} × {dataset_info['n_features']}")
    
    return df, dataset_info, pl, np

@app.cell
def _(df, mo):
    """Data Summary"""
    # Basic statistics
    stats = df.describe()
    
    mo.md(f"""
    ## 📈 Dataset Overview
    
    **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns
    
    **Columns:** {', '.join(df.columns)}
    
    ### Statistics
    {mo.ui.table(stats)}
    """)
    return (stats,)

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run() 