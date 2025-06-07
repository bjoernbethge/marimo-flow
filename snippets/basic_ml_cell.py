import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Basic ML Cell Template""")
    return

@app.cell
def _():
    """Basic ML cell with imports and setup"""
    import mlflow
    import polars as pl
    import altair as alt
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    
    # Your ML code here
    print("✅ ML libraries imported successfully")
    
    return mlflow, pl, alt, np, RandomForestClassifier

@app.cell
def _():
    import marimo as mo
    return (mo,)

if __name__ == "__main__":
    app.run() 