import marimo

__generated_with = "0.18.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # MLflow: Experiment Search & Visualization
        
        This snippet demonstrates how to search MLflow experiments and runs,
        convert them to Polars DataFrames, and visualize metric trends with Altair.
        """
    )
    return


@app.cell
def _():
    import altair as alt
    import mlflow
    import polars as pl
    from mlflow.tracking import MlflowClient
    from pathlib import Path
    from typing import Any
    
    # Setup MLflow
    tracking_path = Path("./data/experiments")
    if tracking_path.exists():
        mlflow.set_tracking_uri(f"file:{tracking_path.resolve()}")
    else:
        mlflow.set_tracking_uri("http://localhost:5000")
    
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    
    return Any, Path, alt, client, mlflow, pl


@app.cell
def _(client, pl):
    # List all experiments
    experiments = client.search_experiments()
    
    experiments_data = [
        {
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "lifecycle_stage": exp.lifecycle_stage,
        }
        for exp in experiments
    ]
    
    experiments_df = pl.DataFrame(experiments_data) if experiments_data else pl.DataFrame()
    
    return experiments, experiments_data, experiments_df


@app.cell
def _(client, experiments_df):
    # Select an experiment (use first one if available)
    if not experiments_df.is_empty():
        experiment_id = experiments_df["experiment_id"][0]
        
        # Search runs in the experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=100,
        )
    else:
        runs = []
        experiment_id = None
    
    return experiment_id, runs


@app.cell
def _(Any, pl, runs):
    # Convert runs to Polars DataFrame
    rows: list[dict[str, Any]] = []
    for run in runs:
        row: dict[str, Any] = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "status": run.info.status,
            "start_time": run.info.start_time,
        }
        # Flatten params
        for key, value in run.data.params.items():
            row[f"param_{key}"] = value
        # Flatten metrics
        for key, value in run.data.metrics.items():
            row[f"metric_{key}"] = value
        rows.append(row)
    
    runs_df = pl.DataFrame(rows) if rows else pl.DataFrame()
    
    return rows, runs_df


@app.cell
def _(alt, runs_df):
    import marimo as mo
    # Visualize metric trends
    if runs_df.is_empty():
        mo.md("_No runs available for visualization._")
        chart = None
    else:
        # Find metric columns
        metric_cols = [
            col for col in runs_df.columns if col.startswith("metric_")
        ]
        
        if metric_cols:
            # Use first metric for visualization
            metric_name = metric_cols[0]
            
            plot_df = (
                runs_df.select(["run_id", metric_name])
                .drop_nulls()
                .with_row_index(name="run_index")
            )
            
            if not plot_df.is_empty():
                chart = (
                    alt.Chart(plot_df.to_pandas())
                    .mark_line(point=True)
                    .encode(
                        x="run_index:Q",
                        y=alt.Y(metric_name, type="quantitative"),
                        tooltip=["run_id", metric_name],
                    )
                    .properties(title=f"Metric trend: {metric_name}")
                )
                chart.display()
            else:
                chart = None
                mo.md("_No metric data available._")
        else:
            chart = None
            mo.md("_No metrics found in runs._")
    
    return chart, metric_cols, metric_name, plot_df


if __name__ == "__main__":
    app.run()

