import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    from typing import Any

    import altair as alt
    import marimo as mo
    import mlflow
    import polars as pl
    from mlflow.tracking import MlflowClient

    return Any, MlflowClient, Path, alt, mlflow, mo, pl


@app.cell
def _(Path):
    def build_tracking_uri(root: str) -> str:
        """Resolve a human-friendly path to an MLflow tracking URI."""
        path = Path(root).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Tracking directory does not exist: {path}")
        return f"file:{path.as_posix()}"

    return (build_tracking_uri,)


@app.cell
def _(MlflowClient, pl):
    def list_experiments(client: MlflowClient) -> pl.DataFrame:
        experiments = client.search_experiments()
        rows = [
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    return (list_experiments,)


@app.cell
def _(Any, mlflow, pl):
    def runs_to_dataframe(runs: list[mlflow.entities.Run]) -> pl.DataFrame:
        rows: list[dict[str, Any]] = []
        for run in runs:
            row: dict[str, Any] = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
            }
            for key, value in run.data.params.items():
                row[f"param_{key}"] = value
            for key, value in run.data.metrics.items():
                row[f"metric_{key}"] = value
            rows.append(row)
        return pl.DataFrame(rows) if rows else pl.DataFrame()

    return (runs_to_dataframe,)


@app.cell
def _(mo):
    tracking_root = mo.ui.text(
        value="data/experiments",
        label="MLflow tracking directory",
    )
    reload_button = mo.ui.button("Connect / Refresh")
    mo.hstack([tracking_root, reload_button], justify="start")
    return reload_button, tracking_root


@app.cell
def _(MlflowClient, build_tracking_uri, mlflow, mo, reload_button, tracking_root):
    reload_button  # reactive dependency

    client = None
    connection_error = None

    try:
        uri = build_tracking_uri(tracking_root.value)
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
    except Exception as exc:
        connection_error = str(exc)

    if connection_error:
        mo.md(f"⚠️ Unable to connect: `{connection_error}`")

    return client, connection_error


@app.cell
def _(client, list_experiments, mo, pl):
    experiments_df = pl.DataFrame()
    exp_dropdown = None

    if client is None:
        mo.md("_Connect to a valid tracking directory to list experiments._")
    else:
        experiments_df = list_experiments(client)
        if experiments_df.is_empty():
            mo.md("_No experiments found._")
        else:
            exp_dropdown = mo.ui.dropdown(
                options=experiments_df["experiment_id"].to_list(),
                value=experiments_df["experiment_id"][0],
                label="Experiment ID",
            )
            mo.hstack([exp_dropdown])

    return exp_dropdown, experiments_df


@app.cell
def _(client, exp_dropdown, pl, runs_to_dataframe):
    runs_df = pl.DataFrame()

    if client is not None and exp_dropdown is not None and exp_dropdown.value:
        runs = client.search_runs(
            experiment_ids=[exp_dropdown.value],
            order_by=["attributes.start_time DESC"],
            max_results=200,
        )
        runs_df = runs_to_dataframe(runs)

    return (runs_df,)


@app.cell
def _(exp_dropdown, mo, runs_df):
    if not runs_df.is_empty() and exp_dropdown is not None:
        mo.md(
            f"### Runs for experiment `{exp_dropdown.value}` "
            f"({runs_df.height} found)"
        )
        mo.ui.table(runs_df.head(20))
    elif exp_dropdown is not None:
        mo.md("_No runs available for the selected experiment._")
    return


@app.cell
def _(alt, mo, runs_df):
    if runs_df.is_empty():
        mo.md("_Select an experiment to view metrics._")
    else:
        metric_cols = [col for col in runs_df.columns if col.startswith("metric_")]

        if not metric_cols:
            mo.md("_No metrics logged for these runs._")
        else:
            metric_select = mo.ui.dropdown(
                options=metric_cols,
                value=metric_cols[0],
                label="Metric to plot",
            )
            mo.hstack([metric_select])

            plot_df = (
                runs_df.select(["run_id", metric_select.value])
                .drop_nulls()
                .with_row_index(name="step")
            )

            if plot_df.is_empty():
                mo.md("_Metric is empty for displayed runs._")
            else:
                chart = (
                    alt.Chart(plot_df.to_pandas())
                    .mark_line(point=True)
                    .encode(
                        x="step:Q",
                        y=alt.Y(metric_select.value, type="quantitative"),
                        tooltip=["run_id", metric_select.value],
                    )
                    .properties(title=f"Metric trend: {metric_select.value}")
                )
                chart.display()
    return


if __name__ == "__main__":
    app.run()
