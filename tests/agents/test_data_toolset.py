"""Tests for the data_toolset (SPEC §9.3)."""

from __future__ import annotations

import mlflow
import pytest
from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets.data import data_toolset


class _Ctx:
    def __init__(self, deps):
        self.deps = deps


@pytest.fixture
def deps_with_task():
    state = FlowState(
        task_spec=TaskSpec(title="t", description="d"),
        mlflow_run_id="mlrun-1",
    )
    return FlowDeps(state=state, provenance_db_path=":memory:")


def test_duckdb_query_rejects_ddl(deps_with_task):
    fn = data_toolset.tools["duckdb_query"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps_with_task), sql="DELETE FROM tasks")
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps_with_task), sql="CREATE TABLE x (i INT)")


def test_duckdb_query_appends_limit(deps_with_task):
    fn = data_toolset.tools["duckdb_query"].function
    # empty tasks table → still works; returns 0 rows.
    result = fn(_Ctx(deps_with_task), sql="SELECT * FROM tasks", limit=5)
    assert result == []


def test_list_tables_returns_known_tables(deps_with_task):
    fn = data_toolset.tools["list_tables"].function
    names = fn(_Ctx(deps_with_task))
    assert "tasks" in names
    assert "agent_decisions" in names


def test_persist_agent_decision_writes_state_and_store(deps_with_task):
    fn = data_toolset.tools["persist_agent_decision"].function
    decision_id = fn(
        _Ctx(deps_with_task),
        agent="data",
        summary="Registered Burgers dataset.",
        tool="register_dataset",
    )
    state = deps_with_task.state
    assert len(state.decisions) == 1
    assert state.decisions[0].decision_id == decision_id
    assert state.decisions[0].agent == "data"

    rows = deps_with_task.provenance().query(
        "SELECT agent, tool FROM agent_decisions WHERE decision_id = ?",
        [decision_id],
    )
    assert rows[0]["agent"] == "data"
    assert rows[0]["tool"] == "register_dataset"


def test_persist_agent_decision_rejects_unknown_agent(deps_with_task):
    fn = data_toolset.tools["persist_agent_decision"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps_with_task), agent="bogus", summary="x")


def test_register_dataset_attaches_to_task_spec(deps_with_task):
    fn = data_toolset.tools["register_dataset"].function
    fn(
        _Ctx(deps_with_task),
        name="train",
        source="parquet",
        location="data/train.parquet",
        input_columns=["x", "t"],
        output_columns=["u"],
    )
    task = deps_with_task.state.task_spec
    assert len(task.available_data) == 1
    assert task.available_data[0].name == "train"

    rows = deps_with_task.provenance().query(
        "SELECT name, source FROM dataset_bindings WHERE task_id = ?",
        [task.task_id],
    )
    assert rows[0]["name"] == "train"
    assert rows[0]["source"] == "parquet"


def test_persist_artifact_ref_roundtrips(deps_with_task):
    fn = data_toolset.tools["persist_artifact_ref"].function
    uri = fn(
        _Ctx(deps_with_task),
        uri="runs:/abc/dataset/train.parquet",
        kind="dataset",
        label="burgers-v1",
    )
    assert uri == "runs:/abc/dataset/train.parquet"
    rows = deps_with_task.provenance().query(
        "SELECT kind, label FROM artifacts WHERE uri = ?", [uri]
    )
    assert rows[0]["kind"] == "dataset"
    assert rows[0]["label"] == "burgers-v1"


@pytest.fixture
def mlflow_runs(tmp_path):
    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("data-toolset-test")
    run_ids: list[str] = []
    for loss in (0.1, 0.05):
        with mlflow.start_run() as run:
            mlflow.log_metric("train_loss", loss)
            run_ids.append(run.info.run_id)
    yield run_ids


def test_list_runs_returns_ids(mlflow_runs, deps_with_task):
    fn = data_toolset.tools["list_runs"].function
    runs = fn(_Ctx(deps_with_task), experiment_name="data-toolset-test", limit=5)
    ids = {r["run_id"] for r in runs}
    assert set(mlflow_runs).issubset(ids)


def test_fetch_run_metrics(mlflow_runs, deps_with_task):
    fn = data_toolset.tools["fetch_run_metrics"].function
    metrics = fn(_Ctx(deps_with_task), run_id=mlflow_runs[0])
    assert metrics["train_loss"] == pytest.approx(0.1)


def test_load_observations_from_csv(tmp_path, deps_with_task):
    csv_path = tmp_path / "sensor.csv"
    csv_path.write_text("x,t,u\n0.1,0.0,0.05\n0.5,0.0,0.25\n0.9,0.0,0.45\n")
    fn = data_toolset.tools["load_observations_from_file"].function
    obs = fn(
        _Ctx(deps_with_task),
        path=str(csv_path),
        field="u",
        axes=["x", "t"],
    )
    assert obs["n_points"] == 3
    assert obs["points"] == [[0.1, 0.0], [0.5, 0.0], [0.9, 0.0]]
    assert obs["values"] == [[0.05], [0.25], [0.45]]
    assert obs["source"] == "data_file"
    # Dataset binding got attached to the task spec.
    assert any(b.name == "data" for b in deps_with_task.state.task_spec.available_data)


def test_load_observations_csv_rejects_missing_column(tmp_path, deps_with_task):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("x,u\n0.1,0.05\n")
    fn = data_toolset.tools["load_observations_from_file"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps_with_task), path=str(csv_path), field="u", axes=["x", "t"])


def test_load_observations_from_npz(tmp_path, deps_with_task):
    import numpy as np

    npz_path = tmp_path / "sensor.npz"
    np.savez(
        npz_path,
        x=np.array([0.0, 1.0, 2.0]),
        y=np.array([0.0, 0.5, 1.0]),
        u=np.array([0.0, 0.25, 1.0]),
    )
    fn = data_toolset.tools["load_observations_from_file"].function
    obs = fn(
        _Ctx(deps_with_task),
        path=str(npz_path),
        field="u",
        axes=["x", "y"],
    )
    assert obs["n_points"] == 3
    assert obs["points"][1] == [1.0, 0.5]
    assert obs["values"][2] == [1.0]


def test_generate_synthetic_observations_smoke(deps_with_task):
    fn = data_toolset.tools["generate_synthetic_observations"].function
    obs = fn(
        _Ctx(deps_with_task),
        truth_form="sin(pi*x)",
        axes=["x"],
        field="u",
        axis_bounds={"x": [0.0, 1.0]},
        n_points=32,
        noise_sigma=0.0,
        true_parameters={"pi": 3.141592653589793},
        seed=0,
    )
    assert obs["source"] == "synthetic"
    assert obs["n_points"] == 32
    # sin(pi*x) for x in [0,1] is in [0,1].
    for (_v,) in obs["values"]:
        assert 0.0 - 1e-6 <= _v <= 1.0 + 1e-6
