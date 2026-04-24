"""Tests for the DuckDB-backed ProvenanceStore (SPEC §12).

Uses an in-memory database so tests leave no files behind and can run
in parallel without a lock collision.
"""

from __future__ import annotations

import json

import pytest

from marimo_flow.agents.schemas import (
    AgentDecision,
    ArtifactRef,
    ConditionSpec,
    DatasetBinding,
    EquationSpec,
    ExperimentRecord,
    HandoffRecord,
    ModelSpec,
    ProblemSpec,
    RunConfig,
    SolverPlan,
    SubdomainSpec,
    TaskSpec,
    ValidationReport,
)
from marimo_flow.agents.services.provenance import ProvenanceStore


@pytest.fixture
def store():
    with ProvenanceStore(":memory:") as s:
        yield s


def test_schema_creates_expected_tables(store):
    rows = store.query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    )
    names = {r["table_name"] for r in rows}
    expected = {
        "tasks",
        "problem_specs",
        "model_specs",
        "solver_plans",
        "run_configs",
        "dataset_bindings",
        "artifacts",
        "experiments",
        "metrics",
        "agent_decisions",
        "handoff_records",
        "validation_reports",
        "lineage_edges",
    }
    assert expected.issubset(names)


def test_schema_init_is_idempotent(store):
    store._init_schema()  # should not raise on a second run
    rows = store.query("SELECT COUNT(*) AS c FROM tasks")
    assert rows[0]["c"] == 0


def test_record_task_and_read_back(store):
    task = TaskSpec(
        title="Burgers", description="1D PINN benchmark", review_required=True
    )
    store.record_task(task)
    rows = store.query("SELECT * FROM tasks WHERE task_id = ?", [task.task_id])
    assert len(rows) == 1
    assert rows[0]["title"] == "Burgers"
    assert rows[0]["review_required"] is True
    payload = json.loads(rows[0]["payload"])
    assert payload["description"] == "1D PINN benchmark"


def test_record_problem_spec(store):
    spec = ProblemSpec(
        name="heat_1d",
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [0.0, 1.0], "t": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(name="heat", form="u_t", outputs=["u"]),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="heat"),
        ],
    )
    # ``kind`` column tracks the ProblemSpec.name for display / grouping.
    store.record_problem_spec("task-a", spec)
    rows = store.query("SELECT * FROM problem_specs WHERE task_id = ?", ["task-a"])
    assert len(rows) == 1
    assert rows[0]["kind"] == "heat_1d"


def test_record_model_spec(store):
    store.record_model_spec("task-a", ModelSpec(kind="fno"))
    rows = store.query("SELECT kind FROM model_specs")
    assert rows[0]["kind"] == "fno"


def test_record_solver_plan(store):
    store.record_solver_plan("task-a", SolverPlan(kind="pinn", learning_rate=5e-4))
    rows = store.query("SELECT kind, learning_rate FROM solver_plans")
    assert rows[0]["kind"] == "pinn"
    assert rows[0]["learning_rate"] == pytest.approx(5e-4)


def test_record_run_config(store):
    store.record_run_config("task-a", RunConfig(max_epochs=42, accelerator="cpu"))
    rows = store.query("SELECT max_epochs, accelerator FROM run_configs")
    assert rows[0]["max_epochs"] == 42
    assert rows[0]["accelerator"] == "cpu"


def test_record_dataset_binding(store):
    store.record_dataset_binding(
        "task-a",
        DatasetBinding(name="train", source="parquet", location="data/train.parquet"),
    )
    rows = store.query("SELECT name, source FROM dataset_bindings")
    assert rows[0]["name"] == "train"
    assert rows[0]["source"] == "parquet"


def test_record_artifact_and_replace_on_conflict(store):
    ref = ArtifactRef(kind="problem", uri="runs:/abc/problem/spec.json", label="v1")
    store.record_artifact(ref, task_id="task-a")
    # Re-record with a new label — PRIMARY KEY (uri) must replace.
    ref2 = ArtifactRef(kind="problem", uri=ref.uri, label="v2")
    store.record_artifact(ref2, task_id="task-a")
    rows = store.query(
        "SELECT label FROM artifacts WHERE uri = ?", [ref.uri]
    )
    assert len(rows) == 1
    assert rows[0]["label"] == "v2"


def test_record_experiment_and_metrics(store):
    exp = ExperimentRecord(task_id="task-a", run_id="mlrun-1", status="completed")
    store.record_experiment(exp)
    store.record_metric(
        experiment_id=exp.experiment_id,
        run_id="mlrun-1",
        name="train_loss",
        value=0.0123,
    )
    store.record_metric(
        experiment_id=exp.experiment_id,
        run_id="mlrun-1",
        name="train_loss",
        value=0.0089,
        step=1,
    )
    rows = store.query(
        "SELECT name, value, step FROM metrics "
        "WHERE experiment_id = ? ORDER BY step",
        [exp.experiment_id],
    )
    assert len(rows) == 2
    assert rows[0]["name"] == "train_loss"
    assert rows[1]["step"] == 1


def test_record_decision(store):
    d = AgentDecision(
        agent="problem",
        tool="compose_problem",
        summary="Composed Burgers",
        task_id="task-a",
        run_id="mlrun-1",
    )
    store.record_decision(d)
    rows = store.query(
        "SELECT agent, tool, summary FROM agent_decisions "
        "WHERE decision_id = ?",
        [d.decision_id],
    )
    assert rows[0]["agent"] == "problem"
    assert rows[0]["tool"] == "compose_problem"


def test_record_handoff(store):
    h = HandoffRecord(
        from_agent="problem",
        to_agent="model",
        reason="Problem registered",
        task_id="task-a",
        artifact_uris=["runs:/abc/problem/spec.json"],
    )
    store.record_handoff(h)
    rows = store.query(
        "SELECT from_agent, to_agent, reason FROM handoff_records "
        "WHERE handoff_id = ?",
        [h.handoff_id],
    )
    assert rows[0]["from_agent"] == "problem"
    assert rows[0]["to_agent"] == "model"


def test_record_validation_report(store):
    report = ValidationReport(
        task_id="task-a",
        run_id="mlrun-1",
        metrics={"train_loss": 0.01},
        constraint_status={"train_loss_leq_0_05": "pass"},
        verdict="accept",
        rationale="train_loss well under threshold",
    )
    store.record_validation_report(report)
    rows = store.query(
        "SELECT verdict, rationale FROM validation_reports WHERE report_id = ?",
        [report.report_id],
    )
    assert rows[0]["verdict"] == "accept"


def test_record_lineage_edge(store):
    store.record_lineage_edge(
        from_uri="runs:/abc/problem/spec.json",
        to_uri="runs:/abc/model/spec.json",
        relation="produced_for",
    )
    rows = store.query(
        "SELECT relation FROM lineage_edges WHERE from_uri = ?",
        ["runs:/abc/problem/spec.json"],
    )
    assert rows[0]["relation"] == "produced_for"
