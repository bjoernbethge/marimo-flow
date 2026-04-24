"""Tests for the Validation toolset + node.

The toolset is wired to MLflow's real client so we test it against a
tmp file:// MLflow store with a seeded run — same pattern as
test_e2e.py.
"""

from __future__ import annotations

import mlflow
import pytest

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets.validation import validation_toolset


@pytest.fixture
def mlflow_run_with_metrics(tmp_path):
    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("validation-test")
    with mlflow.start_run() as run:
        mlflow.log_metric("train_loss", 0.05)
        mlflow.log_metric("val_mse", 0.02)
        yield run.info.run_id


class _Ctx:
    def __init__(self, deps):
        self.deps = deps


def test_evaluate_constraints_requires_training_run():
    from pydantic_ai import ModelRetry

    deps = FlowDeps(state=FlowState(), provenance_db_path=":memory:")
    fn = validation_toolset.tools["evaluate_constraints"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps), thresholds=None)


def test_evaluate_constraints_scores_thresholds(mlflow_run_with_metrics):
    state = FlowState(training_run_id=mlflow_run_with_metrics)
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    fn = validation_toolset.tools["evaluate_constraints"].function
    result = fn(
        _Ctx(deps),
        thresholds={"train_loss": 0.1, "val_mse": 0.01, "missing_metric": 1.0},
    )
    assert result["metrics"]["train_loss"] == pytest.approx(0.05)
    assert result["constraint_status"]["train_loss"] == "pass"
    assert result["constraint_status"]["val_mse"] == "fail"
    assert result["constraint_status"]["missing_metric"] == "warn"


def test_record_validation_writes_state_and_provenance(mlflow_run_with_metrics):
    task = TaskSpec(title="t", description="d")
    state = FlowState(
        training_run_id=mlflow_run_with_metrics,
        mlflow_run_id=mlflow_run_with_metrics,
        task_spec=task,
    )
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    fn = validation_toolset.tools["record_validation"].function

    report_id = fn(
        _Ctx(deps),
        verdict="accept",
        rationale="all metrics within thresholds",
        metrics={"train_loss": 0.05},
        constraint_status={"train_loss": "pass"},
    )

    assert state.validation_report is not None
    assert state.validation_report.verdict == "accept"
    assert state.validation_report.report_id == report_id
    assert state.validation_report.task_id == task.task_id

    rows = deps.provenance().query(
        "SELECT verdict FROM validation_reports WHERE report_id = ?",
        [report_id],
    )
    assert rows[0]["verdict"] == "accept"


def test_record_validation_rejects_unknown_verdict():
    from pydantic_ai import ModelRetry

    state = FlowState(training_run_id="x")
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    fn = validation_toolset.tools["record_validation"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps), verdict="maybe")


def test_record_validation_clamps_invalid_constraint_status(mlflow_run_with_metrics):
    state = FlowState(
        training_run_id=mlflow_run_with_metrics,
        mlflow_run_id=mlflow_run_with_metrics,
    )
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    fn = validation_toolset.tools["record_validation"].function
    fn(
        _Ctx(deps),
        verdict="accept",
        constraint_status={"train_loss": "ok", "val_mse": "pass"},
    )
    # invalid 'ok' must be clamped to 'warn'; valid 'pass' preserved.
    assert state.validation_report.constraint_status["train_loss"] == "warn"
    assert state.validation_report.constraint_status["val_mse"] == "pass"
