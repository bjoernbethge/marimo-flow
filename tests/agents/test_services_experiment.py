"""Tests for services/experiment.py lifecycle helpers."""

from __future__ import annotations

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import TaskSpec, ValidationReport
from marimo_flow.agents.services.experiment import (
    complete_experiment,
    start_experiment,
)
from marimo_flow.agents.state import FlowState


def test_start_experiment_opens_record_with_running_status():
    state = FlowState(
        user_intent="x",
        mlflow_run_id="ml-1",
        task_spec=TaskSpec(title="t", description="d"),
    )
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    record = start_experiment(state, deps)

    assert record.status == "running"
    assert record.task_id == state.task_spec.task_id
    assert state.experiment_id == record.experiment_id

    rows = deps.provenance().query(
        "SELECT status, task_id FROM experiments WHERE experiment_id = ?",
        [record.experiment_id],
    )
    assert rows[0]["status"] == "running"


def test_start_experiment_allows_missing_task_spec():
    state = FlowState(user_intent="x")
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    record = start_experiment(state, deps)
    assert record.task_id is None
    assert record.status == "running"


def test_complete_experiment_marks_completed_after_training():
    state = FlowState(
        mlflow_run_id="ml-1",
        training_run_id="ml-1",
        training_artifact_uri="runs:/ml-1/training/x.json",
        task_spec=TaskSpec(title="t", description="d"),
    )
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    record = start_experiment(state, deps)

    updated = complete_experiment(record, state, deps)

    assert updated.experiment_id == record.experiment_id
    assert updated.status == "completed"
    assert updated.finished_at is not None
    assert updated.training_artifact_uri == "runs:/ml-1/training/x.json"

    rows = deps.provenance().query(
        "SELECT status FROM experiments WHERE experiment_id = ?",
        [record.experiment_id],
    )
    # INSERT OR REPLACE keeps one row with the updated status.
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"


def test_complete_experiment_marks_failed_on_explicit_status():
    state = FlowState(task_spec=TaskSpec(title="t", description="d"))
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    record = start_experiment(state, deps)
    updated = complete_experiment(record, state, deps, status="failed")
    assert updated.status == "failed"


def test_complete_experiment_picks_failed_on_escalation():
    state = FlowState(
        training_run_id="ml-1",
        validation_report=ValidationReport(verdict="escalate", rationale="uncertain"),
        task_spec=TaskSpec(title="t", description="d"),
    )
    deps = FlowDeps(state=state, provenance_db_path=":memory:")
    record = start_experiment(state, deps)
    updated = complete_experiment(record, state, deps)
    assert updated.status == "failed"
    assert updated.validation_report_id == state.validation_report.report_id
