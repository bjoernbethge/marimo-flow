"""ExperimentRecord lifecycle — called by the lead workflow around graph.run().

One ``ExperimentRecord`` is created at the start of a workflow with
``status="running"`` and updated when the graph ends. Since the task
hasn't been triaged yet when the record is opened, ``task_id`` is
allowed to be ``None`` initially and gets filled in at completion.

All writes go through the DuckDB provenance store via
``deps.provenance()``; errors are swallowed so the MLflow-only happy
path never breaks.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import ExperimentRecord
from marimo_flow.agents.services.orchestrator import default_experiment_status
from marimo_flow.agents.state import FlowState


def start_experiment(state: FlowState, deps: FlowDeps) -> ExperimentRecord:
    """Open a new ExperimentRecord with ``status='running'``.

    Mirrors it into DuckDB and stores the ``experiment_id`` on the
    FlowState so sub-nodes (and later tools) can reference it.
    """
    record = ExperimentRecord(
        task_id=state.task_spec.task_id if state.task_spec else None,
        run_id=state.mlflow_run_id,
        status="running",
    )
    state.experiment_id = record.experiment_id
    with contextlib.suppress(Exception):
        deps.provenance().record_experiment(record)
    return record


def complete_experiment(
    record: ExperimentRecord,
    state: FlowState,
    deps: FlowDeps,
    *,
    status: str | None = None,
) -> ExperimentRecord:
    """Replace the running row with a finished one.

    ``status`` defaults to the inferred end-state from
    :func:`default_experiment_status`. Artifact URIs, validation report
    id and task_id are pulled from the FlowState at call time so the
    final row is a snapshot of the post-run world.
    """
    resolved_status = status or default_experiment_status(state)
    updated = ExperimentRecord(
        experiment_id=record.experiment_id,
        task_id=record.task_id
        or (state.task_spec.task_id if state.task_spec else None),
        run_id=state.mlflow_run_id or record.run_id,
        problem_artifact_uri=state.problem_artifact_uri,
        model_artifact_uri=state.model_artifact_uri,
        solver_artifact_uri=state.solver_artifact_uri,
        training_artifact_uri=state.training_artifact_uri,
        validation_report_id=(
            state.validation_report.report_id if state.validation_report else None
        ),
        status=resolved_status,  # type: ignore[arg-type]
        created_at=record.created_at,
        finished_at=datetime.now(UTC),
    )
    with contextlib.suppress(Exception):
        deps.provenance().record_experiment(updated)
    return updated
