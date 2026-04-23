"""Shared helpers for the workflow toolsets.

Every builder tool (build_problem / build_model / build_solver) does the
same three things:
  1. ensure an MLflow run is active (open one if not)
  2. log a JSON record of the init kwargs as an artifact for traceability
  3. stash the real live instance in `deps.registry` under a URI key

`_register_artifact(...)` centralises that so toolsets stay thin.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


def require_state(deps: FlowDeps) -> FlowState:
    """Return `deps.state` or raise with a helpful message."""
    if deps.state is None:
        raise RuntimeError(
            "FlowDeps.state is not set — Node must assign ctx.state "
            "to deps.state before running the agent."
        )
    return deps.state


def register_artifact(
    *,
    deps: FlowDeps,
    state: FlowState,
    artifact_path: str,
    filename: str,
    record: dict[str, Any],
    instance: Any,
) -> str:
    """Log `record` as JSON artifact, register `instance` in deps.registry.

    Returns the MLflow artifact URI (used as both the registry key and the
    State.*_artifact_uri field).
    """
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / filename
        p.write_text(json.dumps(record, indent=2, default=str))
        client.log_artifact(state.mlflow_run_id, str(p), artifact_path=artifact_path)
    uri = f"runs:/{state.mlflow_run_id}/{artifact_path}/{filename}"
    deps.registry[uri] = instance
    return uri
