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
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import mlflow
from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState

T = TypeVar("T")


def require_state(deps: FlowDeps) -> FlowState:
    """Return `deps.state` or raise with a helpful message."""
    if deps.state is None:
        raise RuntimeError(
            "FlowDeps.state is not set — Node must assign ctx.state "
            "to deps.state before running the agent."
        )
    return deps.state


def retry_on_value_error(
    call: Callable[[], T],
    *,
    hint: str | None = None,
    available: Iterable[str] | None = None,
) -> T:
    """Convert ValueError into ModelRetry so pydantic-ai retries with feedback.

    Nearly every manager entry point (``ProblemManager.create``, ``ModelManager``,
    ``SolverManager``) raises ``ValueError`` with a readable message for bad
    ``kind`` or unsupported combinations. Those are exactly the cases the LLM
    can fix on its own if we tell it the valid options, so we raise
    ``ModelRetry`` with the same message rather than crashing the graph.
    """
    try:
        return call()
    except ValueError as e:
        parts = [str(e)]
        if hint:
            parts.append(hint)
        if available is not None:
            parts.append(f"Available: {', '.join(sorted(available))}.")
        raise ModelRetry(" ".join(parts)) from e


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
