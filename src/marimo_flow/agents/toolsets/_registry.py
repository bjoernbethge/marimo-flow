"""Shared helpers for the workflow toolsets.

Every toolset that produces a live object (``compose_problem`` for
problems, ``build_model`` / ``build_solver`` / ``train`` for the other
stages) funnels through ``register_artifact`` which does four things:

  1. ensure an MLflow run is active (open one if not)
  2. log a JSON record of the spec as an MLflow artifact for traceability
  3. stash the real live instance in `deps.registry` under a URI key
  4. mirror the artifact + any derived lineage edges into the DuckDB
     provenance store (best-effort; silent if provenance is unavailable)
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypeVar

import mlflow
from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import ArtifactKind, ArtifactRef
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

    Also mirrors an ArtifactRef into the DuckDB provenance store and
    emits lineage edges from upstream artifacts (problem→model,
    problem→solver, model→solver, solver→training). Provenance writes
    are best-effort — a closed store or missing DuckDB never breaks the
    MLflow-only path.

    Returns the MLflow artifact URI (used as both the registry key and
    the State.*_artifact_uri field).
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
    _mirror_to_provenance(
        deps=deps,
        state=state,
        uri=uri,
        artifact_path=artifact_path,
        filename=filename,
    )
    return uri


_ARTIFACT_KIND_MAP: dict[str, ArtifactKind] = {
    "problem": "problem",
    "model": "model",
    "solver": "solver",
    "training": "training",
}

# Upstream artifacts that feed into each stage — used to emit lineage
# edges from the previously-registered URIs into the new one.
_LINEAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "model": ("problem_artifact_uri",),
    "solver": ("problem_artifact_uri", "model_artifact_uri"),
    "training": ("solver_artifact_uri",),
}


def _mirror_to_provenance(
    *,
    deps: FlowDeps,
    state: FlowState,
    uri: str,
    artifact_path: str,
    filename: str,
) -> None:
    """Write an ArtifactRef + lineage edges into the DuckDB store.

    Silent no-op when the provenance store can't be opened (e.g. another
    process holds the DB file, or the feature was disabled).
    """
    kind = _ARTIFACT_KIND_MAP.get(artifact_path, "other")
    task_id = state.task_spec.task_id if state.task_spec else None
    ref = ArtifactRef(kind=kind, uri=uri, label=filename)
    with contextlib.suppress(Exception):
        store = deps.provenance()
        store.record_artifact(ref, task_id=task_id)
        for attr in _LINEAGE_SOURCES.get(artifact_path, ()):
            from_uri = getattr(state, attr, None)
            if from_uri:
                store.record_lineage_edge(
                    from_uri=from_uri, to_uri=uri, relation=f"produced_for_{kind}"
                )
