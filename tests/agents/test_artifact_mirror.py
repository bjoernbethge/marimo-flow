"""Tests for the DuckDB mirror in _registry.register_artifact.

``register_artifact`` is the shared helper all build-* toolsets go
through. It mirrors an ``ArtifactRef`` into the provenance store and
emits lineage edges from upstream URIs when available.
"""

from __future__ import annotations

import pytest

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets._registry import register_artifact


@pytest.fixture
def deps_and_state(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("artifact-mirror-test")
    with mlflow.start_run() as run:
        state = FlowState(
            mlflow_run_id=run.info.run_id,
            task_spec=TaskSpec(title="t", description="d"),
        )
        deps = FlowDeps(state=state, provenance_db_path=":memory:")
        yield deps, state


def _query_artifacts(deps):
    return deps.provenance().query(
        "SELECT uri, kind, task_id, label FROM artifacts ORDER BY created_at"
    )


def _query_lineage(deps):
    return deps.provenance().query(
        "SELECT from_uri, to_uri, relation FROM lineage_edges"
    )


def test_register_artifact_records_problem(deps_and_state):
    deps, state = deps_and_state
    uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="problem",
        filename="problem_spec.json",
        record={"kind": "poisson"},
        instance=object(),
    )
    rows = _query_artifacts(deps)
    assert len(rows) == 1
    assert rows[0]["uri"] == uri
    assert rows[0]["kind"] == "problem"
    assert rows[0]["task_id"] == state.task_spec.task_id
    # Problem is the upstream root → no lineage edges yet.
    assert _query_lineage(deps) == []


def test_register_artifact_emits_lineage_for_model(deps_and_state):
    deps, state = deps_and_state
    state.problem_artifact_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="problem",
        filename="problem_spec.json",
        record={"kind": "poisson"},
        instance=object(),
    )
    model_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="model",
        filename="model_spec.json",
        record={"kind": "feedforward"},
        instance=object(),
    )
    edges = _query_lineage(deps)
    assert len(edges) == 1
    assert edges[0]["from_uri"] == state.problem_artifact_uri
    assert edges[0]["to_uri"] == model_uri
    assert edges[0]["relation"] == "produced_for_model"


def test_register_artifact_emits_two_edges_for_solver(deps_and_state):
    deps, state = deps_and_state
    state.problem_artifact_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="problem",
        filename="p.json",
        record={},
        instance=object(),
    )
    state.model_artifact_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="model",
        filename="m.json",
        record={},
        instance=object(),
    )
    solver_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="solver",
        filename="s.json",
        record={},
        instance=object(),
    )
    edges = _query_lineage(deps)
    # 1 edge for model, 2 edges for solver → 3 total.
    assert len(edges) == 3
    solver_edges = [e for e in edges if e["to_uri"] == solver_uri]
    assert len(solver_edges) == 2
    sources = {e["from_uri"] for e in solver_edges}
    assert sources == {state.problem_artifact_uri, state.model_artifact_uri}


def test_register_artifact_records_training_lineage(deps_and_state):
    deps, state = deps_and_state
    state.solver_artifact_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="solver",
        filename="s.json",
        record={},
        instance=object(),
    )
    training_uri = register_artifact(
        deps=deps,
        state=state,
        artifact_path="training",
        filename="t.json",
        record={"final_loss": 0.01},
        instance=object(),
    )
    edges = [e for e in _query_lineage(deps) if e["to_uri"] == training_uri]
    assert len(edges) == 1
    assert edges[0]["from_uri"] == state.solver_artifact_uri
    assert edges[0]["relation"] == "produced_for_training"
