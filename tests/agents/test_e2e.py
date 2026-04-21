"""End-to-end smoke test — real MLflow file:// store, TestModel for all LLMs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
import pytest
from pydantic_ai.models.test import TestModel

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState


@pytest.fixture
def tmp_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("agents-e2e")
    with mlflow.start_run() as run:
        yield run.info.run_id


def _make_fake_define(kind: str):
    """Build a `_define_<kind>` replacement that logs a real MLflow artifact.

    TestModel auto-arg synthesis for tools whose signature is
    `define_<kind>(spec: dict[str, Any])` produces a single trivial dict
    (`{"additionalProperty": "a"}`) — usable here because the stub ignores
    the spec content. Crucially, we route the artifact through an explicit
    `MlflowClient().log_artifact(run_id, ...)` call: inside `await graph.run`
    there is no module-level active run, so `mlflow.log_artifact` (used by
    the production `_define_*` helpers) would silently no-op or auto-create
    a fresh run instead of writing to the test's `tmp_mlflow` run.
    """

    def _fake(spec, deps, state):
        run_id = state.mlflow_run_id
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"{kind}_spec.json"
            p.write_text(json.dumps({"stub": True, "kind": kind}, indent=2))
            mlflow.MlflowClient().log_artifact(run_id, str(p), artifact_path=kind)
        uri = f"runs:/{run_id}/{kind}/{kind}_spec.json"
        deps.registry[uri] = {"stub": True, "kind": kind}
        setattr(state, f"{kind}_artifact_uri", uri)
        return uri

    return _fake


async def test_full_workflow_reaches_end(tmp_mlflow, monkeypatch):
    # Drive each node deterministically:
    # route -> problem -> route -> model -> route -> solver -> route -> end
    decisions = iter(
        [
            {"next_node": "problem", "rationale": "need problem first"},
            {"next_node": "model", "rationale": "need architecture"},
            {"next_node": "solver", "rationale": "ready to wire solver"},
            {"next_node": "end", "rationale": "all set; ready to train"},
        ]
    )

    def fake_get_model(role, **_kw):
        if role == "route":
            return TestModel(custom_output_args=next(decisions))
        if role in ("problem", "model", "solver"):
            return TestModel(call_tools=[f"define_{role}"])
        return TestModel(call_tools=[])

    monkeypatch.setattr("marimo_flow.agents.nodes.route.get_model", fake_get_model)
    monkeypatch.setattr("marimo_flow.agents.nodes.problem.get_model", fake_get_model)
    monkeypatch.setattr("marimo_flow.agents.nodes.model.get_model", fake_get_model)
    monkeypatch.setattr("marimo_flow.agents.nodes.solver.get_model", fake_get_model)

    monkeypatch.setattr(
        "marimo_flow.agents.nodes.problem._define_problem",
        _make_fake_define("problem"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.nodes.model._define_model", _make_fake_define("model")
    )
    monkeypatch.setattr(
        "marimo_flow.agents.nodes.solver._define_solver", _make_fake_define("solver")
    )

    graph = build_graph()
    state = FlowState(user_intent="solve poisson 1d", mlflow_run_id=tmp_mlflow)
    deps = FlowDeps()
    persistence = MLflowStatePersistence(run_id=tmp_mlflow)
    persistence.set_graph_types(graph)

    result = await graph.run(
        start_node(), state=state, deps=deps, persistence=persistence
    )
    assert "set" in result.output.lower() or "ready" in result.output.lower()

    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(tmp_mlflow)}
    assert "problem" in artifacts
    assert "model" in artifacts
    assert "solver" in artifacts
    assert "agent_state" in artifacts
