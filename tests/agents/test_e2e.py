"""End-to-end smoke test — real MLflow file:// store, TestModel for all LLMs.

Drives the full sequence problem -> model -> solver -> training -> end with
the Managers + register_artifact stubbed so no torch work is needed, but
real MLflow artifacts are logged to verify persistence.
"""

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


def _make_fake_register_artifact(kind: str):
    """Write a small JSON artifact into the active MLflow run and return its URI."""

    def _fake(*, deps, state, artifact_path, filename, record, instance):
        run_id = state.mlflow_run_id
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / filename
            p.write_text(json.dumps({"stub": True, "kind": kind, **record}, indent=2))
            mlflow.MlflowClient().log_artifact(
                run_id, str(p), artifact_path=artifact_path
            )
        uri = f"runs:/{run_id}/{artifact_path}/{filename}"
        deps.registry[uri] = instance
        return uri

    return _fake


async def test_full_workflow_reaches_end(tmp_mlflow, monkeypatch):
    """problem -> model -> solver -> training -> end; all artifacts logged."""
    decisions = iter(
        [
            {"next_node": "problem", "rationale": "need problem first"},
            {"next_node": "model", "rationale": "need architecture"},
            {"next_node": "solver", "rationale": "wire solver"},
            {"next_node": "training", "rationale": "fit it"},
            {"next_node": "end", "rationale": "all done; solver trained"},
        ]
    )

    fake_problem = object()
    fake_model = object()
    fake_solver = object()
    fake_trainer = type("FT", (), {"callback_metrics": {"train_loss": 0.1}})()

    def fake_model_for(self, role):
        if role == "route":
            return TestModel(custom_output_args=next(decisions))
        if role in ("problem", "model", "solver"):
            return TestModel(call_tools=[f"build_{role}"])
        if role == "training":
            return TestModel(call_tools=["train"])
        return TestModel(call_tools=[])

    monkeypatch.setattr(
        "marimo_flow.agents.deps.FlowDeps.model_for", fake_model_for
    )

    # Stub Manager.create + register_artifact for each toolset module
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.problem.ProblemManager.create",
        lambda kind, **_kw: fake_problem,
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.model.ModelManager.create",
        lambda kind, *, problem, **_kw: fake_model,
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.solver.SolverManager.create",
        lambda kind, *, problem, model, **_kw: fake_solver,
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.training.train_solver",
        lambda solver, **_kw: fake_trainer,  # noqa: ARG005
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.problem.register_artifact",
        _make_fake_register_artifact("problem"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.model.register_artifact",
        _make_fake_register_artifact("model"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.solver.register_artifact",
        _make_fake_register_artifact("solver"),
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.training.register_artifact",
        _make_fake_register_artifact("training"),
    )

    graph = build_graph()
    state = FlowState(user_intent="solve burgers 1d", mlflow_run_id=tmp_mlflow)
    deps = FlowDeps()
    persistence = MLflowStatePersistence(run_id=tmp_mlflow)
    persistence.set_graph_types(graph)

    result = await graph.run(
        start_node(), state=state, deps=deps, persistence=persistence
    )
    assert "done" in result.output.lower() or "trained" in result.output.lower()

    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(tmp_mlflow)}
    assert "problem" in artifacts
    assert "model" in artifacts
    assert "solver" in artifacts
    assert "training" in artifacts
    assert "agent_state" in artifacts
