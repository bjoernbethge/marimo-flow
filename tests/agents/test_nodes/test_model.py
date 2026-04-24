"""Tests for ModelNode — monkeypatch core builder + MLflow helper."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.state import FlowState


async def test_model_node_builds_and_registers_model(monkeypatch):
    fake_problem = object()
    fake_model = object()

    def fake_create(kind, *, problem, **kwargs):  # noqa: ARG001
        assert problem is fake_problem
        return fake_model

    def fake_register_artifact(*, deps, state, **kwargs):  # noqa: ARG001
        uri = "runs:/fake/model/spec.json"
        deps.registry[uri] = kwargs["instance"]
        return uri

    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.model.ModelManager.create", fake_create
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.model.register_artifact", fake_register_artifact
    )

    test_model = TestModel(call_tools=["build_model"])
    deps = FlowDeps(provenance_db_path=":memory:")
    problem_uri = "runs:/fake/problem/spec.json"
    deps.registry[problem_uri] = fake_problem
    state = FlowState(
        user_intent="pick architecture",
        problem_artifact_uri=problem_uri,
    )
    node = ModelNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)

    assert state.model_artifact_uri == "runs:/fake/model/spec.json"
    assert deps.registry[state.model_artifact_uri] is fake_model
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "model"
