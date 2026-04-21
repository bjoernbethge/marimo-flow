"""Tests for ModelNode."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.state import FlowState


async def test_model_node_records_uri(monkeypatch):
    def fake_define(spec, deps, state):
        uri = "mlflow:/fake/model/spec"
        deps.registry[uri] = spec
        state.model_artifact_uri = uri
        return uri

    monkeypatch.setattr("marimo_flow.agents.nodes.model._define_model", fake_define)
    test_model = TestModel(call_tools=["define_model"])
    state = FlowState(user_intent="pick architecture", problem_artifact_uri="mlflow:/p")
    deps = FlowDeps()
    node = ModelNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert state.model_artifact_uri == "mlflow:/fake/model/spec"
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "model"
