"""Tests for ProblemNode — uses a fake registrar to avoid MLflow + PINA in unit tests."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.state import FlowState


async def test_problem_node_records_artifact_uri_when_tool_called(monkeypatch):
    captured = {}

    def fake_register(spec, deps, state):
        uri = "mlflow:/fake/problem"
        deps.registry[uri] = ("fake-problem", spec)
        state.problem_artifact_uri = uri
        captured["called"] = True
        return uri

    monkeypatch.setattr(
        "marimo_flow.agents.nodes.problem._define_problem", fake_register
    )

    test_model = TestModel(call_tools=["define_problem"])
    state = FlowState(user_intent="define a 1D Poisson")
    deps = FlowDeps()
    node = ProblemNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)

    assert captured.get("called") is True
    assert state.problem_artifact_uri == "mlflow:/fake/problem"
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "problem"
