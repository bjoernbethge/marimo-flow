"""Tests for SolverNode."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.state import FlowState


async def test_solver_records_uri(monkeypatch):
    def fake_define(spec, deps, state):
        uri = "mlflow:/fake/solver"
        state.solver_artifact_uri = uri
        deps.registry[uri] = spec
        return uri

    monkeypatch.setattr("marimo_flow.agents.nodes.solver._define_solver", fake_define)
    test_model = TestModel(call_tools=["define_solver"])
    state = FlowState(
        user_intent="train it",
        problem_artifact_uri="mlflow:/p",
        model_artifact_uri="mlflow:/m",
    )
    deps = FlowDeps()
    node = SolverNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert state.solver_artifact_uri == "mlflow:/fake/solver"
    assert state.last_node == "solver"
    assert next_node.__class__.__name__ == "RouteNode"
