"""Tests for ProblemNode — monkeypatch core builder + MLflow helper."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.state import FlowState


async def test_problem_node_builds_and_registers_problem(monkeypatch):
    fake_problem = object()

    def fake_create(kind, **kwargs):  # noqa: ARG001
        return fake_problem

    def fake_register_artifact(*, deps, state, **kwargs):  # noqa: ARG001
        uri = "runs:/fake/problem/spec.json"
        deps.registry[uri] = kwargs["instance"]
        return uri

    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.problem.ProblemManager.create", fake_create
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.problem.register_artifact", fake_register_artifact
    )

    test_model = TestModel(call_tools=["build_problem"])
    state = FlowState(user_intent="define a 1D Poisson")
    deps = FlowDeps()
    node = ProblemNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)

    assert state.problem_artifact_uri == "runs:/fake/problem/spec.json"
    assert deps.registry[state.problem_artifact_uri] is fake_problem
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "problem"
