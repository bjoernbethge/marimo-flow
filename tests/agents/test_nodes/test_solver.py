"""Tests for SolverNode — monkeypatch core builder + MLflow helper."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.state import FlowState


async def test_solver_node_builds_and_registers_solver(monkeypatch):
    fake_problem = object()
    fake_model = object()
    fake_solver = object()

    def fake_create(kind, *, problem, model, **kwargs):  # noqa: ARG001
        assert problem is fake_problem
        assert model is fake_model
        return fake_solver

    def fake_register_artifact(*, deps, state, **kwargs):  # noqa: ARG001
        uri = "runs:/fake/solver/spec.json"
        deps.registry[uri] = kwargs["instance"]
        return uri

    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.solver.SolverManager.create", fake_create
    )
    monkeypatch.setattr(
        "marimo_flow.agents.toolsets.solver.register_artifact", fake_register_artifact
    )

    test_model = TestModel(call_tools=["build_solver"])
    deps = FlowDeps(provenance_db_path=":memory:")
    deps.registry["runs:/fake/problem/spec.json"] = fake_problem
    deps.registry["runs:/fake/model/spec.json"] = fake_model
    state = FlowState(
        user_intent="train it",
        problem_artifact_uri="runs:/fake/problem/spec.json",
        model_artifact_uri="runs:/fake/model/spec.json",
    )
    node = SolverNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)

    assert state.solver_artifact_uri == "runs:/fake/solver/spec.json"
    assert deps.registry[state.solver_artifact_uri] is fake_solver
    assert state.last_node == "solver"
    assert next_node.__class__.__name__ == "RouteNode"
