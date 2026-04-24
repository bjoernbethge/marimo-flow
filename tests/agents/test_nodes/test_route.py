"""Tests for RouteNode classifier."""

from __future__ import annotations

import typing

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_graph import End, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.route import RouteDecision, RouteNode
from marimo_flow.agents.state import FlowState


@pytest.mark.parametrize(
    "decision_value, expected_cls_name",
    [
        ("notebook", "NotebookNode"),
        ("problem", "ProblemNode"),
        ("model", "ModelNode"),
        ("solver", "SolverNode"),
        ("training", "TrainingNode"),
        ("mlflow", "MLflowNode"),
    ],
)
async def test_route_dispatches_to_each_node(decision_value, expected_cls_name):
    test_model = TestModel(
        custom_output_args={"next_node": decision_value, "rationale": "x"}
    )
    state = FlowState(user_intent="something")
    deps = FlowDeps()
    node = RouteNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == expected_cls_name


async def test_route_end_returns_end_with_summary():
    test_model = TestModel(
        custom_output_args={"next_node": "end", "rationale": "all done"}
    )
    state = FlowState(user_intent="done already")
    deps = FlowDeps()
    node = RouteNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    result = await node.run(ctx)
    assert isinstance(result, End)
    assert "done" in result.data.lower()


def test_route_decision_schema_lists_all_options():
    options = set(typing.get_args(RouteDecision.model_fields["next_node"].annotation))
    assert options == {
        "notebook",
        "problem",
        "model",
        "solver",
        "training",
        "mlflow",
        "end",
    }


async def test_route_increments_counter():
    test_model = TestModel(
        custom_output_args={"next_node": "problem", "rationale": "x"}
    )
    state = FlowState(user_intent="go")
    assert state.route_count == 0
    node = RouteNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=FlowDeps())
    await node.run(ctx)
    assert state.route_count == 1


async def test_route_circuit_breaker_trips_when_exceeded():
    test_model = TestModel(
        custom_output_args={"next_node": "problem", "rationale": "x"}
    )
    state = FlowState(user_intent="go", max_route_steps=3, route_count=3)
    node = RouteNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=FlowDeps())
    result = await node.run(ctx)
    assert isinstance(result, End)
    assert "Circuit breaker" in result.data
