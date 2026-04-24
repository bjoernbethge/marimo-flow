"""Tests for ProblemNode — post-pivot, uses compose_problem."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.state import FlowState


async def test_problem_node_runs_and_hands_back_to_route():
    """With TestModel issuing no tool calls, ProblemNode should still
    complete its run loop and return to the router — no dependence on a
    specific tool being available.
    """
    test_model = TestModel(call_tools=[])
    state = FlowState(user_intent="define a 1D Poisson")
    deps = FlowDeps(provenance_db_path=":memory:")
    node = ProblemNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "problem"
