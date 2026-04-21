"""Tests for MLflowNode."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.state import FlowState


def _stub_mlflow_mcp(monkeypatch):
    """Replace the real mlflow MCP builder with an empty in-process toolset
    so tests do not need to spawn `mlflow mcp run`."""
    monkeypatch.setattr(
        "marimo_flow.agents.nodes.mlflow_node.build_mlflow_mcp",
        lambda tracking_uri=None: FunctionToolset(),
    )


async def test_mlflow_node_returns_to_route(monkeypatch):
    _stub_mlflow_mcp(monkeypatch)
    test_model = TestModel(custom_output_text="Listed experiments.", call_tools=[])
    state = FlowState(user_intent="show me last run")
    deps = FlowDeps()
    node = MLflowNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "mlflow"
    assert "mlflow" in state.history
