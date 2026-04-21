"""Tests for NotebookNode."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.state import FlowState


def _stub_marimo_mcp(monkeypatch):
    """Replace the real marimo MCP builder with an empty in-process toolset
    so tests do not need a running marimo server."""
    monkeypatch.setattr(
        "marimo_flow.agents.nodes.notebook.build_marimo_mcp",
        lambda url=None: FunctionToolset(),
    )


async def test_notebook_returns_to_route(monkeypatch):
    _stub_marimo_mcp(monkeypatch)
    test_model = TestModel(
        custom_output_text="Listed cells; nothing to change.", call_tools=[]
    )
    state = FlowState(user_intent="list cells")
    deps = FlowDeps()
    node = NotebookNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "notebook"
    assert "notebook" in state.history


async def test_notebook_appends_to_history(monkeypatch):
    _stub_marimo_mcp(monkeypatch)
    test_model = TestModel(custom_output_text="ok", call_tools=[])
    state = FlowState(user_intent="list cells")
    deps = FlowDeps()
    node = NotebookNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    await node.run(ctx)
    assert len(state.history["notebook"]) >= 1
