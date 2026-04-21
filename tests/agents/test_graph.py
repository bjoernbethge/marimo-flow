"""Tests for the assembled graph."""

from __future__ import annotations

from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode


def test_graph_contains_all_six_nodes():
    graph = build_graph()
    expected = {cls.__name__ for cls in (RouteNode, NotebookNode, ProblemNode, ModelNode, SolverNode, MLflowNode)}
    assert expected.issubset(set(graph.node_defs.keys()))


def test_start_node_is_route():
    assert isinstance(start_node(), RouteNode)


def test_graph_renders_mermaid():
    graph = build_graph()
    code = graph.mermaid_code(start_node=RouteNode)
    assert "RouteNode" in code
    assert "stateDiagram" in code or "graph" in code.lower()
