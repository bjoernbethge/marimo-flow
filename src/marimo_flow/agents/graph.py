"""Graph assembly — six nodes, RouteNode as start."""

from __future__ import annotations

from pydantic_graph import Graph

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.state import FlowState


def build_graph() -> Graph[FlowState, FlowDeps, str]:
    return Graph(
        nodes=(RouteNode, NotebookNode, ProblemNode, ModelNode, SolverNode, MLflowNode),
        state_type=FlowState,
    )


def start_node() -> RouteNode:
    return RouteNode()
