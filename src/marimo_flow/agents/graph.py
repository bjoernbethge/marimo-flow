"""Graph assembly — seven nodes, RouteNode as start."""

from __future__ import annotations

from pydantic_graph import Graph

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes import mlflow_node as _mlflow_node_mod
from marimo_flow.agents.nodes import model as _model_mod
from marimo_flow.agents.nodes import notebook as _notebook_mod
from marimo_flow.agents.nodes import problem as _problem_mod
from marimo_flow.agents.nodes import route as _route_mod
from marimo_flow.agents.nodes import solver as _solver_mod
from marimo_flow.agents.nodes import training as _training_mod
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.nodes.training import TrainingNode
from marimo_flow.agents.state import FlowState


def build_graph() -> Graph[FlowState, FlowDeps, str]:
    # Inject RouteNode into each specialist module's globals so that
    # `pydantic_graph.BaseNode.get_node_def` -> `get_type_hints(cls.run)`
    # can resolve the `-> RouteNode` forward refs. Likewise inject the
    # specialists into RouteNode's module so its return-type union
    # resolves. We patch the module namespace at runtime to keep the
    # source-level `if TYPE_CHECKING:` imports cycle-free.
    for mod in (
        _notebook_mod,
        _problem_mod,
        _model_mod,
        _solver_mod,
        _training_mod,
        _mlflow_node_mod,
    ):
        mod.__dict__.setdefault("RouteNode", RouteNode)
    _route_mod.__dict__.update(
        NotebookNode=NotebookNode,
        ProblemNode=ProblemNode,
        ModelNode=ModelNode,
        SolverNode=SolverNode,
        TrainingNode=TrainingNode,
        MLflowNode=MLflowNode,
    )
    return Graph(
        nodes=(
            RouteNode,
            NotebookNode,
            ProblemNode,
            ModelNode,
            SolverNode,
            TrainingNode,
            MLflowNode,
        ),
        state_type=FlowState,
    )


def start_node() -> RouteNode:
    return RouteNode()
