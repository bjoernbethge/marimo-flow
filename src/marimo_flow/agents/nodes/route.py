"""RouteNode — classifier that dispatches to the right specialist sub-node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.mlflow_node import MLflowNode
    from marimo_flow.agents.nodes.model import ModelNode
    from marimo_flow.agents.nodes.notebook import NotebookNode
    from marimo_flow.agents.nodes.problem import ProblemNode
    from marimo_flow.agents.nodes.solver import SolverNode

ROUTE_INSTRUCTIONS = """\
You are a router for a PINA (Physics-Informed NN) workflow team.
Given the user intent and the current FlowState, choose exactly one next node:

- notebook: edit/inspect/run marimo notebook cells
- problem:  define a PINA Problem (PDE, BCs, domain, conditions)
- model:    pick / configure a neural architecture (FNN, FNO, KAN, DeepONet)
- solver:   wire a Solver (PINN, SAPINN, GAROM) and a Trainer
- mlflow:   log/inspect/register experiments and runs
- end:      the user's intent is satisfied; return a brief summary as rationale

Pick `end` only when the FlowState clearly satisfies the user intent
(e.g. solver was trained AND mlflow run id is set when the user asked for training).
"""


class RouteDecision(BaseModel):
    next_node: Literal["notebook", "problem", "model", "solver", "mlflow", "end"]
    rationale: str


@dataclass
class RouteNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(
        self, ctx: GraphRunContext[FlowState, FlowDeps]
    ) -> NotebookNode | ProblemNode | ModelNode | SolverNode | MLflowNode | End[str]:
        from marimo_flow.agents.nodes.mlflow_node import MLflowNode
        from marimo_flow.agents.nodes.model import ModelNode
        from marimo_flow.agents.nodes.notebook import NotebookNode
        from marimo_flow.agents.nodes.problem import ProblemNode
        from marimo_flow.agents.nodes.solver import SolverNode

        model = self.model_override or get_model(
            "route", override=ctx.deps.models["route"]
        )
        agent = Agent(model, output_type=RouteDecision, instructions=ROUTE_INSTRUCTIONS)
        prompt = (
            f"User intent: {ctx.state.user_intent!r}\n"
            f"State: last_node={ctx.state.last_node}, "
            f"problem={'set' if ctx.state.problem_artifact_uri else 'unset'}, "
            f"model={'set' if ctx.state.model_artifact_uri else 'unset'}, "
            f"solver={'set' if ctx.state.solver_artifact_uri else 'unset'}, "
            f"mlflow_run_id={ctx.state.mlflow_run_id}"
        )
        result = await agent.run(prompt)
        decision = result.output
        ctx.state.last_node = "route"

        dispatch: dict[str, type[BaseNode]] = {
            "notebook": NotebookNode,
            "problem": ProblemNode,
            "model": ModelNode,
            "solver": SolverNode,
            "mlflow": MLflowNode,
        }
        if decision.next_node == "end":
            return End(decision.rationale)
        return dispatch[decision.next_node]()
