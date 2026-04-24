"""RouteNode — classifier that dispatches to the right specialist sub-node."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import AgentRole, HandoffRecord
from marimo_flow.agents.services.orchestrator import check_escalation
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.mlflow_node import MLflowNode
    from marimo_flow.agents.nodes.model import ModelNode
    from marimo_flow.agents.nodes.notebook import NotebookNode
    from marimo_flow.agents.nodes.problem import ProblemNode
    from marimo_flow.agents.nodes.solver import SolverNode
    from marimo_flow.agents.nodes.training import TrainingNode
    from marimo_flow.agents.nodes.validation import ValidationNode

ROUTE_INSTRUCTIONS = """\
You are a router for a PINA (Physics-Informed NN) workflow team.
Given the user intent and the current FlowState, choose exactly one next node:

- notebook:   edit/inspect/run marimo notebook cells
- problem:    define a PINA Problem (PDE, BCs, domain, conditions)
- model:      pick / configure a neural architecture (FeedForward, FNO, DeepONet, ...)
- solver:     wire a Solver (PINN, SAPINN, CausalPINN, ...)
- training:   fit the solver via pina.Trainer (requires problem+model+solver registered)
- validation: grade a completed training run against task_spec.constraints
- mlflow:     log/inspect/register experiments and runs
- end:        the user's intent is satisfied; return a brief summary as rationale

Typical sequence for a full solve:
  problem -> model -> solver -> training -> validation -> (optional mlflow) -> end

Rules:
- After training_run_id is set and validation_report is unset, route to `validation`.
- After validation_report.verdict == "accept", route to `end` with a summary.
- If validation_report.verdict is "retry", you may re-enter training with
  tuned kwargs; the circuit breaker will stop runaways.
- Pick `end` only when the user's intent is clearly satisfied.
"""


class RouteDecision(BaseModel):
    next_node: Literal[
        "notebook",
        "problem",
        "model",
        "solver",
        "training",
        "validation",
        "mlflow",
        "end",
    ]
    rationale: str


@dataclass
class RouteNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(
        self, ctx: GraphRunContext[FlowState, FlowDeps]
    ) -> (
        NotebookNode
        | ProblemNode
        | ModelNode
        | SolverNode
        | TrainingNode
        | ValidationNode
        | MLflowNode
        | End[str]
    ):
        from marimo_flow.agents.nodes.mlflow_node import MLflowNode
        from marimo_flow.agents.nodes.model import ModelNode
        from marimo_flow.agents.nodes.notebook import NotebookNode
        from marimo_flow.agents.nodes.problem import ProblemNode
        from marimo_flow.agents.nodes.solver import SolverNode
        from marimo_flow.agents.nodes.training import TrainingNode
        from marimo_flow.agents.nodes.validation import ValidationNode

        ctx.state.route_count += 1
        if ctx.state.route_count > ctx.state.max_route_steps:
            return End(
                f"Circuit breaker: exceeded max_route_steps="
                f"{ctx.state.max_route_steps}. Last node={ctx.state.last_node}. "
                "The router looped without reaching 'end' — inspect the "
                "FlowState history for why."
            )

        escalation = check_escalation(ctx.state)
        if escalation is not None:
            return End(escalation)

        model = self.model_override or ctx.deps.model_for("route")
        agent = Agent(model, output_type=RouteDecision, instructions=ROUTE_INSTRUCTIONS)
        validation_state = (
            ctx.state.validation_report.verdict
            if ctx.state.validation_report is not None
            else "unset"
        )
        prompt = (
            f"User intent: {ctx.state.user_intent!r}\n"
            f"Task: "
            f"{ctx.state.task_spec.title if ctx.state.task_spec else 'unset'}\n"
            f"State: last_node={ctx.state.last_node}, "
            f"problem={'set' if ctx.state.problem_artifact_uri else 'unset'}, "
            f"model={'set' if ctx.state.model_artifact_uri else 'unset'}, "
            f"solver={'set' if ctx.state.solver_artifact_uri else 'unset'}, "
            f"training_run_id={ctx.state.training_run_id}, "
            f"validation={validation_state}, "
            f"mlflow_run_id={ctx.state.mlflow_run_id}, "
            f"step={ctx.state.route_count}/{ctx.state.max_route_steps}"
        )
        result = await agent.run(prompt)
        decision = result.output
        ctx.state.last_node = "route"

        dispatch: dict[str, type[BaseNode]] = {
            "notebook": NotebookNode,
            "problem": ProblemNode,
            "model": ModelNode,
            "solver": SolverNode,
            "training": TrainingNode,
            "validation": ValidationNode,
            "mlflow": MLflowNode,
        }
        if decision.next_node == "end":
            return End(decision.rationale)
        _record_handoff(ctx.state, ctx.deps, decision.next_node, decision.rationale)
        return dispatch[decision.next_node]()


def _record_handoff(
    state: FlowState, deps: FlowDeps, to_agent: str, reason: str
) -> None:
    """Append a HandoffRecord to state + mirror it to the DuckDB store.

    ``from_agent`` is always "route" — this node is the one deciding the
    handoff. Artifact URIs already present in the state are included so
    downstream inspection can rebuild the graph of handed-off work.
    """
    if to_agent not in _VALID_AGENT_ROLES:
        # Should never happen — dispatch dict keys are a subset of
        # RouteDecision's Literal, which is a subset of AgentRole.
        return
    handoff = HandoffRecord(
        from_agent="route",
        to_agent=to_agent,  # type: ignore[arg-type]
        reason=reason,
        task_id=state.task_spec.task_id if state.task_spec else None,
        run_id=state.mlflow_run_id,
        artifact_uris=[
            uri
            for uri in (
                state.problem_artifact_uri,
                state.model_artifact_uri,
                state.solver_artifact_uri,
                state.training_artifact_uri,
            )
            if uri
        ],
    )
    state.handoffs.append(handoff)
    with contextlib.suppress(Exception):
        deps.provenance().record_handoff(handoff)


_VALID_AGENT_ROLES: frozenset[str] = frozenset(
    {
        "route",
        "notebook",
        "problem",
        "model",
        "solver",
        "training",
        "mlflow",
        "lead",
        "triage",
        "data",
        "validation",
        "orchestrator",
    }
)
# Pin import so the AgentRole Literal stays referenced.
_ = AgentRole
