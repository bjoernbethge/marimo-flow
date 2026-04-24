"""SolverNode — defines a PINA Solver + Trainer config tailored to the problem+model.

The agent designs the spec freely (kind, optimiser, scheduler, loss weights,
epochs, batch size). A follow-up `train()` tool can invoke the actual
pina.Trainer.fit(); mlflow.pytorch.autolog() (enabled by build_lead_agent)
captures metrics and checkpoints automatically.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

SOLVER_SKILLS = ["pina-solver"]


def _define_solver(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "solver_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        client.log_artifact(state.mlflow_run_id, str(p), artifact_path="solver")
    uri = f"runs:/{state.mlflow_run_id}/solver/solver_spec.json"
    deps.registry[uri] = spec
    state.solver_artifact_uri = uri
    return uri


@dataclass
class SolverNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode
        from marimo_flow.agents.toolsets.solver import solver_toolset

        model = self.model_override or ctx.deps.model_for("solver")
        ctx.deps.state = ctx.state
        agent = Agent(
            model,
            deps_type=FlowDeps,
            instructions=build_skill_instructions(SOLVER_SKILLS),
            toolsets=[solver_toolset],
            retries=3,
        )
        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Problem URI: {ctx.state.problem_artifact_uri}\n"
            f"Model URI:   {ctx.state.model_artifact_uri}\n"
            "Inspect the problem and model, then design a solver+trainer spec "
            "and call define_solver.",
            deps=ctx.deps,
        )
        ctx.state.last_node = "solver"
        ctx.state.history.setdefault("solver", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
