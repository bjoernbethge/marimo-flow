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
from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

SOLVER_SKILLS = ["pina"]


def _define_solver(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "solver_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        mlflow.log_artifact(str(p), artifact_path="solver")
    uri = f"runs:/{state.mlflow_run_id}/solver/solver_spec.json"
    deps.registry[uri] = spec
    state.solver_artifact_uri = uri
    return uri


@dataclass
class SolverNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model(
            "solver", override=ctx.deps.models["solver"]
        )
        agent = Agent(model, instructions=build_skill_instructions(SOLVER_SKILLS))

        @agent.tool
        def define_solver(rc: RunContext[None], spec: dict[str, Any]) -> str:
            """Define a Solver+Trainer spec tailored to the problem and model.

            `spec` is free-form JSON the agent designs (e.g.
            {kind:"PINN", optimiser:"adam", lr:1e-3, max_epochs:5000,
             loss_weights:{interior:1.0, boundary:10.0}}). Returns MLflow URI.
            """
            return _define_solver(spec, ctx.deps, ctx.state)

        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Problem URI: {ctx.state.problem_artifact_uri}\n"
            f"Model URI:   {ctx.state.model_artifact_uri}\n"
            "Inspect the problem and model, then design a solver+trainer spec "
            "and call define_solver."
        )
        ctx.state.last_node = "solver"
        ctx.state.history.setdefault("solver", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
