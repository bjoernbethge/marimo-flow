"""ModelNode — defines a neural architecture spec for the PINA Solver.

The agent composes the spec based on the registered Problem and pina skill;
no fixed enum of architectures (the spec can describe FNN/FNO/KAN/DeepONet
or any custom torch module the agent designs).
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

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

MODEL_SKILLS = ["pina-model"]


def _define_model(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        client.log_artifact(state.mlflow_run_id, str(p), artifact_path="model")
    uri = f"runs:/{state.mlflow_run_id}/model/model_spec.json"
    deps.registry[uri] = spec
    state.model_artifact_uri = uri
    return uri


@dataclass
class ModelNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode
        from marimo_flow.agents.toolsets.model import model_toolset

        model = self.model_override or get_model(
            "model", override=ctx.deps.models["model"]
        )
        ctx.deps.state = ctx.state
        agent = Agent(
            model,
            deps_type=FlowDeps,
            instructions=build_skill_instructions(MODEL_SKILLS),
            toolsets=[model_toolset],
        )
        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Problem URI: {ctx.state.problem_artifact_uri}\n"
            "Inspect the problem and design an architecture, then call define_model.",
            deps=ctx.deps,
        )
        ctx.state.last_node = "model"
        ctx.state.history.setdefault("model", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
