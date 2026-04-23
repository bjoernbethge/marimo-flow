"""TrainingNode — fits the registered solver via pina.Trainer.

Runs `training_toolset` which wraps `core.train_solver`. Requires that
a problem + model + solver have already been registered by their
respective sub-nodes earlier in the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

TRAINING_SKILLS = ["pina-training"]


@dataclass
class TrainingNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode
        from marimo_flow.agents.toolsets.training import training_toolset

        role_model = ctx.deps.models.get("training")
        model = self.model_override or get_model("training", override=role_model)
        ctx.deps.state = ctx.state
        agent = Agent(
            model,
            deps_type=FlowDeps,
            instructions=build_skill_instructions(TRAINING_SKILLS),
            toolsets=[training_toolset],
        )
        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Solver URI: {ctx.state.solver_artifact_uri}\n"
            "Call discretise_domain (usually n=1000, random) and then train "
            "(max_epochs 1000 by default, accelerator='auto'). Start small "
            "if the user asks for a quick test.",
            deps=ctx.deps,
        )
        ctx.state.last_node = "training"
        ctx.state.history.setdefault("training", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
