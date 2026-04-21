"""MLflowNode — talks to the mlflow MCP server for tracking + registry ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.mcp import build_mlflow_mcp
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

MLFLOW_SKILLS = ["mlflow"]


@dataclass
class MLflowNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model(
            "mlflow", override=ctx.deps.models["mlflow"]
        )
        toolsets = [build_mlflow_mcp(tracking_uri=ctx.deps.mlflow_tracking_uri)]
        agent = Agent(
            model,
            instructions=build_skill_instructions(MLFLOW_SKILLS),
            toolsets=toolsets,
        )

        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Active MLflow run: {ctx.state.mlflow_run_id}\n"
            "Use the MLflow MCP tools to satisfy the request."
        )
        ctx.state.last_node = "mlflow"
        ctx.state.history.setdefault("mlflow", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
