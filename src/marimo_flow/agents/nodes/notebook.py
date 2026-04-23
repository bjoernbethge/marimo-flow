"""NotebookNode — interacts with the running marimo notebook via the marimo MCP server.

The agent is initialised with:
  * instructions = lazy callable concatenating skills [marimo, marimo-pair]
  * toolsets     = [skills_toolset, marimo MCP server]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.mcp import build_marimo_mcp
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets.skills import skills_toolset

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

NOTEBOOK_SKILLS = ["marimo", "marimo-pair"]


@dataclass
class NotebookNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model(
            "notebook", override=ctx.deps.models["notebook"]
        )
        ctx.deps.state = ctx.state
        agent = Agent(
            model,
            deps_type=FlowDeps,
            instructions=build_skill_instructions(NOTEBOOK_SKILLS),
            toolsets=[skills_toolset, build_marimo_mcp(url=ctx.deps.marimo_mcp_url)],
        )
        prompt = (
            f"User intent: {ctx.state.user_intent}\n"
            "Inspect or modify the notebook to satisfy the intent."
        )
        result = await agent.run(prompt, deps=ctx.deps)
        ctx.state.last_node = "notebook"
        ctx.state.history.setdefault("notebook", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
