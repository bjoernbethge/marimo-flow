"""NotebookNode — interacts with the running marimo notebook via the marimo MCP server.

The agent is initialised with:
  * instructions = lazy callable concatenating skills [marimo, marimo-pair]
  * toolsets    = [marimo MCP server at http://127.0.0.1:2718/mcp/server]
  * tools       = list_skills, read_skill_reference (progressive disclosure)
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.mcp import build_marimo_mcp
from marimo_flow.agents.skills import (
    build_skill_instructions,
    list_skills,
    read_skill_reference,
)
from marimo_flow.agents.state import FlowState

NOTEBOOK_SKILLS = ["marimo", "marimo-pair"]


@dataclass
class NotebookNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model(
            "notebook", override=ctx.deps.models["notebook"]
        )
        toolsets = [build_marimo_mcp(url=ctx.deps.marimo_mcp_url)]
        agent = Agent(
            model,
            instructions=build_skill_instructions(NOTEBOOK_SKILLS),
            toolsets=toolsets,
        )

        @agent.tool
        def discover_skills(_ctx: RunContext[None]) -> list[str]:
            """List all installed skills (project + user)."""
            return list_skills()

        @agent.tool
        def fetch_skill_reference(
            _ctx: RunContext[None], name: str, ref_path: str
        ) -> str:
            """Read an additional file from a skill (e.g. references/api.md)."""
            return read_skill_reference(name, ref_path)

        prompt = (
            f"User intent: {ctx.state.user_intent}\n"
            "Inspect or modify the notebook to satisfy the intent."
        )
        result = await agent.run(prompt)
        ctx.state.last_node = "notebook"
        ctx.state.history.setdefault("notebook", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
