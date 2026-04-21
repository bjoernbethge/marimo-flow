"""SolverNode — placeholder; real implementation in Task 11."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class SolverNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        ctx.state.last_node = "solver"
        return RouteNode()
