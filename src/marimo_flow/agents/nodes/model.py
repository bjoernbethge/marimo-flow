"""ModelNode — placeholder; real implementation in Task 10."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class ModelNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        ctx.state.last_node = "model"
        return RouteNode()
