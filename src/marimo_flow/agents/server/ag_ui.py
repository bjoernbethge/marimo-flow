"""AG-UI server — `marimo-flow` lead agent exposed over the AG-UI protocol."""

from __future__ import annotations

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def build_ag_ui_app(*, model=None, deps: FlowDeps | None = None, debug: bool = False):
    agent = build_lead_agent(model=model, deps=deps)
    return agent.to_ag_ui(deps=deps or FlowDeps(), debug=debug)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(build_ag_ui_app(), host="0.0.0.0", port=8001)
