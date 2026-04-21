"""Lead agent — single Pydantic-AI Agent wrapping the graph as one tool.

Used by:
  * marimo `mo.ui.chat` (see chat.py)
  * A2A    `agent.to_a2a()`  (see server/a2a.py)
  * AG-UI  `agent.to_ag_ui()` (see server/ag_ui.py)

mlflow.pydantic_ai.autolog() is enabled here so every nested sub-agent
call inside the graph produces traces under the active MLflow run.
"""

from __future__ import annotations

import mlflow
from pydantic_ai import Agent, RunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState

LEAD_INSTRUCTIONS = """\
You are the lead of a PINA (Physics-Informed NN) team.
For any user request that needs the team, call run_pina_workflow(intent).
For trivial chit-chat, answer directly.
"""

_AUTOLOG_ENABLED = False


def _ensure_autolog() -> None:
    global _AUTOLOG_ENABLED
    if _AUTOLOG_ENABLED:
        return
    mlflow.pydantic_ai.autolog()
    mlflow.pytorch.autolog()
    _AUTOLOG_ENABLED = True


def build_lead_agent(*, model=None, deps: FlowDeps | None = None) -> Agent:
    _ensure_autolog()
    model = model or get_model("lead")
    deps = deps or FlowDeps()
    graph = build_graph()
    agent = Agent(model, instructions=LEAD_INSTRUCTIONS)

    @agent.tool
    async def run_pina_workflow(_rc: RunContext[None], intent: str) -> str:
        """Run the PINA team graph end-to-end. Returns the team's final summary."""
        if mlflow.active_run() is None:
            run = mlflow.start_run()
            run_id = run.info.run_id
        else:
            run_id = mlflow.active_run().info.run_id
        state = FlowState(user_intent=intent, mlflow_run_id=run_id)
        persistence = MLflowStatePersistence(run_id=run_id)
        persistence.set_graph_types(graph)
        result = await graph.run(
            start_node(), state=state, deps=deps, persistence=persistence
        )
        return str(result.output)

    return agent
