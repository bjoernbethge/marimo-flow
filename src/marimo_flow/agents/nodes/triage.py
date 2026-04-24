"""TriageNode — parses free-form ``user_intent`` into a typed TaskSpec.

Sits at the start of the graph before RouteNode. One-shot:

* If ``state.task_spec`` is already set (caller built it manually),
  skip the LLM and go straight to RouteNode.
* Otherwise call the triage LLM with ``output_type=TaskSpec`` and
  store the result on ``state.task_spec``.

Every successful triage also emits an ``AgentDecision`` and, if the
provenance store is reachable, mirrors the TaskSpec + decision into
DuckDB so the rest of the graph has a stable audit trail.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import AgentDecision, TaskSpec
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

TRIAGE_INSTRUCTIONS = """\
You are the triage agent for a PINA (Physics-Informed NN) team.
Parse a free-form user intent into a structured TaskSpec.

Rules:
- title: short label (5–10 words).
- description: 1–2 sentences rephrasing the request.
- problem_kind: one of forward, inverse, parameter_identification,
  optimization, operator_learning, supervised, unknown.
- equation_family: fill only if clearly identifiable (poisson, heat,
  burgers, allen_cahn, advection_diffusion, helmholtz, wave). Leave
  null otherwise.
- physics_domain: e.g. fluid, heat_transfer, structural, electromagnetics.
- geometry_type: e.g. unit_square, 1d_interval, 2d_plate.
- observables: list quantities the user explicitly mentioned.
- constraints: hard requirements the user stated (accuracy, determinism,
  budget) as short strings.
- compute_budget: only include keys the user actually gave you
  (e.g. max_epochs, wall_time_seconds).
- review_required: true when the intent is ambiguous, safety-critical,
  or the user explicitly asked for review. Default false.
- notes: short pointer to unresolved questions.

Be conservative: prefer null over guessing.
"""


def _record_triage_decision(state: FlowState, task: TaskSpec, deps: FlowDeps) -> None:
    """Append an AgentDecision to state and mirror it to provenance."""
    decision = AgentDecision(
        agent="triage",
        tool=None,
        summary=f"Built TaskSpec: {task.title}",
        task_id=task.task_id,
        run_id=state.mlflow_run_id,
        output_schema="TaskSpec",
    )
    state.decisions.append(decision)
    with contextlib.suppress(Exception):
        store = deps.provenance()
        store.record_task(task)
        store.record_decision(decision)


@dataclass
class TriageNode(BaseNode[FlowState, FlowDeps, str]):
    """First node in the graph. Produces a TaskSpec from ``user_intent``."""

    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode

        # Caller already built a TaskSpec → skip the LLM entirely.
        if ctx.state.task_spec is not None:
            ctx.state.last_node = "triage"
            return RouteNode()

        model = self.model_override or ctx.deps.model_for("triage")
        agent = Agent(model, output_type=TaskSpec, instructions=TRIAGE_INSTRUCTIONS)
        prompt = f"User intent: {ctx.state.user_intent!r}\nProduce a TaskSpec."
        result = await agent.run(prompt)
        task = result.output
        ctx.state.task_spec = task
        ctx.state.last_node = "triage"
        ctx.state.history.setdefault("triage", []).append(
            {"role": "assistant", "content": task.title}
        )
        _record_triage_decision(ctx.state, task, ctx.deps)
        return RouteNode()
