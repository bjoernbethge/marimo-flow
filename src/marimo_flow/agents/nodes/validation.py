"""ValidationNode — grades a completed training run against task constraints.

Runs after TrainingNode. Agent reads ``task_spec.constraints`` and the
MLflow metrics of ``training_run_id``, calls ``evaluate_constraints``
to score them, and calls ``record_validation`` to persist a
``ValidationReport`` on state + in the DuckDB provenance store.

RouteNode reads ``state.validation_report.verdict`` to decide whether
to finish (``accept``) or end with an escalation message
(``escalate`` / ``reject``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

VALIDATION_INSTRUCTIONS = """\
You validate a completed PINA training run.

Steps:
1. Call `evaluate_constraints` (optionally with thresholds built from
   the task constraints) to read MLflow metrics and score them.
2. Decide a verdict:
     * accept   — all constraints pass, metrics look sensible.
     * retry    — close miss; another training pass with tuned kwargs
                  is likely to fix it.
     * escalate — ambiguous or out-of-scope; hand back to the human.
     * reject   — fundamentally wrong setup; stop.
3. Call `record_validation` with the verdict, a short rationale, the
   metrics, and the constraint_status dict.

Prefer `accept` when there are no explicit constraints and training
converged. Prefer `escalate` when review_required was set on the task.
"""


@dataclass
class ValidationNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode
        from marimo_flow.agents.toolsets.validation import validation_toolset

        model = self.model_override or ctx.deps.model_for("validation")
        ctx.deps.state = ctx.state
        agent = Agent(
            model,
            deps_type=FlowDeps,
            instructions=VALIDATION_INSTRUCTIONS,
            toolsets=[validation_toolset],
            retries=3,
        )
        constraints = (
            ctx.state.task_spec.constraints if ctx.state.task_spec else []
        )
        review_required = bool(
            ctx.state.task_spec and ctx.state.task_spec.review_required
        )
        prompt = (
            f"Training run: {ctx.state.training_run_id}\n"
            f"Task constraints: {constraints}\n"
            f"Review requested by user: {review_required}\n"
            "Validate the run and record the verdict."
        )
        result = await agent.run(prompt, deps=ctx.deps)
        ctx.state.last_node = "validation"
        ctx.state.history.setdefault("validation", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
