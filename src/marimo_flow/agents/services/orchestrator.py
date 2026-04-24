"""Orchestrator policy helpers (SPEC §7.1).

Pure functions that RouteNode / the lead workflow call to decide when
to short-circuit automated routing in favour of human review. Keeping
them here (instead of inlined in RouteNode) makes the policy isolated
and testable without spinning up a graph.

The current policies are intentionally narrow:

* ``check_escalation(state)`` — returns a human-readable reason string
  when the active ValidationReport says ``escalate`` or ``reject``.
* ``requires_human_review(task)`` — reads ``TaskSpec.review_required``.
* ``default_experiment_status(state)`` — maps the graph's end-state to
  ``completed`` / ``failed`` for ``ExperimentRecord.status``.
"""

from __future__ import annotations

from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState

ESCALATION_VERDICTS: frozenset[str] = frozenset({"escalate", "reject"})


def check_escalation(state: FlowState) -> str | None:
    """Return an escalation message if the run needs human review, else None.

    Today the single trigger is a ``ValidationReport`` with a verdict
    in ``ESCALATION_VERDICTS``. The return string is what RouteNode
    puts inside ``End(...)``.
    """
    report = state.validation_report
    if report is None or report.verdict not in ESCALATION_VERDICTS:
        return None
    return (
        f"Human review needed — validation verdict={report.verdict}. "
        f"Rationale: {report.rationale or 'n/a'}"
    )


def requires_human_review(task: TaskSpec | None) -> bool:
    """True when the caller wants a human to approve before finishing."""
    return bool(task and task.review_required)


def default_experiment_status(state: FlowState) -> str:
    """Infer the ``ExperimentRecord.status`` from the end-of-run state.

    * ``failed`` when the validation verdict was escalate/reject or the
      circuit breaker tripped without reaching training.
    * ``completed`` when training finished, with or without a verdict.
    * ``pending`` otherwise (nothing ran).
    """
    if state.validation_report and state.validation_report.verdict in ESCALATION_VERDICTS:
        return "failed"
    if state.training_run_id:
        return "completed"
    if state.route_count >= state.max_route_steps:
        return "failed"
    return "pending"
