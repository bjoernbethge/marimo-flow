"""FunctionToolset for the Validation agent (SPEC §7.6).

Reads training metrics from MLflow, compares them against the
constraints carried in ``state.task_spec``, and records a
``ValidationReport`` that the orchestrator reads to decide
accept / retry / escalate / reject.
"""

from __future__ import annotations

import contextlib
from typing import Any

import mlflow
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import ValidationReport
from marimo_flow.agents.toolsets._registry import require_state

validation_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="validation")


@validation_toolset.tool
def evaluate_constraints(
    ctx: RunContext[FlowDeps],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Read MLflow metrics for the training run and score them.

    Args:
        thresholds: optional ``{metric_name: max_acceptable_value}`` —
            any metric above its threshold is marked ``"fail"``. Metrics
            listed in ``thresholds`` that aren't in MLflow are ``"warn"``.
            Metrics not in ``thresholds`` appear in the returned dict
            without a status so the validation agent can reason about them.

    Returns:
        ``{"metrics": {...}, "constraint_status": {...}}``.
    """
    state = require_state(ctx.deps)
    if state.training_run_id is None:
        raise ModelRetry(
            "No training run id yet — the training agent must run first, "
            "then retry evaluate_constraints."
        )
    client = mlflow.MlflowClient()
    run = client.get_run(state.training_run_id)
    metrics = {k: float(v) for k, v in run.data.metrics.items()}

    constraint_status: dict[str, str] = {}
    for name, threshold in (thresholds or {}).items():
        if name not in metrics:
            constraint_status[name] = "warn"
            continue
        constraint_status[name] = "pass" if metrics[name] <= threshold else "fail"
    return {"metrics": metrics, "constraint_status": constraint_status}


@validation_toolset.tool
def record_validation(
    ctx: RunContext[FlowDeps],
    verdict: str,
    rationale: str = "",
    metrics: dict[str, float] | None = None,
    constraint_status: dict[str, str] | None = None,
) -> str:
    """Persist a ValidationReport on state + in the provenance store.

    Args:
        verdict: one of ``accept``, ``retry``, ``escalate``, ``reject``.
        rationale: short reason.
        metrics / constraint_status: typically the dict returned by
            ``evaluate_constraints``.

    Returns the report_id.
    """
    if verdict not in {"accept", "retry", "escalate", "reject"}:
        raise ModelRetry(
            f"Unknown verdict {verdict!r}. Allowed: accept, retry, "
            "escalate, reject."
        )
    state = require_state(ctx.deps)
    task_id = state.task_spec.task_id if state.task_spec else None
    report = ValidationReport(
        task_id=task_id,
        run_id=state.training_run_id or state.mlflow_run_id,
        metrics=metrics or {},
        constraint_status=_coerce_status_dict(constraint_status or {}),
        verdict=verdict,  # type: ignore[arg-type]
        rationale=rationale or None,
    )
    state.validation_report = report
    with contextlib.suppress(Exception):
        ctx.deps.provenance().record_validation_report(report)
    return report.report_id


def _coerce_status_dict(raw: dict[str, str]) -> dict[str, Any]:
    """Clamp any non-{pass,warn,fail} status to ``warn``.

    LLMs occasionally invent values like ``"ok"``; clamp them instead
    of raising so the happy path doesn't break.
    """
    allowed = {"pass", "warn", "fail"}
    return {k: (v if v in allowed else "warn") for k, v in raw.items()}
