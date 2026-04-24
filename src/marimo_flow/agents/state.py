"""FlowState — JSON-serialisable shared state across all graph nodes.

Live, non-serialisable objects (pina.Problem, torch.nn.Module, pina.Trainer)
are NOT held here. Only their MLflow artifact URIs are kept; the live
instances live in FlowDeps.registry, keyed by URI.

Typed specs (SPEC §8) sit next to the URIs as optional fields — they
are produced by the triage/problem/model/solver agents and consumed by
downstream agents and the provenance layer. Callers that only want the
old URI-based flow can ignore them; they default to ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from pydantic import BaseModel

from marimo_flow.agents.schemas import (
    AgentDecision,
    HandoffRecord,
    ModelSpec,
    ProblemSpec,
    RunConfig,
    SolverPlan,
    TaskSpec,
    ValidationReport,
)


@dataclass
class FlowState:
    user_intent: str | None = None

    # Typed specs (SPEC §8). Optional so the existing URI-based run path
    # stays backwards-compatible — the triage/problem/model/solver/validation
    # agents populate these as they run.
    task_spec: TaskSpec | None = None
    problem_spec: ProblemSpec | None = None
    model_spec: ModelSpec | None = None
    solver_plan: SolverPlan | None = None
    run_config: RunConfig | None = None
    validation_report: ValidationReport | None = None

    # MLflow artifact URIs — keys into FlowDeps.registry.
    problem_artifact_uri: str | None = None
    model_artifact_uri: str | None = None
    solver_artifact_uri: str | None = None
    training_artifact_uri: str | None = None
    training_run_id: str | None = None
    mlflow_run_id: str | None = None
    last_node: str | None = None
    history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # Provenance captured during the graph run — persisted alongside the
    # MLflow snapshots and later mirrored into the DuckDB provenance
    # tables (SPEC §12).
    decisions: list[AgentDecision] = field(default_factory=list)
    handoffs: list[HandoffRecord] = field(default_factory=list)
    experiment_id: str | None = None

    # Circuit breaker: how many RouteNode decisions have been made this
    # run. Raised by RouteNode each pass; caps runaway loops where the
    # router keeps picking the same branch without progress.
    route_count: int = 0
    max_route_steps: int = 12

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the state.

        Pydantic spec fields are rendered via ``model_dump(mode='json')``
        so datetimes, UUIDs and Literals survive a ``json.dumps`` round
        trip; plain dataclass fields pass through unchanged.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = _to_jsonable(value)
        return out


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    return value
