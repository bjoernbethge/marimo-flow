"""Provenance records — AgentDecision, HandoffRecord, ValidationReport, ExperimentRecord.

These feed the SPEC §12 persistence layer (DuckDB tables
``agent_decisions``, ``handoff_records``, ``experiments``). Today they
are also emitted into ``FlowState.decisions`` / ``FlowState.handoffs``
so the graph carries provenance even before DuckDB is wired up.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

AgentRole = Literal[
    "route",
    "notebook",
    "problem",
    "model",
    "solver",
    "training",
    "mlflow",
    "lead",
    "triage",
    "data",
    "validation",
    "orchestrator",
    "design",
    "control",
]


class AgentDecision(BaseModel):
    """One step taken by one agent.

    Captures the metadata needed to replay or audit a decision: which
    agent, which tool, which spec versions went in and out, plus a short
    human-readable summary. ``input_schema`` / ``output_schema`` are
    free-form version tags (e.g. ``"ProblemSpec@1"``) — bump them when
    the spec shape changes.
    """

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str | None = None
    run_id: str | None = None
    agent: AgentRole
    tool: str | None = None
    input_schema: str | None = None
    output_schema: str | None = None
    summary: str
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HandoffRecord(BaseModel):
    """Transition between two agents plus the artifacts they agreed on."""

    model_config = ConfigDict(extra="forbid")

    handoff_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str | None = None
    run_id: str | None = None
    from_agent: AgentRole
    to_agent: AgentRole
    reason: str
    artifact_uris: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ValidationReport(BaseModel):
    """Outcome of a validation/review pass.

    ``metrics`` holds scalar final scores; ``constraint_status`` records
    per-constraint pass/warn/fail; ``verdict`` is the call the
    orchestrator reads to decide accept / retry / escalate / reject.
    """

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str | None = None
    run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    constraint_status: dict[str, Literal["pass", "warn", "fail"]] = Field(
        default_factory=dict
    )
    verdict: Literal["accept", "retry", "escalate", "reject"] = "accept"
    rationale: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExperimentRecord(BaseModel):
    """One end-to-end run — binds a task to its artifacts + report.

    The provenance layer uses this as the primary row in the
    ``experiments`` table; MLflow still owns the actual artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str | None = None
    run_id: str | None = None
    problem_artifact_uri: str | None = None
    model_artifact_uri: str | None = None
    solver_artifact_uri: str | None = None
    training_artifact_uri: str | None = None
    validation_report_id: str | None = None
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
