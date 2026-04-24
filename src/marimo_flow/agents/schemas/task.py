"""TaskSpec — the normalised scientist-facing request.

Triage produces this from free-form ``user_intent``; every downstream
agent reads it. Field list mirrors SPEC §8.1 and stays additive on top
of the existing FlowState — old callers that only set ``user_intent``
keep working.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from marimo_flow.agents.schemas.artifacts import DatasetBinding

ProblemKindHint = Literal[
    "forward",
    "inverse",
    "parameter_identification",
    "optimization",
    "operator_learning",
    "supervised",
    "unknown",
]


class TaskSpec(BaseModel):
    """Normalised request for the PINA team.

    Produced by the triage/spec agent; mutated by the orchestrator only
    in response to human-in-the-loop feedback. ``problem_kind`` is the
    category hint (forward/inverse/…); the concrete PINA preset lives
    in ``ProblemSpec.kind`` once the problem agent has translated it.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(default_factory=lambda: uuid4().hex)
    title: str
    description: str
    problem_kind: ProblemKindHint = "unknown"
    physics_domain: str | None = None
    equation_family: str | None = None
    geometry_type: str | None = None
    # Free-text hints — triage extracts from user intent; the problem
    # agent converts them into structured EquationSpec + ConditionSpec
    # for the ProblemSpec composition.
    boundary_conditions: list[str] = Field(default_factory=list)
    initial_conditions: list[str] = Field(default_factory=list)
    material_properties: dict[str, float] = Field(default_factory=dict)
    observables: list[str] = Field(default_factory=list)
    available_data: list[DatasetBinding] = Field(default_factory=list)
    objective_metrics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    compute_budget: dict[str, float | int] = Field(default_factory=dict)
    preferred_backend: str | None = None
    review_required: bool = False
    notes: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
