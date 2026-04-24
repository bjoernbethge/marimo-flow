"""Typed schemas for the PINA agent team (SPEC §8).

Composition-first: ``ProblemSpec`` is built from ``EquationSpec`` +
``SubdomainSpec`` + ``ConditionSpec`` primitives — no hardcoded kind
enum. Any PDE expressible in PINA is reachable without touching Python.
"""

from marimo_flow.agents.schemas.artifacts import (
    ArtifactKind,
    ArtifactRef,
    DatasetBinding,
)
from marimo_flow.agents.schemas.decisions import (
    AgentDecision,
    AgentRole,
    ExperimentRecord,
    HandoffRecord,
    ValidationReport,
)
from marimo_flow.agents.schemas.equation import (
    ConditionKind,
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    SubdomainSpec,
)
from marimo_flow.agents.schemas.preset import (
    PresetFamily,
    PresetRecord,
    PresetStatus,
)
from marimo_flow.agents.schemas.problem import ProblemSpec
from marimo_flow.agents.schemas.run import (
    ModelKind,
    ModelSpec,
    RunConfig,
    SolverKind,
    SolverPlan,
)
from marimo_flow.agents.schemas.task import ProblemKindHint, TaskSpec

__all__ = [
    "AgentDecision",
    "AgentRole",
    "ArtifactKind",
    "ArtifactRef",
    "ConditionKind",
    "ConditionSpec",
    "DatasetBinding",
    "DerivativeSpec",
    "EquationSpec",
    "ExperimentRecord",
    "HandoffRecord",
    "ModelKind",
    "ModelSpec",
    "PresetFamily",
    "PresetRecord",
    "PresetStatus",
    "ProblemKindHint",
    "ProblemSpec",
    "RunConfig",
    "SolverKind",
    "SolverPlan",
    "SubdomainSpec",
    "TaskSpec",
    "ValidationReport",
]
