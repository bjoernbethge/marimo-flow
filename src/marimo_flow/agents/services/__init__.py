"""Cross-cutting services: provenance store, orchestrator policy, experiment lifecycle.

The provenance store is a DuckDB-backed log of every typed spec, agent
decision, handoff and artifact — sitting next to MLflow (which still
owns the actual artifacts) so the graph can be queried with SQL.

Orchestrator helpers are pure functions over FlowState that RouteNode
and the lead workflow call to decide escalation and experiment status.
"""

from marimo_flow.agents.services.composer import build_equation, compose_problem
from marimo_flow.agents.services.experiment import (
    complete_experiment,
    start_experiment,
)
from marimo_flow.agents.services.orchestrator import (
    ESCALATION_VERDICTS,
    check_escalation,
    default_experiment_status,
    requires_human_review,
)
from marimo_flow.agents.services.preset_catalog import (
    PresetCatalog,
    resolve_builder,
)
from marimo_flow.agents.services.provenance import (
    DEFAULT_PROVENANCE_DB_PATH,
    ProvenanceStore,
)

__all__ = [
    "DEFAULT_PROVENANCE_DB_PATH",
    "ESCALATION_VERDICTS",
    "PresetCatalog",
    "ProvenanceStore",
    "build_equation",
    "check_escalation",
    "complete_experiment",
    "compose_problem",
    "default_experiment_status",
    "requires_human_review",
    "resolve_builder",
    "start_experiment",
]
