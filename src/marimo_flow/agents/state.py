"""FlowState — JSON-serialisable shared state across all graph nodes.

Live, non-serialisable objects (pina.Problem, torch.nn.Module, pina.Trainer)
are NOT held here. Only their MLflow artifact URIs are kept; the live
instances live in FlowDeps.registry, keyed by URI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowState:
    user_intent: str | None = None
    problem_artifact_uri: str | None = None
    model_artifact_uri: str | None = None
    solver_artifact_uri: str | None = None
    training_artifact_uri: str | None = None
    training_run_id: str | None = None
    mlflow_run_id: str | None = None
    last_node: str | None = None
    history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
