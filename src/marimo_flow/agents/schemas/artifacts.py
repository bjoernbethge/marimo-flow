"""ArtifactRef and DatasetBinding — typed pointers into MLflow + the registry.

ArtifactRef carries the kind of artifact so callers can resolve a URI to
the right FlowDeps.registry entry (pina.Problem, torch.nn.Module, solver,
trainer) without re-parsing the URI. DatasetBinding describes data that
feeds ProblemManager.create_from_dataframe / create_supervised_problem.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ArtifactKind = Literal[
    "problem", "model", "solver", "training", "dataset", "validation", "other"
]


class ArtifactRef(BaseModel):
    """Typed reference to an artifact living in MLflow + FlowDeps.registry."""

    model_config = ConfigDict(extra="forbid")

    kind: ArtifactKind
    uri: str = Field(description="MLflow artifact URI — also the registry key")
    label: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )


class DatasetBinding(BaseModel):
    """Binding of a dataset to the active workflow.

    Supports in-memory tensors (``source='tensor'``, ``location`` is a
    registry URI) and tabular sources consumed by
    ProblemManager.create_from_dataframe (``source='parquet' | 'csv' |
    'dataframe'``, ``location`` is a file path).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    source: Literal["tensor", "dataframe", "parquet", "csv", "other"]
    location: str | None = None
    input_columns: list[str] | None = None
    output_columns: list[str] | None = None
    n_rows: int | None = None
    checksum: str | None = None
    notes: str | None = None
