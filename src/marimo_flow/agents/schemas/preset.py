"""PresetRecord — reusable Problem / Model / Solver configuration.

The catalog stores one PresetRecord per named, reusable configuration.
Each record points to a Python builder via ``builder_ref`` (import path
of a callable) plus a ``spec_json`` dict of default kwargs. Runtime
callers can override the stored kwargs.

The catalog is persisted in the DuckDB provenance store (table
``preset_<family>``) and optionally mirrored to YAML files under
``.marimo-flow/presets/<family>/<name>.yaml`` so presets can be
version-controlled alongside the code.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PresetFamily = Literal["problem", "model", "solver"]
PresetStatus = Literal["active", "deprecated", "experimental"]


class PresetRecord(BaseModel):
    """One reusable configuration keyed by ``name``.

    ``builder_ref`` is an ``"module.path:Object.method"`` import path
    that resolves to a callable; ``spec_json`` holds the default kwargs
    for that callable. The runtime factory merges stored kwargs with
    runtime kwargs (runtime wins) before calling the builder.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="unique key in the catalog")
    family: PresetFamily
    version: int = 1
    builder_ref: str = Field(
        description="'module.path:Object.method' import path of the builder"
    )
    spec_json: dict[str, Any] = Field(
        default_factory=dict,
        description="default kwargs for the builder + optional spec snapshot",
    )
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    parent_name: str | None = None
    author: str = "user"
    status: PresetStatus = "active"
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
