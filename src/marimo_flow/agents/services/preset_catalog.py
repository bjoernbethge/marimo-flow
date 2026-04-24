"""PresetCatalog — persistent library of reusable Problem/Model/Solver configs.

Sits on top of the DuckDB provenance store. Every registered preset is:

1. a ``PresetRecord`` in one of the ``preset_<family>`` tables
2. optionally mirrored to ``.marimo-flow/presets/<family>/<name>.yaml``
   for VCS-friendly sharing across sessions / machines
3. hooked into the corresponding in-memory Manager registry
   (``ProblemManager._PRESETS`` / ``ModelManager._REGISTRY`` /
   ``SolverManager._REGISTRY``) so ``Manager.create(preset_name, ...)``
   works unchanged

Runtime kwargs merge on top of the preset's stored defaults (runtime wins).
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from marimo_flow.agents.schemas import PresetFamily, PresetRecord
from marimo_flow.agents.services.provenance import ProvenanceStore
from marimo_flow.core import ModelManager, ProblemManager, SolverManager

_MANAGER_BY_FAMILY: dict[str, Any] = {
    "problem": ProblemManager,
    "model": ModelManager,
    "solver": SolverManager,
}

_DEFAULT_MIRROR_DIR = Path(".marimo-flow") / "presets"


def resolve_builder(ref: str) -> Callable[..., Any]:
    """Import-path resolver for ``"module.path:Object.method"`` refs.

    Supports dotted ``Object.method`` traversal so static/class methods
    on Manager classes can be referenced without pre-binding.
    """
    if ":" not in ref:
        raise ValueError(
            f"builder_ref must be 'module:Attr[.nested]', got {ref!r}"
        )
    mod_path, obj_path = ref.split(":", 1)
    obj: Any = importlib.import_module(mod_path)
    for part in obj_path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"builder_ref {ref!r} did not resolve to a callable")
    return obj


def _row_to_record(row: dict[str, Any], family: str) -> PresetRecord:
    """DuckDB row dict → PresetRecord (unpacking the JSON columns)."""
    tags = row.get("tags") or "[]"
    spec = row.get("spec_json") or "{}"
    return PresetRecord(
        name=row["name"],
        family=family,  # type: ignore[arg-type]
        version=int(row.get("version") or 1),
        builder_ref=row["builder_ref"],
        spec_json=json.loads(spec) if isinstance(spec, str) else spec,
        description=row.get("description") or "",
        tags=json.loads(tags) if isinstance(tags, str) else list(tags),
        parent_name=row.get("parent_name"),
        author=row.get("author") or "user",
        status=row.get("status") or "active",  # type: ignore[arg-type]
        created_at=row.get("created_at") or datetime.now(UTC),
    )


class PresetCatalog:
    """Library façade over the DuckDB preset tables + optional YAML mirror.

    One instance per FlowDeps. Agents consume via ``curator_toolset``;
    direct users can instantiate ``PresetCatalog(store)`` themselves.
    """

    def __init__(
        self,
        store: ProvenanceStore,
        *,
        mirror_dir: Path | str | None = _DEFAULT_MIRROR_DIR,
    ) -> None:
        self.store = store
        self.mirror_dir = Path(mirror_dir) if mirror_dir else None

    # --- read ---

    def list(
        self, family: PresetFamily, *, status: str | None = "active"
    ) -> list[PresetRecord]:
        rows = self.store.list_presets(family, status=status)
        return [_row_to_record(r, family) for r in rows]

    def get(self, family: PresetFamily, name: str) -> PresetRecord | None:
        row = self.store.get_preset(family, name)
        return _row_to_record(row, family) if row else None

    def search(
        self,
        family: PresetFamily,
        query: str = "",
        tags: list[str] | None = None,
    ) -> list[PresetRecord]:
        """Simple LIKE search over name/description + tag filter."""
        records = self.list(family, status="active")
        query_lower = query.strip().lower()
        filtered: list[PresetRecord] = []
        for rec in records:
            if query_lower and query_lower not in (
                rec.name + " " + rec.description
            ).lower():
                continue
            if tags and not all(t in rec.tags for t in tags):
                continue
            filtered.append(rec)
        return filtered

    # --- write ---

    def register(self, record: PresetRecord, *, mirror: bool = True) -> PresetRecord:
        """Persist a new / updated preset. Runs the builder-ref sanity check."""
        resolve_builder(record.builder_ref)  # fail fast on bad ref
        self.store.upsert_preset(record)
        if mirror and self.mirror_dir is not None:
            self._write_mirror(record)
        return record

    def clone(
        self,
        family: PresetFamily,
        source_name: str,
        new_name: str,
        *,
        overrides: dict[str, Any] | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> PresetRecord:
        """Copy an existing preset under a new name with merged kwargs."""
        source = self.get(family, source_name)
        if source is None:
            raise KeyError(f"Preset {source_name!r} not found in {family}")
        merged = dict(source.spec_json)
        if overrides:
            merged = _deep_merge(merged, overrides)
        clone = PresetRecord(
            name=new_name,
            family=family,
            builder_ref=source.builder_ref,
            spec_json=merged,
            description=description or source.description,
            tags=list(tags if tags is not None else source.tags),
            parent_name=source_name,
        )
        return self.register(clone)

    def deprecate(self, family: PresetFamily, name: str) -> None:
        self.store.set_preset_status(family, name, "deprecated")
        if self.mirror_dir is not None:
            record = self.get(family, name)
            if record is not None:
                self._write_mirror(record)

    # --- integration with Managers ---

    def load_into_managers(self) -> int:
        """Register every active preset into its family's Manager._REGISTRY.

        Returns the number of presets registered across all three families.
        Idempotent — re-registering is cheap and does not duplicate.
        """
        count = 0
        for family, manager in _MANAGER_BY_FAMILY.items():
            for record in self.list(family, status="active"):
                factory = self._make_factory(record)
                manager.register(record.name, factory)
                count += 1
        return count

    # --- helpers ---

    def _make_factory(self, record: PresetRecord) -> Callable[..., Any]:
        """Wrap the builder so stored kwargs are merged with runtime kwargs.

        Runtime kwargs win on key conflict — presets are *defaults*, not
        locks. The wrapped factory is what ends up in ``Manager._REGISTRY``.
        """
        builder = resolve_builder(record.builder_ref)
        stored = dict(record.spec_json.get("kwargs") or {})

        def _factory(**runtime_kwargs: Any) -> Any:
            merged = _deep_merge(stored, runtime_kwargs)
            return builder(**merged)

        _factory.__name__ = f"preset_{record.family}_{record.name}"
        return _factory

    def _write_mirror(self, record: PresetRecord) -> None:
        target_dir = self.mirror_dir / record.family  # type: ignore[operator]
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = record.model_dump(mode="json")
        (target_dir / f"{record.name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge where ``override`` wins on conflict.

    Nested dicts merge key-wise; everything else is replaced wholesale.
    """
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result
