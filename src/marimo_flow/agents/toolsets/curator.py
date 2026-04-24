"""FunctionToolset for the preset-catalog curator (SPEC reuse-first).

Exposed to every specialist agent (problem/model/solver/triage) so the
team can search the persistent catalog before building a fresh spec,
and register successful configs back for future reuse. Without this
toolset the agents would rebuild the same Burgers / FNO / PINN config
from scratch every run and the catalog would stay empty.

Tools:
    search_presets       — LIKE-search by query + tag filter
    list_presets         — full list per family (active only by default)
    describe_preset      — detailed view incl. kwargs, tags, parent
    register_preset      — persist a new preset (validates builder_ref)
    clone_preset         — copy under a new name with kwargs overrides
    deprecate_preset     — mark inactive; other callers stop seeing it
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import PresetRecord

curator_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="curator")

_VALID_FAMILIES: frozenset[str] = frozenset({"problem", "model", "solver"})


def _require_family(family: str) -> None:
    if family not in _VALID_FAMILIES:
        raise ModelRetry(
            f"Unknown preset family {family!r}. Allowed: "
            f"{', '.join(sorted(_VALID_FAMILIES))}."
        )


def _record_to_dict(rec: PresetRecord) -> dict[str, Any]:
    return rec.model_dump(mode="json")


@curator_toolset.tool
def list_presets(
    ctx: RunContext[FlowDeps],
    family: str,
    include_deprecated: bool = False,
) -> list[dict[str, Any]]:
    """List every preset registered for ``family`` (problem / model / solver).

    Active only by default; set ``include_deprecated=True`` to see all.
    Returns flat dicts suitable for the LLM to pattern-match on.
    """
    _require_family(family)
    catalog = ctx.deps.preset_catalog()
    status = None if include_deprecated else "active"
    return [_record_to_dict(r) for r in catalog.list(family, status=status)]  # type: ignore[arg-type]


@curator_toolset.tool
def search_presets(
    ctx: RunContext[FlowDeps],
    family: str,
    query: str = "",
    tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search active presets by name/description substring + optional tag filter.

    Call this BEFORE building a new ProblemSpec / ModelSpec / SolverPlan —
    if a match exists, clone it instead of building from scratch.
    """
    _require_family(family)
    catalog = ctx.deps.preset_catalog()
    results = catalog.search(family, query=query, tags=tags)  # type: ignore[arg-type]
    return [_record_to_dict(r) for r in results]


@curator_toolset.tool
def describe_preset(
    ctx: RunContext[FlowDeps], family: str, name: str
) -> dict[str, Any]:
    """Return the full PresetRecord for ``name``, including stored kwargs."""
    _require_family(family)
    catalog = ctx.deps.preset_catalog()
    record = catalog.get(family, name)  # type: ignore[arg-type]
    if record is None:
        raise ModelRetry(
            f"No preset named {name!r} in family {family!r}. "
            f"Call list_presets or search_presets first."
        )
    return _record_to_dict(record)


@curator_toolset.tool
def register_preset(
    ctx: RunContext[FlowDeps],
    family: str,
    name: str,
    builder_ref: str,
    description: str,
    spec_json: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> str:
    """Persist a new reusable preset in the catalog.

    Args:
        family: one of 'problem', 'model', 'solver'.
        name: unique key — convention ``<domain>_<descriptor>_<version>``,
            e.g. ``burgers_nu0p02_2d``. Avoid ``builtin.*`` namespace.
        builder_ref: import path ``'module.path:Object.method'`` of the
            callable that constructs the instance. Typically one of the
            ``Manager.create`` classmethods.
        description: one-sentence summary; shown in search results.
        spec_json: dict with ``kwargs`` key holding the default kwargs
            for the builder; the wrapped factory merges runtime kwargs
            on top of these (runtime wins).
        tags: free-form tag list for filtering (e.g. ['2d','incompressible']).

    Returns the preset's registered name on success.
    """
    _require_family(family)
    if name.startswith("builtin."):
        raise ModelRetry(
            "Preset names prefixed with 'builtin.' are reserved for "
            "auto-seeded records. Use a different name."
        )
    catalog = ctx.deps.preset_catalog()
    record = PresetRecord(
        name=name,
        family=family,  # type: ignore[arg-type]
        builder_ref=builder_ref,
        spec_json=spec_json or {},
        description=description,
        tags=list(tags) if tags else [],
    )
    try:
        catalog.register(record)
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        raise ModelRetry(
            f"Could not register preset {name!r}: {exc}. "
            "Check builder_ref points to an importable callable."
        ) from exc
    # Reload into managers so this preset is usable immediately this session.
    catalog.load_into_managers()
    return name


@curator_toolset.tool
def clone_preset(
    ctx: RunContext[FlowDeps],
    family: str,
    source_name: str,
    new_name: str,
    overrides: dict[str, Any] | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Copy an existing preset under a new name with merged kwargs.

    Use this when an existing preset ALMOST fits — e.g. clone an
    already-registered Burgers composition and override the viscosity
    in its ``parameters`` block. The clone's ``parent_name`` records
    the lineage.

    ``overrides`` is deep-merged into the source's ``spec_json`` (runtime
    structure wins). Shape depends on family:

    * ``problem``: ``{"equations": [{"parameters": {"nu": 0.005}}], ...}``
      (or replace whole subtrees — deep-merge keeps sibling keys)
    * ``model``: ``{"kwargs": {"layers": [128, 128]}}``
    * ``solver``: ``{"kwargs": {"learning_rate": 5e-4}}``
    """
    _require_family(family)
    if new_name.startswith("builtin."):
        raise ModelRetry("'builtin.' prefix is reserved — pick another name.")
    catalog = ctx.deps.preset_catalog()
    try:
        record = catalog.clone(
            family,  # type: ignore[arg-type]
            source_name,
            new_name,
            overrides=overrides,
            description=description,
            tags=tags,
        )
    except KeyError as exc:
        raise ModelRetry(str(exc)) from exc
    catalog.load_into_managers()
    return record.name


@curator_toolset.tool
def deprecate_preset(ctx: RunContext[FlowDeps], family: str, name: str) -> str:
    """Mark a preset as deprecated. Hidden from ``search_presets`` by default."""
    _require_family(family)
    if name.startswith("builtin."):
        raise ModelRetry(
            "Built-in presets can't be deprecated — they're the "
            "Manager's baseline. Clone one instead if you want to "
            "supersede it."
        )
    catalog = ctx.deps.preset_catalog()
    if catalog.get(family, name) is None:  # type: ignore[arg-type]
        raise ModelRetry(f"No preset named {name!r} in family {family!r}.")
    catalog.deprecate(family, name)  # type: ignore[arg-type]
    return name
