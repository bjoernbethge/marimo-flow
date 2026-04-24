"""Tests for the curator_toolset — post-pivot (composition catalog, no builtins)."""

from __future__ import annotations

import pytest
from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets.curator import curator_toolset


class _Ctx:
    def __init__(self, deps):
        self.deps = deps


@pytest.fixture
def deps():
    return FlowDeps(state=FlowState(), provenance_db_path=":memory:")


def _burgers_spec() -> dict:
    return {
        "name": "burgers_1d_nu0p01",
        "output_variables": ["u"],
        "domain_bounds": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        "subdomains": [
            {"name": "D", "bounds": {"x": [-1.0, 1.0], "t": [0.0, 1.0]}},
        ],
        "equations": [
            {
                "name": "burgers",
                "form": "u_t + u*u_x - nu*u_xx",
                "outputs": ["u"],
                "derivatives": [
                    {"name": "u_t", "field": "u", "wrt": ["t"]},
                    {"name": "u_x", "field": "u", "wrt": ["x"]},
                    {"name": "u_xx", "field": "u", "wrt": ["x", "x"]},
                ],
                "parameters": {"nu": 0.01},
            },
        ],
        "conditions": [
            {"subdomain": "D", "kind": "equation", "equation_name": "burgers"},
        ],
    }


def test_catalog_starts_empty(deps):
    """Post-pivot: no built-in seeding. Agents author presets as they go."""
    fn = curator_toolset.tools["list_presets"].function
    assert fn(_Ctx(deps), family="problem") == []
    assert fn(_Ctx(deps), family="model") == []
    assert fn(_Ctx(deps), family="solver") == []


def test_list_presets_rejects_unknown_family(deps):
    fn = curator_toolset.tools["list_presets"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps), family="bogus")


def test_register_and_describe(deps):
    register = curator_toolset.tools["register_preset"].function
    register(
        _Ctx(deps),
        family="problem",
        name="burgers_1d_nu0p01",
        builder_ref="marimo_flow.agents.services.composer:compose_problem",
        description="1D Burgers, viscosity 0.01.",
        spec_json=_burgers_spec(),
        tags=["burgers", "1d"],
    )

    describe = curator_toolset.tools["describe_preset"].function
    rec = describe(_Ctx(deps), family="problem", name="burgers_1d_nu0p01")
    assert rec["name"] == "burgers_1d_nu0p01"
    assert rec["spec_json"]["equations"][0]["name"] == "burgers"
    assert "burgers" in rec["tags"]


def test_register_rejects_bad_builder_ref(deps):
    register = curator_toolset.tools["register_preset"].function
    with pytest.raises(ModelRetry):
        register(
            _Ctx(deps),
            family="problem",
            name="broken",
            builder_ref="not.a.module:Thing",
            description="bad",
        )


def test_register_rejects_builtin_namespace(deps):
    register = curator_toolset.tools["register_preset"].function
    with pytest.raises(ModelRetry):
        register(
            _Ctx(deps),
            family="problem",
            name="builtin.problem.reserved",
            builder_ref="marimo_flow.agents.services.composer:compose_problem",
            description="shouldn't fly",
        )


def test_search_presets_substring_and_tags(deps):
    register = curator_toolset.tools["register_preset"].function
    register(
        _Ctx(deps),
        family="problem",
        name="burgers_1d_nu0p01",
        builder_ref="marimo_flow.agents.services.composer:compose_problem",
        description="1D Burgers, viscosity 0.01.",
        spec_json=_burgers_spec(),
        tags=["burgers", "1d"],
    )
    register(
        _Ctx(deps),
        family="problem",
        name="heat_2d_default",
        builder_ref="marimo_flow.agents.services.composer:compose_problem",
        description="2D heat equation with unit square domain.",
        spec_json={
            "output_variables": ["u"],
            "domain_bounds": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        },
        tags=["heat", "2d"],
    )

    search = curator_toolset.tools["search_presets"].function
    burgers_hits = search(_Ctx(deps), family="problem", query="burgers")
    assert len(burgers_hits) == 1
    assert burgers_hits[0]["name"] == "burgers_1d_nu0p01"

    heat_hits = search(_Ctx(deps), family="problem", tags=["heat"])
    assert len(heat_hits) == 1
    assert heat_hits[0]["name"] == "heat_2d_default"


def test_clone_preset_with_overrides(deps):
    register = curator_toolset.tools["register_preset"].function
    register(
        _Ctx(deps),
        family="problem",
        name="burgers_1d_nu0p01",
        builder_ref="marimo_flow.agents.services.composer:compose_problem",
        description="seed",
        spec_json=_burgers_spec(),
    )
    clone = curator_toolset.tools["clone_preset"].function
    out = clone(
        _Ctx(deps),
        family="problem",
        source_name="burgers_1d_nu0p01",
        new_name="burgers_1d_nu0p05",
        overrides={"equations": [{"parameters": {"nu": 0.05}}]},
        description="Higher viscosity variant.",
        tags=["burgers", "user"],
    )
    assert out == "burgers_1d_nu0p05"

    describe = curator_toolset.tools["describe_preset"].function
    rec = describe(_Ctx(deps), family="problem", name="burgers_1d_nu0p05")
    assert rec["parent_name"] == "burgers_1d_nu0p01"


def test_deprecate_user_preset(deps):
    register = curator_toolset.tools["register_preset"].function
    register(
        _Ctx(deps),
        family="problem",
        name="tmp_preset",
        builder_ref="marimo_flow.agents.services.composer:compose_problem",
        description="ephemeral",
        spec_json={"output_variables": ["u"], "domain_bounds": {"x": [0.0, 1.0]}},
    )
    deprec = curator_toolset.tools["deprecate_preset"].function
    deprec(_Ctx(deps), family="problem", name="tmp_preset")

    listing = curator_toolset.tools["list_presets"].function
    assert listing(_Ctx(deps), family="problem") == []
    everything = listing(_Ctx(deps), family="problem", include_deprecated=True)
    assert any(r["name"] == "tmp_preset" for r in everything)


def test_deprecate_builtin_namespace_forbidden(deps):
    fn = curator_toolset.tools["deprecate_preset"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps), family="problem", name="builtin.problem.reserved")


def test_deprecate_unknown_retries(deps):
    fn = curator_toolset.tools["deprecate_preset"].function
    with pytest.raises(ModelRetry):
        fn(_Ctx(deps), family="problem", name="never_existed")
