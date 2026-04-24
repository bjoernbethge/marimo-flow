"""Tests for the Design agent: apply_overrides + ConstraintAggregator + toolset."""

from __future__ import annotations

import pytest

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import (
    ConditionSpec,
    ConstraintSpec,
    DerivativeSpec,
    DesignVariableSpec,
    EquationSpec,
    ProblemSpec,
    SubdomainSpec,
)
from marimo_flow.agents.services.design import (
    ConstraintAggregator,
    apply_design_overrides,
)
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets.design import design_toolset


def _burgers_spec() -> ProblemSpec:
    return ProblemSpec(
        name="burgers_design",
        output_variables=["u"],
        domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="burgers",
                form="u_t + u*u_x - nu*u_xx",
                outputs=["u"],
                derivatives=[
                    DerivativeSpec(name="u_t", field="u", wrt=["t"]),
                    DerivativeSpec(name="u_x", field="u", wrt=["x"]),
                    DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
                ],
                parameters={"nu": 0.01},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="burgers"),
        ],
    )


def test_apply_design_overrides_replaces_parameter():
    spec = _burgers_spec()
    dvs = [
        DesignVariableSpec(
            name="viscosity",
            parameter_path="equations[0].parameters.nu",
            low=0.001,
            high=0.1,
        )
    ]
    updated = apply_design_overrides(spec, dvs, {"viscosity": 0.05})
    assert updated.equations[0].parameters["nu"] == 0.05
    # Original untouched.
    assert spec.equations[0].parameters["nu"] == 0.01


def test_apply_design_overrides_preserves_spec_on_missing_values():
    spec = _burgers_spec()
    dvs = [
        DesignVariableSpec(
            name="viscosity",
            parameter_path="equations[0].parameters.nu",
            low=0.001,
            high=0.1,
        )
    ]
    updated = apply_design_overrides(spec, dvs, values={})
    assert updated.equations[0].parameters["nu"] == 0.01


def test_constraint_aggregator_penalty_on_violation():
    cspec = ConstraintSpec(
        name="drag_cap",
        expression="u",
        op="<=",
        rhs=0.5,
        aggregation="max",
        penalty_weight=2.0,
    )
    agg = ConstraintAggregator([cspec], design_variable_names=[])
    residuals, penalty = agg.evaluate_penalty(
        field_samples={"u": [0.3, 0.7, 0.9]},
        design_values={},
    )
    assert residuals["drag_cap"] == pytest.approx(0.9)
    # gap = 0.9 - 0.5 = 0.4 ; penalty = 2.0 * 0.4² = 0.32
    assert penalty == pytest.approx(0.32)


def test_constraint_aggregator_no_violation_zero_penalty():
    cspec = ConstraintSpec(
        name="under_cap",
        expression="u",
        op="<=",
        rhs=1.0,
        aggregation="max",
    )
    agg = ConstraintAggregator([cspec], design_variable_names=[])
    _, penalty = agg.evaluate_penalty(
        field_samples={"u": [0.3, 0.7, 0.9]},
        design_values={},
    )
    assert penalty == 0.0


def test_constraint_aggregator_augmented_lagrangian_updates_multipliers():
    cspec = ConstraintSpec(
        name="drag_cap",
        expression="u",
        op="<=",
        rhs=0.5,
        aggregation="max",
        penalty_weight=1.0,
    )
    agg = ConstraintAggregator([cspec], design_variable_names=[])
    residuals, _ = agg.evaluate_augmented_lagrangian(
        field_samples={"u": [0.8]}, design_values={}
    )
    agg.update_multipliers(residuals)
    # gap = 0.3; λ-bump = 2·1.0·0.3 = 0.6
    assert agg.multipliers["drag_cap"] == pytest.approx(0.6)


def test_design_toolset_build_optimization_plan_roundtrips():
    fn = design_toolset.tools["build_optimization_plan"].function
    plan = fn(
        object(),
        name="minimize_drag",
        objective_expression="u",
        design_variables=[
            {
                "name": "viscosity",
                "parameter_path": "equations[0].parameters.nu",
                "low": 0.001,
                "high": 0.1,
            }
        ],
        constraints=[
            {
                "name": "cap",
                "expression": "u",
                "op": "<=",
                "rhs": 1.0,
            }
        ],
    )
    assert plan["name"] == "minimize_drag"
    assert plan["design_variables"][0]["name"] == "viscosity"
    assert plan["method"] == "optuna_tpe"


def test_design_toolset_apply_overrides_serialises():
    class _Ctx:
        deps = FlowDeps(state=FlowState(), provenance_db_path=":memory:")

    fn = design_toolset.tools["apply_overrides"].function
    spec_dict = _burgers_spec().model_dump()
    result = fn(
        _Ctx(),
        spec=spec_dict,
        design_variables=[
            {
                "name": "viscosity",
                "parameter_path": "equations[0].parameters.nu",
                "low": 0.001,
                "high": 0.1,
            }
        ],
        values={"viscosity": 0.07},
    )
    assert result["equations"][0]["parameters"]["nu"] == 0.07
