"""Tests for the typed schemas (SPEC §8) — composition-first."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marimo_flow.agents.schemas import (
    AgentDecision,
    ArtifactRef,
    ConditionSpec,
    DatasetBinding,
    DerivativeSpec,
    EquationSpec,
    ExperimentRecord,
    HandoffRecord,
    ModelSpec,
    ProblemSpec,
    RunConfig,
    SolverPlan,
    SubdomainSpec,
    TaskSpec,
    ValidationReport,
)


def test_artifact_ref_roundtrip():
    ref = ArtifactRef(kind="problem", uri="runs:/abc/problem/problem_spec.json")
    data = ref.model_dump(mode="json")
    restored = ArtifactRef.model_validate(data)
    assert restored == ref


def test_artifact_ref_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        ArtifactRef(kind="bogus", uri="runs:/x")


def test_dataset_binding_defaults():
    b = DatasetBinding(name="train", source="parquet", location="data/train.parquet")
    assert b.input_columns is None


def test_equation_spec_minimal():
    eq = EquationSpec(name="burgers", form="u_t", outputs=["u"])
    assert eq.derivatives == []
    assert eq.parameters == {}


def test_equation_spec_full_roundtrip():
    eq = EquationSpec(
        name="burgers",
        form="u_t + u*u_x - nu*u_xx",
        outputs=["u"],
        derivatives=[
            DerivativeSpec(name="u_t", field="u", wrt=["t"]),
            DerivativeSpec(name="u_x", field="u", wrt=["x"]),
            DerivativeSpec(name="u_xx", field="u", wrt=["x", "x"]),
        ],
        parameters={"nu": 0.01},
    )
    data = eq.model_dump(mode="json")
    assert EquationSpec.model_validate(data) == eq


def test_subdomain_spec_allows_scalar_and_interval():
    sd = SubdomainSpec(name="left", bounds={"x": -1.0, "t": [0.0, 1.0]})
    assert sd.bounds["x"] == -1.0
    assert sd.bounds["t"] == [0.0, 1.0]


def test_condition_spec_fixed_value():
    c = ConditionSpec(subdomain="left", kind="fixed_value", value=0.0)
    assert c.value == 0.0


def test_condition_spec_rejects_unknown_kind():
    with pytest.raises(ValidationError):
        ConditionSpec(
            subdomain="left", kind="whatever", value=0.0
        )  # type: ignore[arg-type]


def test_problem_spec_time_dependent_flag():
    spec = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
    )
    assert spec.time_dependent is True

    stationary = ProblemSpec(
        output_variables=["u"],
        domain_bounds={"x": [0.0, 1.0], "y": [0.0, 1.0]},
    )
    assert stationary.time_dependent is False


def test_problem_spec_burgers_roundtrip():
    spec = ProblemSpec(
        name="burgers_1d",
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
            )
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="burgers"),
        ],
    )
    data = spec.model_dump(mode="json")
    assert ProblemSpec.model_validate(data) == spec


def test_task_spec_fields_are_free_text():
    t = TaskSpec(
        title="t",
        description="d",
        boundary_conditions=["u=0 on left", "periodic in y"],
        initial_conditions=["u(x,0) = sin(pi*x)"],
        material_properties={"viscosity": 0.01},
    )
    assert t.boundary_conditions[0].startswith("u=0")
    assert t.material_properties["viscosity"] == 0.01


def test_task_spec_auto_assigns_ids():
    a = TaskSpec(title="t", description="d")
    b = TaskSpec(title="t", description="d")
    assert a.task_id != b.task_id


def test_model_spec_accepts_known_kinds():
    for kind in ("feedforward", "residual", "fno", "deeponet", "pirate", "walrus"):
        ModelSpec(kind=kind)  # type: ignore[arg-type]


def test_solver_plan_defaults():
    p = SolverPlan(kind="pinn")
    assert p.learning_rate == 1e-3
    assert p.optimizer_type == "Adam"


def test_run_config_defaults():
    c = RunConfig()
    assert c.max_epochs == 1000
    assert c.accelerator == "auto"


def test_run_config_rejects_unknown_accelerator():
    with pytest.raises(ValidationError):
        RunConfig(accelerator="tpu")  # type: ignore[arg-type]


def test_agent_decision_minimal():
    d = AgentDecision(agent="problem", summary="ok")
    assert d.decision_id
    assert d.payload == {}


def test_handoff_record_requires_reason():
    with pytest.raises(ValidationError):
        HandoffRecord(from_agent="problem", to_agent="model")  # type: ignore[call-arg]


def test_validation_report_verdict_enum():
    r = ValidationReport(verdict="retry")
    assert r.verdict == "retry"
    with pytest.raises(ValidationError):
        ValidationReport(verdict="maybe")  # type: ignore[arg-type]


def test_experiment_record_binds_task():
    e = ExperimentRecord(task_id="task-1")
    assert e.status == "pending"
    assert e.finished_at is None
