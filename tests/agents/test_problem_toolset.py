"""Tests for the composition-first problem_toolset.

No more ``build_problem(kind, kwargs)`` — the tool is ``compose_problem(spec)``.
We call the toolset function directly with a full ``ProblemSpec`` dict.
"""

from __future__ import annotations

import mlflow
import pytest
from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets.problem import problem_toolset


class _Ctx:
    def __init__(self, deps):
        self.deps = deps


@pytest.fixture
def deps_and_state():
    mlflow.set_experiment("composer-toolset-test")
    with mlflow.start_run() as run:
        state = FlowState(
            mlflow_run_id=run.info.run_id,
            task_spec=TaskSpec(title="Burgers", description="forward PINN"),
        )
        deps = FlowDeps(state=state, provenance_db_path=":memory:")
        yield deps, state


def _burgers_spec_dict() -> dict:
    return {
        "name": "burgers_1d_test",
        "output_variables": ["u"],
        "domain_bounds": {"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        "subdomains": [
            {"name": "left", "bounds": {"x": -1.0, "t": [0.0, 1.0]}},
            {"name": "right", "bounds": {"x": 1.0, "t": [0.0, 1.0]}},
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
            {"subdomain": "left", "kind": "fixed_value", "value": 0.0},
            {"subdomain": "right", "kind": "fixed_value", "value": 0.0},
            {"subdomain": "D", "kind": "equation", "equation_name": "burgers"},
        ],
    }


def test_compose_problem_registers_and_sets_state(deps_and_state):
    deps, state = deps_and_state
    compose = problem_toolset.tools["compose_problem"].function
    uri = compose(_Ctx(deps), spec=_burgers_spec_dict())

    assert uri.endswith("problem_spec.json")
    assert state.problem_artifact_uri == uri
    assert state.problem_spec is not None
    assert state.problem_spec.name == "burgers_1d_test"
    assert len(state.problem_spec.equations) == 1
    assert uri in deps.registry


def test_compose_problem_mirrors_to_provenance(deps_and_state):
    deps, state = deps_and_state
    compose = problem_toolset.tools["compose_problem"].function
    compose(_Ctx(deps), spec=_burgers_spec_dict())

    rows = deps.provenance().query(
        "SELECT payload FROM problem_specs WHERE task_id = ?",
        [state.task_spec.task_id],
    )
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert "burgers_1d_test" in payload


def test_compose_problem_validation_error_becomes_model_retry(deps_and_state):
    deps, _ = deps_and_state
    compose = problem_toolset.tools["compose_problem"].function
    with pytest.raises(ModelRetry):
        compose(_Ctx(deps), spec={"not": "a valid spec"})


def test_compose_problem_unknown_equation_ref_becomes_model_retry(deps_and_state):
    deps, _ = deps_and_state
    compose = problem_toolset.tools["compose_problem"].function
    broken = _burgers_spec_dict()
    broken["conditions"][-1]["equation_name"] = "does_not_exist"
    with pytest.raises(ModelRetry):
        compose(_Ctx(deps), spec=broken)


def test_inspect_problem_returns_summary(deps_and_state):
    deps, _ = deps_and_state
    compose = problem_toolset.tools["compose_problem"].function
    compose(_Ctx(deps), spec=_burgers_spec_dict())

    inspect = problem_toolset.tools["inspect_problem"].function
    summary = inspect(_Ctx(deps))
    assert summary["output_variables"] == ["u"]
    assert "D" in summary["subdomains"]
    assert "burgers" in (eq["name"] for eq in summary["equations"])


def test_inspect_problem_retries_when_empty(deps_and_state):
    deps, _ = deps_and_state
    inspect = problem_toolset.tools["inspect_problem"].function
    with pytest.raises(ModelRetry):
        inspect(_Ctx(deps))


def test_list_input_vars_hint(deps_and_state):
    deps, _ = deps_and_state
    fn = problem_toolset.tools["list_input_vars_hint"].function
    assert fn(_Ctx(deps)) == ["x", "y", "z", "t"]
