"""Tests for FlowState — the serialisable shared state across graph nodes."""

import json
from dataclasses import asdict

from marimo_flow.agents.schemas import (
    AgentDecision,
    ConditionSpec,
    EquationSpec,
    HandoffRecord,
    ProblemSpec,
    SubdomainSpec,
    TaskSpec,
)
from marimo_flow.agents.state import FlowState


def test_default_state_has_empty_history():
    state = FlowState()
    assert state.user_intent is None
    assert state.problem_artifact_uri is None
    assert state.model_artifact_uri is None
    assert state.solver_artifact_uri is None
    assert state.mlflow_run_id is None
    assert state.last_node is None
    assert state.history == {}


def test_default_state_has_empty_typed_fields():
    state = FlowState()
    assert state.task_spec is None
    assert state.problem_spec is None
    assert state.model_spec is None
    assert state.solver_plan is None
    assert state.run_config is None
    assert state.validation_report is None
    assert state.decisions == []
    assert state.handoffs == []


def test_state_is_json_serialisable():
    state = FlowState(
        user_intent="Solve 1D Poisson",
        mlflow_run_id="abc123",
        last_node="problem",
        history={"problem": []},
    )
    payload = json.dumps(asdict(state))
    restored = FlowState(**json.loads(payload))
    assert restored == state


def test_history_per_role_is_independent():
    state = FlowState()
    state.history.setdefault("problem", []).append({"role": "user", "content": "test"})
    state.history.setdefault("solver", []).append({"role": "user", "content": "go"})
    assert len(state.history["problem"]) == 1
    assert len(state.history["solver"]) == 1


def test_to_jsonable_roundtrips_pydantic_fields():
    """to_jsonable() must render Pydantic specs as JSON-friendly dicts."""
    task = TaskSpec(title="Burgers 1D", description="PINN benchmark")
    problem = ProblemSpec(
        name="burgers_1d",
        output_variables=["u"],
        domain_bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]}),
        ],
        equations=[
            EquationSpec(
                name="burgers",
                form="u_t",
                outputs=["u"],
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="burgers"),
        ],
    )
    decision = AgentDecision(
        agent="problem",
        tool="compose_problem",
        summary="Composed Burgers 1D problem.",
        task_id=task.task_id,
    )
    handoff = HandoffRecord(
        from_agent="problem", to_agent="model", reason="Problem ready."
    )
    state = FlowState(
        user_intent="Solve Burgers",
        task_spec=task,
        problem_spec=problem,
        decisions=[decision],
        handoffs=[handoff],
    )
    payload = json.dumps(state.to_jsonable())
    data = json.loads(payload)

    assert data["task_spec"]["title"] == "Burgers 1D"
    assert data["problem_spec"]["name"] == "burgers_1d"
    assert data["problem_spec"]["equations"][0]["name"] == "burgers"
    assert data["decisions"][0]["tool"] == "compose_problem"
    assert data["handoffs"][0]["from_agent"] == "problem"
    assert isinstance(data["task_spec"]["created_at"], str)
