"""Tests for FlowState — the serialisable shared state across graph nodes."""

import json
from dataclasses import asdict

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
