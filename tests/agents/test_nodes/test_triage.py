"""Tests for TriageNode — turns user_intent into TaskSpec.

Uses the fast-path (pre-set ``task_spec`` on state) for the no-LLM
case, and a TestModel with custom TaskSpec output for the LLM path.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.triage import TriageNode
from marimo_flow.agents.schemas import TaskSpec
from marimo_flow.agents.state import FlowState


async def test_triage_fast_paths_when_task_spec_already_set():
    """If caller provides a TaskSpec, triage returns RouteNode without LLM."""
    task = TaskSpec(title="Prebuilt", description="no triage needed")
    state = FlowState(user_intent="irrelevant", task_spec=task)
    # model_override=None — if the fast path worked, the LLM is never called.
    node = TriageNode(model_override=None)
    ctx = GraphRunContext(state=state, deps=FlowDeps(provenance_db_path=":memory:"))
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "triage"
    assert state.task_spec is task  # not replaced


async def test_triage_builds_task_spec_from_intent():
    test_model = TestModel(
        custom_output_args={
            "title": "Solve Burgers 1D",
            "description": "Forward PINN benchmark.",
            "problem_kind": "forward",
            "equation_family": "burgers",
        }
    )
    state = FlowState(user_intent="solve burgers 1d")
    deps = FlowDeps(provenance_db_path=":memory:")
    node = TriageNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)

    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.task_spec is not None
    assert state.task_spec.title == "Solve Burgers 1D"
    assert state.task_spec.equation_family == "burgers"
    assert state.task_spec.problem_kind == "forward"
    assert state.last_node == "triage"


async def test_triage_records_decision_in_state():
    test_model = TestModel(
        custom_output_args={
            "title": "Heat 2D",
            "description": "Forward PINN.",
            "problem_kind": "forward",
        }
    )
    state = FlowState(user_intent="heat 2d")
    deps = FlowDeps(provenance_db_path=":memory:")
    node = TriageNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)

    await node.run(ctx)
    assert len(state.decisions) == 1
    decision = state.decisions[0]
    assert decision.agent == "triage"
    assert decision.task_id == state.task_spec.task_id
    assert "Heat 2D" in decision.summary


async def test_triage_mirrors_task_to_provenance():
    test_model = TestModel(
        custom_output_args={
            "title": "Poisson 2D",
            "description": "Forward PINN.",
        }
    )
    state = FlowState(user_intent="solve poisson 2d")
    deps = FlowDeps(provenance_db_path=":memory:")
    node = TriageNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)

    await node.run(ctx)
    rows = deps.provenance().query(
        "SELECT title FROM tasks WHERE task_id = ?",
        [state.task_spec.task_id],
    )
    assert len(rows) == 1
    assert rows[0]["title"] == "Poisson 2D"
