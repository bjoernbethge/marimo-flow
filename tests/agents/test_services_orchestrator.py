"""Tests for services/orchestrator.py policy helpers."""

from __future__ import annotations

from marimo_flow.agents.schemas import TaskSpec, ValidationReport
from marimo_flow.agents.services.orchestrator import (
    check_escalation,
    default_experiment_status,
    requires_human_review,
)
from marimo_flow.agents.state import FlowState


def test_check_escalation_returns_none_when_no_report():
    assert check_escalation(FlowState()) is None


def test_check_escalation_returns_none_for_accept():
    state = FlowState(validation_report=ValidationReport(verdict="accept"))
    assert check_escalation(state) is None


def test_check_escalation_returns_none_for_retry():
    state = FlowState(validation_report=ValidationReport(verdict="retry"))
    assert check_escalation(state) is None


def test_check_escalation_flags_escalate():
    state = FlowState(
        validation_report=ValidationReport(
            verdict="escalate", rationale="uncertain metrics"
        )
    )
    msg = check_escalation(state)
    assert msg is not None
    assert "escalate" in msg
    assert "uncertain" in msg


def test_check_escalation_flags_reject():
    state = FlowState(validation_report=ValidationReport(verdict="reject"))
    msg = check_escalation(state)
    assert msg is not None
    assert "reject" in msg


def test_check_escalation_handles_missing_rationale():
    state = FlowState(validation_report=ValidationReport(verdict="reject"))
    msg = check_escalation(state)
    assert "n/a" in msg


def test_requires_human_review_defaults_false():
    assert requires_human_review(None) is False
    task = TaskSpec(title="t", description="d")
    assert requires_human_review(task) is False


def test_requires_human_review_honours_flag():
    task = TaskSpec(title="t", description="d", review_required=True)
    assert requires_human_review(task) is True


def test_default_experiment_status_pending_when_nothing_ran():
    assert default_experiment_status(FlowState()) == "pending"


def test_default_experiment_status_completed_after_training():
    assert default_experiment_status(FlowState(training_run_id="run-1")) == "completed"


def test_default_experiment_status_failed_on_escalation():
    state = FlowState(
        training_run_id="run-1",
        validation_report=ValidationReport(verdict="escalate"),
    )
    assert default_experiment_status(state) == "failed"


def test_default_experiment_status_failed_on_circuit_breaker():
    state = FlowState(route_count=12, max_route_steps=12)
    assert default_experiment_status(state) == "failed"
