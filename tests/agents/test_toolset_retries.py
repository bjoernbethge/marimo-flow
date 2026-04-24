"""Tool-side error conversion to ModelRetry.

The sub-agent toolsets wrap manager calls with ``retry_on_value_error`` so
that a bad ``kind`` becomes a ``ModelRetry`` (LLM-correctable) instead of a
``ValueError`` that crashes the graph. We verify the wrapper directly —
pydantic-ai's own retry plumbing is tested upstream.
"""

from __future__ import annotations

import pytest
from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState
from marimo_flow.agents.toolsets._registry import retry_on_value_error


def test_retry_on_value_error_includes_available():
    with pytest.raises(ModelRetry) as excinfo:
        retry_on_value_error(
            lambda: (_ for _ in ()).throw(ValueError("Unknown kind 'brgers'")),
            available=["burgers", "heat", "poisson"],
        )
    msg = str(excinfo.value)
    assert "brgers" in msg
    assert "burgers" in msg


def test_retry_on_value_error_passthrough():
    assert retry_on_value_error(lambda: 42) == 42


def test_retry_on_value_error_honours_hint():
    with pytest.raises(ModelRetry) as excinfo:
        retry_on_value_error(
            lambda: (_ for _ in ()).throw(ValueError("bad kwargs")),
            hint="Pass 'layers' as list[int].",
        )
    assert "layers" in str(excinfo.value)


def test_build_problem_converts_bad_kind_to_retry():
    from marimo_flow.agents.toolsets.problem import problem_toolset

    deps = FlowDeps(state=FlowState())

    class Ctx:
        def __init__(self, d):
            self.deps = d

    build = problem_toolset.tools["build_problem"].function
    with pytest.raises(ModelRetry) as excinfo:
        build(Ctx(deps), kind="notreal", kwargs=None)
    assert "Available" in str(excinfo.value)


def test_build_model_retries_when_no_problem():
    from marimo_flow.agents.toolsets.model import model_toolset

    deps = FlowDeps(state=FlowState())

    class Ctx:
        def __init__(self, d):
            self.deps = d

    build = model_toolset.tools["build_model"].function
    with pytest.raises(ModelRetry):
        build(Ctx(deps), kind="feedforward", kwargs=None)


def test_build_solver_retries_when_problem_or_model_missing():
    from marimo_flow.agents.toolsets.solver import solver_toolset

    deps = FlowDeps(state=FlowState())

    class Ctx:
        def __init__(self, d):
            self.deps = d

    build = solver_toolset.tools["build_solver"].function
    with pytest.raises(ModelRetry):
        build(Ctx(deps), kind="pinn", kwargs=None)
