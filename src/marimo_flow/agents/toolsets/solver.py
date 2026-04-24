"""FunctionToolset for the Solver agent.

Thin wrapper on `marimo_flow.core.SolverManager`. Tools build real
`pina.solver.PINN` / SAPINN / ... instances tailored to the problem
and model registered earlier in the workflow.
"""

from __future__ import annotations

import contextlib
from typing import Any

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.toolsets._registry import (
    register_artifact,
    require_state,
    retry_on_value_error,
)
from marimo_flow.agents.toolsets._specs import solver_plan_from
from marimo_flow.core import SolverManager

solver_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="solver")


@solver_toolset.tool
def list_solver_kinds(ctx: RunContext[FlowDeps]) -> list[str]:  # noqa: ARG001
    """List all solver kinds available from SolverManager.

    Returns names usable as `kind` in `build_solver`, e.g. 'pinn',
    'sapinn', 'causalpinn', 'gradientpinn', 'rbapinn', 'supervised'.
    """
    return list(SolverManager.available())


@solver_toolset.tool
def build_solver(
    ctx: RunContext[FlowDeps],
    kind: str,
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Build a PINA solver wrapping the registered problem + model.

    Args:
        kind: one of list_solver_kinds() — e.g. 'pinn', 'causalpinn', ...
        kwargs: solver-specific keyword arguments. Common ones:
          * learning_rate: float (default 1e-3)
          * optimizer_type: torch.optim.Optimizer class (default Adam)
          * eps (causalpinn), eta + gamma (rbapinn)
          * loss: torch.nn loss module (supervised)

    Returns the MLflow artifact URI. The live solver instance is stored
    in `deps.registry[uri]` for the training node.
    """
    state = require_state(ctx.deps)
    if state.problem_artifact_uri is None or state.model_artifact_uri is None:
        raise ModelRetry(
            "build_solver needs both a problem and a model to be registered "
            "first. Hand control back so those agents can run, then retry."
        )
    problem = ctx.deps.registry[state.problem_artifact_uri]
    model = ctx.deps.registry[state.model_artifact_uri]
    kwargs = kwargs or {}
    solver = retry_on_value_error(
        lambda: SolverManager.create(kind, problem=problem, model=model, **kwargs),
        available=SolverManager.available(),
    )
    uri = register_artifact(
        deps=ctx.deps,
        state=state,
        artifact_path="solver",
        filename="solver_spec.json",
        record={"kind": kind, "kwargs": kwargs},
        instance=solver,
    )
    state.solver_artifact_uri = uri
    with contextlib.suppress(Exception):
        plan = solver_plan_from(kind, kwargs)
        state.solver_plan = plan
        task_id = state.task_spec.task_id if state.task_spec else "unknown"
        ctx.deps.provenance().record_solver_plan(task_id, plan)
    return uri
