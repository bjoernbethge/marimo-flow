"""FunctionToolset for the Problem agent.

Thin wrapper on `marimo_flow.core.ProblemManager`. Tools build real
`pina.Problem` instances (or subclasses) and stash them in
`deps.registry[uri]`; the URI is also written to
`state.problem_artifact_uri` so downstream nodes can resolve it.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.toolsets._registry import register_artifact, require_state
from marimo_flow.core import ProblemManager

problem_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="problem")


@problem_toolset.tool
def list_problem_kinds(ctx: RunContext[FlowDeps]) -> list[str]:  # noqa: ARG001
    """List all problem kinds available from ProblemManager.

    Returns names usable as `kind` in `build_problem`, e.g. 'poisson',
    'heat', 'wave', 'burgers', 'allen_cahn', 'advection_diffusion',
    'helmholtz', 'spatial', 'time_dependent', 'supervised'.
    """
    return list(ProblemManager.available())


@problem_toolset.tool
def build_problem(
    ctx: RunContext[FlowDeps],
    kind: str,
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Build a PINA Problem and register it.

    Args:
        kind: one of list_problem_kinds() — e.g. 'burgers', 'poisson', ...
        kwargs: template-specific keyword arguments. Common ones:
          * domain_bounds: dict like {"x": [-1, 1], "t": [0, 1]}
          * viscosity (burgers), diffusivity (heat/advection_diffusion),
            epsilon (allen_cahn), wave_speed (wave), wave_number (helmholtz)

    Returns the MLflow artifact URI. The live problem instance is stored
    in `deps.registry[uri]` for downstream nodes.
    """
    state = require_state(ctx.deps)
    kwargs = kwargs or {}
    problem_cls_or_instance = ProblemManager.create(kind, **kwargs)
    # Presets return classes, supervised/from_dataframe return instances —
    # instantiate classes so downstream nodes always see a problem instance.
    problem = (
        problem_cls_or_instance()
        if isinstance(problem_cls_or_instance, type)
        else problem_cls_or_instance
    )
    uri = register_artifact(
        deps=ctx.deps,
        state=state,
        artifact_path="problem",
        filename="problem_spec.json",
        record={"kind": kind, "kwargs": kwargs},
        instance=problem,
    )
    state.problem_artifact_uri = uri
    return uri
