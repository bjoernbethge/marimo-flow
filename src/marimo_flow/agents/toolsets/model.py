"""FunctionToolset for the Model agent.

Thin wrapper on `marimo_flow.core.ModelManager`. Tools build real
`torch.nn.Module` instances tailored to the problem registered in
`state.problem_artifact_uri` and stash them in `deps.registry[uri]`.
"""

from __future__ import annotations

from typing import Any

from pydantic_ai import FunctionToolset, RunContext

from pydantic_ai import ModelRetry

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.toolsets._registry import (
    register_artifact,
    require_state,
    retry_on_value_error,
)
from marimo_flow.core import ModelManager

model_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="model")


@model_toolset.tool
def list_model_kinds(ctx: RunContext[FlowDeps]) -> list[str]:  # noqa: ARG001
    """List all model kinds available from ModelManager.

    Returns names usable as `kind` in `build_model`, e.g. 'feedforward',
    'residual', 'fno', 'deeponet', 'pirate', 'walrus'.
    """
    return list(ModelManager.available())


@model_toolset.tool
def build_model(
    ctx: RunContext[FlowDeps],
    kind: str,
    kwargs: dict[str, Any] | None = None,
) -> str:
    """Build a neural-network model sized for the registered problem.

    Args:
        kind: one of list_model_kinds() — e.g. 'feedforward', 'fno', ...
        kwargs: architecture-specific keyword arguments. Common ones:
          * layers: list[int] (feedforward, residual, pirate)
          * activation: torch.nn activation class (default Tanh/GELU)
          * n_modes, dimensions, inner_size, n_layers (fno)
          * checkpoint, freeze_backbone (walrus)

    Returns the MLflow artifact URI. The live model instance is stored
    in `deps.registry[uri]` for downstream nodes.
    """
    state = require_state(ctx.deps)
    if state.problem_artifact_uri is None:
        raise ModelRetry(
            "No problem registered yet. Hand control back so the problem "
            "agent can run first, then retry build_model."
        )
    problem = ctx.deps.registry[state.problem_artifact_uri]
    kwargs = kwargs or {}
    model = retry_on_value_error(
        lambda: ModelManager.create(kind, problem=problem, **kwargs),
        available=ModelManager.available(),
    )
    uri = register_artifact(
        deps=ctx.deps,
        state=state,
        artifact_path="model",
        filename="model_spec.json",
        record={"kind": kind, "kwargs": kwargs},
        instance=model,
    )
    state.model_artifact_uri = uri
    return uri
