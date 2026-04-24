"""Helpers that build typed Model/Solver/Run specs from raw kwargs.

The Model / Solver / Training toolsets keep their thin ``kind`` + ``kwargs``
shape — ``ModelKind`` and ``SolverKind`` are small, finite enumerations
over PINA's built-in architectures and solvers, which ARE a legitimate
choice space for an agent. (The Problem layer is composition-first and
lives in ``composer.py`` — see ``compose_problem``.)
"""

from __future__ import annotations

import json
from typing import Any

from marimo_flow.agents.schemas import (
    ModelSpec,
    RunConfig,
    SolverPlan,
)


def _json_safe(value: Any) -> bool:
    """True if ``value`` round-trips cleanly through json.dumps."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def _safe_kwargs(kwargs: dict[str, Any], *, drop: frozenset[str]) -> dict[str, Any]:
    """Return the JSON-safe subset of ``kwargs`` minus ``drop`` keys."""
    return {k: v for k, v in kwargs.items() if k not in drop and _json_safe(v)}


def model_spec_from(kind: str, kwargs: dict[str, Any]) -> ModelSpec:
    """Build a ModelSpec from the kwargs passed to ``build_model``."""
    activation = kwargs.get("activation")
    activation_name: str | None
    if isinstance(activation, type):
        activation_name = activation.__name__
    elif isinstance(activation, str):
        activation_name = activation
    else:
        activation_name = None
    layers = kwargs.get("layers")
    layers_list: list[int] | None = (
        [int(x) for x in layers]
        if isinstance(layers, list | tuple)
        and all(isinstance(x, int | float) for x in layers)
        else None
    )
    return ModelSpec(
        kind=kind,  # type: ignore[arg-type]
        layers=layers_list,
        activation=activation_name,
        kwargs=_safe_kwargs(kwargs, drop=frozenset({"layers", "activation"})),
    )


def solver_plan_from(kind: str, kwargs: dict[str, Any]) -> SolverPlan:
    """Build a SolverPlan from the kwargs passed to ``build_solver``."""
    lr = kwargs.get("learning_rate", 1e-3)
    if not isinstance(lr, int | float):
        lr = 1e-3
    optimizer_type = kwargs.get("optimizer_type")
    if isinstance(optimizer_type, type):
        optimizer_name = optimizer_type.__name__
    elif isinstance(optimizer_type, str):
        optimizer_name = optimizer_type
    else:
        optimizer_name = "Adam"
    loss = kwargs.get("loss")
    loss_name: str | None
    if isinstance(loss, type):
        loss_name = loss.__name__
    elif isinstance(loss, str):
        loss_name = loss
    else:
        loss_name = None
    return SolverPlan(
        kind=kind,  # type: ignore[arg-type]
        learning_rate=float(lr),
        optimizer_type=optimizer_name,
        loss=loss_name,
        kwargs=_safe_kwargs(
            kwargs,
            drop=frozenset({"learning_rate", "optimizer_type", "loss"}),
        ),
    )


def run_config_from(
    *,
    max_epochs: int,
    accelerator: str,
    n_points: int,
    sample_mode: str,
) -> RunConfig:
    """Build a RunConfig from the kwargs passed to ``train``."""
    return RunConfig(
        max_epochs=int(max_epochs),
        accelerator=accelerator,  # type: ignore[arg-type]
        n_points=int(n_points),
        sample_mode=sample_mode,  # type: ignore[arg-type]
    )
