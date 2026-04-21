"""Solver manager for creating PINA solvers via a single API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from pina.optim import TorchOptimizer
from pina.problem import AbstractProblem
from pina.solver import PINN, SupervisedSolver
from pina.solver import SelfAdaptivePINN as SAPINN


def _resolve_optimizer(
    *,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None,
    learning_rate: float,
    optimizer_type: type[torch.optim.Optimizer] | None,
) -> tuple[torch.optim.Optimizer | TorchOptimizer, type[torch.optim.Optimizer]]:
    resolved_type = optimizer_type or torch.optim.Adam
    if optimizer is not None:
        return optimizer, resolved_type
    return TorchOptimizer(resolved_type, lr=learning_rate), resolved_type


def _create_pinn(
    problem: AbstractProblem,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
    learning_rate: float = 1e-3,
    optimizer_type: type[torch.optim.Optimizer] | None = None,
    **solver_kwargs: Any,
) -> PINN:
    """Create a PINN solver."""
    resolved_optimizer, _ = _resolve_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
    )
    return PINN(
        problem=problem,
        model=model,
        optimizer=resolved_optimizer,
        **solver_kwargs,
    )


def _create_supervised_solver(
    problem: AbstractProblem,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
    learning_rate: float = 1e-3,
    optimizer_type: type[torch.optim.Optimizer] | None = None,
    loss: nn.Module | None = None,
    use_lt: bool = False,
    **solver_kwargs: Any,
) -> SupervisedSolver:
    """Create a supervised solver."""
    resolved_optimizer, _ = _resolve_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
    )
    return SupervisedSolver(
        problem=problem,
        model=model,
        optimizer=resolved_optimizer,
        loss=loss or nn.MSELoss(),
        use_lt=use_lt,
        **solver_kwargs,
    )


def _create_sapinn(
    problem: AbstractProblem,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
    learning_rate: float = 1e-3,
    optimizer_type: type[torch.optim.Optimizer] | None = None,
    **solver_kwargs: Any,
) -> SAPINN:
    """Create a self-adaptive PINN solver."""
    optimizer_model, resolved_type = _resolve_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimizer_type=optimizer_type,
    )
    return SAPINN(
        problem=problem,
        model=model,
        optimizer_model=optimizer_model,
        optimizer_weights=TorchOptimizer(resolved_type, lr=learning_rate),
        **solver_kwargs,
    )


class SolverManager:
    """Single entry point for creating solver instances."""

    _REGISTRY: dict[str, Callable[..., PINN | SAPINN | SupervisedSolver]] = {
        "pinn": _create_pinn,
        "sapinn": _create_sapinn,
        "supervised": _create_supervised_solver,
    }

    @classmethod
    def available(cls) -> tuple[str, ...]:
        """Return supported solver kinds."""
        return tuple(sorted(cls._REGISTRY))

    @classmethod
    def create(
        cls,
        kind: str,
        *,
        problem: AbstractProblem,
        model: nn.Module,
        **kwargs: Any,
    ) -> PINN | SAPINN | SupervisedSolver:
        """Create a solver by kind (`pinn`, `sapinn`, `supervised`)."""
        key = kind.strip().lower()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown solver kind '{kind}'. Available: {', '.join(cls.available())}"
            )
        return cls._REGISTRY[key](problem=problem, model=model, **kwargs)
