"""Solver Manager for creating and configuring PINA solvers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pina.optim import TorchOptimizer
from pina.problem import AbstractProblem
from pina.solver import PINN, SelfAdaptivePINN as SAPINN, SupervisedSolver


class SolverManager:
    """Manager for creating and configuring PINA solvers."""

    @staticmethod
    def create_pinn(
        problem: AbstractProblem,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
        learning_rate: float = 1e-3,
        optimizer_type: type[torch.optim.Optimizer] | None = None,
        **solver_kwargs: Any,
    ) -> PINN:
        """Create a PINN (Physics-Informed Neural Network) solver.
        
        Args:
            problem: PINA problem instance
            model: Neural network model
            optimizer: Optional optimizer instance. If None, creates Adam optimizer
            learning_rate: Learning rate for optimizer (if optimizer not provided)
            optimizer_type: Optimizer class (defaults to torch.optim.Adam)
            **solver_kwargs: Additional arguments for PINN constructor
        
        Returns:
            PINN solver instance
        """
        if optimizer is None:
            if optimizer_type is None:
                optimizer_type = torch.optim.Adam
            optimizer = TorchOptimizer(optimizer_type, lr=learning_rate)

        return PINN(
            problem=problem,
            model=model,
            optimizer=optimizer,
            **solver_kwargs,
        )

    @staticmethod
    def create_supervised_solver(
        problem: AbstractProblem,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
        learning_rate: float = 1e-3,
        optimizer_type: type[torch.optim.Optimizer] | None = None,
        loss: nn.Module | None = None,
        use_lt: bool = False,
        **solver_kwargs: Any,
    ) -> SupervisedSolver:
        """Create a SupervisedSolver for data-driven problems.
        
        Args:
            problem: PINA problem instance (typically SupervisedProblem)
            model: Neural network model
            optimizer: Optional optimizer instance. If None, creates Adam optimizer
            learning_rate: Learning rate for optimizer (if optimizer not provided)
            optimizer_type: Optimizer class (defaults to torch.optim.Adam)
            loss: Loss function (defaults to nn.MSELoss)
            use_lt: Whether to use LabelTensors
            **solver_kwargs: Additional arguments for SupervisedSolver constructor
        
        Returns:
            SupervisedSolver instance
        """
        if optimizer is None:
            if optimizer_type is None:
                optimizer_type = torch.optim.Adam
            optimizer = TorchOptimizer(optimizer_type, lr=learning_rate)

        if loss is None:
            loss = nn.MSELoss()

        return SupervisedSolver(
            problem=problem,
            model=model,
            optimizer=optimizer,
            loss=loss,
            use_lt=use_lt,
            **solver_kwargs,
        )

    @staticmethod
    def create_sapinn(
        problem: AbstractProblem,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | TorchOptimizer | None = None,
        learning_rate: float = 1e-3,
        optimizer_type: type[torch.optim.Optimizer] | None = None,
        **solver_kwargs: Any,
    ) -> SAPINN:
        """Create a SAPINN (Self-Adaptive PINN) solver.
        
        Args:
            problem: PINA problem instance
            model: Neural network model
            optimizer: Optional optimizer instance. If None, creates Adam optimizer
            learning_rate: Learning rate for optimizer (if optimizer not provided)
            optimizer_type: Optimizer class (defaults to torch.optim.Adam)
            **solver_kwargs: Additional arguments for SAPINN constructor
        
        Returns:
            SAPINN solver instance
        """
        if optimizer is None:
            if optimizer_type is None:
                optimizer_type = torch.optim.Adam
            optimizer = TorchOptimizer(optimizer_type, lr=learning_rate)

        return SAPINN(
            problem=problem,
            model=model,
            optimizer=optimizer,
            **solver_kwargs,
        )

