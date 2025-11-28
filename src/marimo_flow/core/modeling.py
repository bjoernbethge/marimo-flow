"""Model and solver factories for the PINA demo."""

from __future__ import annotations

import torch
import torch.nn as nn
from pina.model import FeedForward
from pina.optim import TorchOptimizer
from pina.problem import AbstractProblem
from pina.solver import PINN


def build_model() -> FeedForward:
    """Create a simple fully-connected network for the Poisson problem."""

    return FeedForward(
        input_dimensions=2,
        output_dimensions=1,
        layers=[20, 20, 20],
        func=nn.Tanh,
    )


def build_solver(problem: AbstractProblem, lr: float = 1e-3) -> PINN:
    """Instantiate a solver for the provided problem."""

    model = build_model()
    optimizer = TorchOptimizer(torch.optim.Adam, lr=lr)
    return PINN(problem=problem, model=model, optimizer=optimizer)

