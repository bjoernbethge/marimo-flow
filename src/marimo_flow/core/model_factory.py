"""Model creation helper for PINA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
from pina.model import FeedForward

if TYPE_CHECKING:
    from pina.problem import AbstractProblem


def create_model_for_problem(
    problem: AbstractProblem,
    *,
    layers: list[int] | None = None,
    activation: type[nn.Module] | None = None,
) -> FeedForward:
    """Create one feedforward model sized from a problem definition."""
    if problem.input_variables is None or problem.output_variables is None:
        raise ValueError("Problem must define input_variables and output_variables.")
    if layers is None:
        layers = [64, 64, 64]
    if activation is None:
        activation = nn.Tanh

    return FeedForward(
        input_dimensions=len(problem.input_variables),
        output_dimensions=len(problem.output_variables),
        layers=layers,
        func=activation,  # type: ignore[arg-type]
    )
