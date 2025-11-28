"""Model Factory for creating PINA neural network models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from pina.model import FeedForward


class ModelFactory:
    """Factory for creating neural network models for PINA."""

    @staticmethod
    def create_feedforward(
        input_dimensions: int,
        output_dimensions: int,
        layers: list[int] | None = None,
        activation: nn.Module | type[nn.Module] | None = None,
    ) -> FeedForward:
        """Create a FeedForward neural network model.
        
        Args:
            input_dimensions: Number of input dimensions
            output_dimensions: Number of output dimensions
            layers: List of hidden layer sizes. Defaults to [64, 64, 64]
            activation: Activation function. Defaults to nn.Tanh
        
        Returns:
            FeedForward model instance
        """
        if layers is None:
            layers = [64, 64, 64]
        
        if activation is None:
            activation = nn.Tanh

        return FeedForward(
            input_dimensions=input_dimensions,
            output_dimensions=output_dimensions,
            layers=layers,
            func=activation,
        )

    @staticmethod
    def create_custom_model(
        model_class: type[nn.Module],
        **kwargs: Any,
    ) -> nn.Module:
        """Create a custom PyTorch model.
        
        Args:
            model_class: Custom PyTorch Module class
            **kwargs: Arguments to pass to model constructor
        
        Returns:
            Model instance
        """
        return model_class(**kwargs)

    @staticmethod
    def create_model_for_problem(
        problem: AbstractProblem,
        layers: list[int] | None = None,
        activation: nn.Module | type[nn.Module] | None = None,
    ) -> FeedForward:
        """Create a FeedForward model automatically sized for a problem.
        
        Args:
            problem: PINA problem instance
            layers: List of hidden layer sizes. Defaults to [64, 64, 64]
            activation: Activation function. Defaults to nn.Tanh
        
        Returns:
            FeedForward model instance
        """
        input_dim = len(problem.input_variables)
        output_dim = len(problem.output_variables)
        
        return ModelFactory.create_feedforward(
            input_dimensions=input_dim,
            output_dimensions=output_dim,
            layers=layers,
            activation=activation,
        )

