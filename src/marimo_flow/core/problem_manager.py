"""Problem Manager for creating and managing PINA problems."""

from __future__ import annotations

from typing import Any, Callable

import torch
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.problem import AbstractProblem, SpatialProblem, TimeDependentProblem
from pina.problem.zoo import SupervisedProblem


class ProblemManager:
    """Manager for creating and configuring PINA problems."""

    @staticmethod
    def create_from_dataframe(
        df: Any,
        input_cols: list[str],
        output_cols: list[str],
    ) -> SupervisedProblem:
        """Create a SupervisedProblem from a DataFrame (Polars/Pandas).
        
        Args:
            df: Polars or Pandas DataFrame
            input_cols: List of column names for input features
            output_cols: List of column names for target values
        
        Returns:
            SupervisedProblem instance
        """
        import numpy as np
        import polars as pl
        
        # Convert to numpy first
        if isinstance(df, pl.DataFrame):
            input_arr = df.select(input_cols).to_numpy()
            target_arr = df.select(output_cols).to_numpy()
        else:  # Pandas
            input_arr = df[input_cols].to_numpy()
            target_arr = df[output_cols].to_numpy()
            
        # Convert to torch tensors
        input_tensor = torch.from_numpy(input_arr).float()
        target_tensor = torch.from_numpy(target_arr).float()
        
        return SupervisedProblem(input_=input_tensor, output_=target_tensor)

        """Create a Cartesian domain from variable definitions.
        
        Args:
            variables: Dictionary mapping variable names to [min, max] lists
                      e.g., {"x": [0, 1], "y": [0, 1]}
        
        Returns:
            Configured CartesianDomain object
        """
        return CartesianDomain(variables)

    @staticmethod
    def create_supervised_problem(
        input_data: torch.Tensor,
        target_data: torch.Tensor,
    ) -> SupervisedProblem:
        """Create a supervised learning problem from data.
        
        Args:
            input_data: Input tensor
            target_data: Target tensor
        
        Returns:
            SupervisedProblem instance
        """
        return SupervisedProblem(input_=input_data, output_=target_data)

    @staticmethod
    def create_spatial_problem(
        output_variables: list[str],
        spatial_domain: CartesianDomain,
        domains: dict[str, CartesianDomain] | None = None,
        conditions: dict[str, Condition] | None = None,
    ) -> type[SpatialProblem]:
        """Create a spatial problem class.
        
        Args:
            output_variables: List of output variable names
            spatial_domain: Spatial domain definition
            domains: Optional dictionary of named domains
            conditions: Optional dictionary of conditions
        
        Returns:
            Problem class (not instance)
        """
        domains = domains or {}
        conditions = conditions or {}

        class CustomSpatialProblem(SpatialProblem):
            output_variables = output_variables
            spatial_domain = spatial_domain
            domains = domains
            conditions = conditions

        return CustomSpatialProblem

    @staticmethod
    def create_time_dependent_problem(
        output_variables: list[str],
        spatial_domain: CartesianDomain,
        temporal_domain: CartesianDomain,
        domains: dict[str, CartesianDomain] | None = None,
        conditions: dict[str, Condition] | None = None,
    ) -> type[TimeDependentProblem]:
        """Create a time-dependent problem class.
        
        Args:
            output_variables: List of output variable names
            spatial_domain: Spatial domain definition
            temporal_domain: Temporal domain definition
            domains: Optional dictionary of named domains
            conditions: Optional dictionary of conditions
        
        Returns:
            Problem class (not instance)
        """
        domains = domains or {}
        conditions = conditions or {}

        class CustomTimeDependentProblem(TimeDependentProblem, SpatialProblem):
            output_variables = output_variables
            spatial_domain = spatial_domain
            temporal_domain = temporal_domain
            domains = domains
            conditions = conditions

        return CustomTimeDependentProblem

    @staticmethod
    def create_poisson_problem(
        domain_bounds: dict[str, list[float]] | None = None,
        source_term: Callable | None = None,
    ) -> type[SpatialProblem]:
        """Create a Poisson problem class with configurable domain and source term.
        
        Args:
            domain_bounds: Domain bounds, e.g., {"x": [0, 1], "y": [0, 1]}
                         Defaults to unit square
            source_term: Source term function. Defaults to sin(pi*x)*sin(pi*y)
        
        Returns:
            Problem class (not instance)
        """
        if domain_bounds is None:
            domain_bounds = {"x": [0, 1], "y": [0, 1]}

        spatial_domain = CartesianDomain(domain_bounds)

        if source_term is None:
            def default_source(input_, output_):
                x = input_.extract(["x"])
                y = input_.extract(["y"])
                return -torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
            source_term = default_source

        from pina.operator import laplacian

        def poisson_equation(input_, output_):
            lap_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
            f = source_term(input_, output_)
            return lap_u - f

        x_min, x_max = domain_bounds["x"]
        y_min, y_max = domain_bounds["y"]

        domains = {
            "g1": CartesianDomain({"x": [x_min, x_max], "y": y_max}),
            "g2": CartesianDomain({"x": [x_min, x_max], "y": y_min}),
            "g3": CartesianDomain({"x": x_max, "y": [y_min, y_max]}),
            "g4": CartesianDomain({"x": x_min, "y": [y_min, y_max]}),
            "D": spatial_domain,
        }

        conditions = {
            "g1": Condition(domain="g1", equation=FixedValue(0.0)),
            "g2": Condition(domain="g2", equation=FixedValue(0.0)),
            "g3": Condition(domain="g3", equation=FixedValue(0.0)),
            "g4": Condition(domain="g4", equation=FixedValue(0.0)),
            "D": Condition(domain="D", equation=Equation(poisson_equation)),
        }

        class Poisson(SpatialProblem):
            output_variables = ["u"]
            spatial_domain = spatial_domain
            domains = domains
            conditions = conditions

        return Poisson
