"""Problem setup helpers for PINA demos."""

from __future__ import annotations

import torch
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.operator import laplacian
from pina.problem import SpatialProblem


class PoissonProblem(SpatialProblem):
    """Poisson equation on the unit square with Dirichlet boundaries."""

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})

    domains = {
        "g1": CartesianDomain({"x": [0, 1], "y": 1}),
        "g2": CartesianDomain({"x": [0, 1], "y": 0}),
        "g3": CartesianDomain({"x": 1, "y": [0, 1]}),
        "g4": CartesianDomain({"x": 0, "y": [0, 1]}),
        "D": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
    }

    def poisson_equation(input_, output_):
        lap_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
        x = input_.extract(["x"])
        y = input_.extract(["y"])
        f = -torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
        return lap_u - f

    conditions = {
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "g3": Condition(domain="g3", equation=FixedValue(0.0)),
        "g4": Condition(domain="g4", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(poisson_equation)),
    }


def build_problem() -> PoissonProblem:
    """Return a configured Poisson problem."""

    return PoissonProblem()

