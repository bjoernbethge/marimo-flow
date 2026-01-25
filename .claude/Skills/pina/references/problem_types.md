# Problem Types in PINA

## Overview

PINA supports various problem types for different PDE scenarios.

## Poisson Equation (2D PDE)

```python
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.operator import laplacian
import torch

def poisson_equation(input_, output_):
    force_term = (
        torch.sin(input_.extract(["x"]) * torch.pi) *
        torch.sin(input_.extract(["y"]) * torch.pi)
    )
    laplacian_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return laplacian_u - force_term

class Poisson(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-2, 2], "y": [-2, 2]})

    domains = {
        "border": CartesianDomain({"x": [-2, 2], "y": 2}) |  # Top
                  CartesianDomain({"x": [-2, 2], "y": -2}) |  # Bottom
                  CartesianDomain({"x": 2, "y": [-2, 2]}) |  # Right
                  CartesianDomain({"x": -2, "y": [-2, 2]}),  # Left
        "interior": CartesianDomain({"x": [-2, 2], "y": [-2, 2]})
    }

    conditions = {
        "border": Condition(domain="border", equation=FixedValue(0.0)),
        "interior": Condition(domain="interior", equation=Equation(poisson_equation))
    }

problem = Poisson()
```

## Time-Dependent Wave Equation

```python
from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operator import grad

def wave_equation(input_, output_):
    u_t = grad(output_, input_, components=["u"], d=["t"])
    u_tt = grad(u_t, input_, components=["dudt"], d=["t"])
    nabla_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return nabla_u - u_tt

def initial_condition(input_, output_):
    u_expected = (
        torch.sin(torch.pi * input_.extract(["x"])) *
        torch.sin(torch.pi * input_.extract(["y"]))
    )
    return output_.extract(["u"]) - u_expected

class Wave(TimeDependentProblem, SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        # Spatial boundaries
        "g1": CartesianDomain({"x": 1, "y": [0, 1], "t": [0, 1]}),
        "g2": CartesianDomain({"x": 0, "y": [0, 1], "t": [0, 1]}),
        "g3": CartesianDomain({"x": [0, 1], "y": 0, "t": [0, 1]}),
        "g4": CartesianDomain({"x": [0, 1], "y": 1, "t": [0, 1]}),
        # Initial condition
        "initial": CartesianDomain({"x": [0, 1], "y": [0, 1], "t": 0}),
        # Physics domain
        "D": CartesianDomain({"x": [0, 1], "y": [0, 1], "t": [0, 1]})
    }

    conditions = {
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "g3": Condition(domain="g3", equation=FixedValue(0.0)),
        "g4": Condition(domain="g4", equation=FixedValue(0.0)),
        "initial": Condition(domain="initial", equation=Equation(initial_condition)),
        "D": Condition(domain="D", equation=Equation(wave_equation))
    }

problem = Wave()
```

## Inverse Problems

```python
from pina.problem import InverseProblem, SpatialProblem

class PoissonInverse(SpatialProblem, InverseProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-2, 2], "y": [-2, 2]})
    unknown_parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})

    # Define domains...
    domains = {
        "border": CartesianDomain({"x": [-2, 2], "y": 2}) |
                  CartesianDomain({"x": [-2, 2], "y": -2}) |
                  CartesianDomain({"x": 2, "y": [-2, 2]}) |
                  CartesianDomain({"x": -2, "y": [-2, 2]}),
        "interior": CartesianDomain({"x": [-2, 2], "y": [-2, 2]})
    }

    conditions = {
        # Physics conditions
        "border": Condition(domain="border", equation=FixedValue(0.0)),
        "interior": Condition(domain="interior", equation=Equation(laplace_equation)),
        # Data condition for inverse problem
        "data": Condition(input=data_input, target=data_output)
    }
```

## Custom Equations

### Navier-Stokes System

```python
from pina.equation import SystemEquation
from pina.operator import grad

class NavierStokes(SystemEquation):
    def __init__(self, nu):
        self.nu = nu  # Viscosity

    def residual(self, input_, output_):
        # Extract velocity and pressure
        u = output_.extract(["u"])
        v = output_.extract(["v"])
        p = output_.extract(["p"])

        # Compute gradients
        u_x = grad(u, input_, d=["x"])
        u_y = grad(u, input_, d=["y"])
        v_x = grad(v, input_, d=["x"])
        v_y = grad(v, input_, d=["y"])
        p_x = grad(p, input_, d=["x"])
        p_y = grad(p, input_, d=["y"])

        # Second derivatives
        u_xx = grad(u_x, input_, d=["x"])
        u_yy = grad(u_y, input_, d=["y"])
        v_xx = grad(v_x, input_, d=["x"])
        v_yy = grad(v_y, input_, d=["y"])

        # Navier-Stokes equations
        momentum_x = u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)
        momentum_y = u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)
        continuity = u_x + v_y

        return [momentum_x, momentum_y, continuity]
```

## Common Patterns

### 1D Boundary Value Problem

```python
from pina.operator import grad

# Problem: -u'' = f(x), u(0) = u(1) = 0
def ode_equation(input_, output_):
    u = output_.extract(["u"])
    u_x = grad(u, input_, d=["x"])
    u_xx = grad(u_x, input_, d=["x"])
    f = torch.sin(torch.pi * input_.extract(["x"]))
    return -u_xx - f

class BVP(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})
    # Define conditions...
```

### 2D Diffusion Equation

```python
from pina.operator import laplacian

# Problem: du/dt = D * nabla^2(u)
def diffusion_equation(input_, output_):
    u_t = grad(output_, input_, components=["u"], d=["t"])
    laplacian_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    D = 0.1  # Diffusion coefficient
    return u_t - D * laplacian_u
```

### Supervised Learning Pattern

```python
from pina.problem.zoo import SupervisedProblem
from pina.solver import SupervisedSolver

# Data-driven regression
x_train = torch.linspace(0, 2*torch.pi, 100).view(-1, 1)
y_train = torch.sin(x_train)

problem = SupervisedProblem(x_train, y_train)
model = FeedForward(1, 1, layers=[32, 32])
solver = SupervisedSolver(problem, model, use_lt=False)

trainer = Trainer(solver, max_epochs=500, batch_size=16)
trainer.train()
```
