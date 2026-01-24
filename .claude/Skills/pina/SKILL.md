---
name: pina
description: Physics-Informed Neural Networks with PINA - solve PDEs, inverse problems, and operator learning with PyTorch
triggers:
  - pina
  - physics-informed neural networks
  - pinns
  - pde solver
  - operator learning
  - fourier neural operator
  - deeponet
  - neural operator
  - pde residual
allowed_tools:
  - Read
  - Write
  - Edit
  - Bash
  - mcp__plugin_context7_context7__resolve-library-id
  - mcp__plugin_context7_context7__query-docs
  - mcp__mlflow__search_traces
  - mcp__mlflow__get_trace
  - mcp__mlflow__log_feedback
---

# PINA Development Skill

Expert guidance for Physics-Informed Neural Networks (PINNs) and Scientific Machine Learning with PINA.

## What is PINA?

PINA (Physics-Informed Neural networks for Advanced modeling) is a PyTorch-based library for solving partial differential equations (PDEs) using neural networks. It combines:

- **Physics-Informed Neural Networks (PINNs)**: Solve forward and inverse PDE problems
- **Neural Operators**: FNO, DeepONet for operator learning
- **Data-Driven Modeling**: Supervised learning with physics constraints
- **Reduced Order Modeling**: POD-NN for efficient simulations

Built on: **PyTorch**, **PyTorch Lightning**, **PyTorch Geometric**

## Core Workflow

Every PINA project follows these 4 steps:

```python
from pina import Trainer
from pina.problem import SpatialProblem
from pina.solver import PINN
from pina.model import FeedForward

# Step 1: Define Problem
problem = MyProblem()
problem.discretise_domain(n=100, mode="grid")

# Step 2: Design Model
model = FeedForward(input_dimensions=1, output_dimensions=1, layers=[64, 64])

# Step 3: Define Solver
solver = PINN(problem, model)

# Step 4: Train
trainer = Trainer(solver, max_epochs=1000, accelerator='gpu')
trainer.train()
```

## Simple ODE Example

```python
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.operator import grad
import torch

def ode_equation(input_, output_):
    """PDE residual: du/dx - u = 0"""
    u_x = grad(output_, input_, components=["u"], d=["x"])
    u = output_.extract(["u"])
    return u_x - u

class SimpleODE(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 1]})

    domains = {
        "x0": CartesianDomain({"x": 0.0}),  # Boundary
        "D": CartesianDomain({"x": [0, 1]})  # Interior
    }

    conditions = {
        "bound_cond": Condition(domain="x0", equation=FixedValue(1.0)),
        "phys_cond": Condition(domain="D", equation=Equation(ode_equation))
    }

    def solution(self, pts):
        """Analytical solution for validation."""
        return torch.exp(pts.extract(["x"]))

problem = SimpleODE()
```

## Models

### FeedForward Networks

```python
from pina.model import FeedForward

# Basic network
model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    layers=[64, 64, 64],  # Hidden layers
    func=torch.nn.Tanh   # Activation function
)

# Alternative activations
model = FeedForward(
    input_dimensions=1,
    output_dimensions=1,
    layers=[100, 100, 100],
    func=torch.nn.Softplus  # or torch.nn.SiLU
)
```

See [Custom Models Reference](references/custom_models.md) for advanced architectures including:
- Hard constraints
- Fourier feature embeddings
- Periodic boundary embeddings
- POD-NN
- Graph neural networks

See [Neural Operators Reference](references/neural_operators.md) for operator learning with FNO, DeepONet, and more.

## PINN Solver

```python
from pina.solver import PINN
from pina.optim import TorchOptimizer
import torch

pinn = PINN(
    problem=problem,
    model=model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=0.001)
)
```

See [Advanced Solvers Reference](references/advanced_solvers.md) for:
- Self-Adaptive PINN (SAPINN)
- Supervised Solver
- Custom solvers
- Training strategies

## Training

### Basic Training

```python
from pina import Trainer
from pina.callbacks import MetricTracker

# Discretize domain
problem.discretise_domain(n=1000, mode="random", domains="all")

# Create trainer
trainer = Trainer(
    solver=pinn,
    max_epochs=1500,
    accelerator="cpu",  # or "gpu"
    enable_model_summary=False,
    callbacks=[MetricTracker()]
)

# Train
trainer.train()
```

### Training Configuration

```python
trainer = Trainer(
    solver=solver,
    max_epochs=1000,
    accelerator="gpu",
    devices=1,
    batch_size=32,
    gradient_clip_val=0.1,  # Gradient clipping
    callbacks=[MetricTracker()]
)
trainer.train()
```

### Testing

```python
# Test the model
test_results = trainer.test()

# Manual evaluation
with torch.no_grad():
    test_pts = problem.spatial_domain.sample(100, "grid")
    prediction = solver(test_pts)
    true_solution = problem.solution(test_pts)
    error = torch.abs(prediction - true_solution)
```

## Domain Discretization

### Sampling Modes

```python
# Grid sampling (uniform points)
problem.discretise_domain(n=100, mode="grid", domains=["D", "x0"])

# Random sampling (Monte Carlo)
problem.discretise_domain(n=1000, mode="random", domains="all")

# Latin Hypercube Sampling
problem.discretise_domain(n=500, mode="lh", domains=["D"])

# Manual sampling
pts = problem.spatial_domain.sample(256, "grid", variables="x")
```

**Best Practice**: Start with grid for testing, use random/LH for training with more points.

## Visualization

```python
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_solution(solver, n_points=256):
    # Sample points
    pts = solver.problem.spatial_domain.sample(n_points, "grid")

    # Get predictions
    predicted = solver(pts).extract("u").detach()
    true = solver.problem.solution(pts).detach()

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(pts.extract(["x"]), true, label="True", color="blue")
    axes[0].set_title("True Solution")
    axes[0].legend()

    axes[1].plot(pts.extract(["x"]), predicted, label="PINN", color="green")
    axes[1].set_title("PINN Solution")
    axes[1].legend()

    diff = torch.abs(true - predicted)
    axes[2].plot(pts.extract(["x"]), diff, label="Error", color="red")
    axes[2].set_title("Absolute Error")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
```

See [Visualization Reference](references/visualization.md) for comprehensive plotting techniques.

## Best Practices

### 1. Start Simple

```python
# Begin with small network
model = FeedForward(input_dimensions=2, output_dimensions=1, layers=[20, 20])

# Gradually increase complexity
model = FeedForward(input_dimensions=2, output_dimensions=1, layers=[64, 64, 64])
```

### 2. Monitor Losses

```python
from pina.callbacks import MetricTracker

trainer = Trainer(
    solver=pinn,
    max_epochs=1000,
    callbacks=[MetricTracker(["train_loss", "bound_cond_loss", "phys_cond_loss"])]
)
```

### 3. Two-Phase Training

```python
# Phase 1: Rough solution (high LR)
pinn = PINN(problem, model, optimizer=TorchOptimizer(torch.optim.Adam, lr=0.01))
trainer = Trainer(pinn, max_epochs=500)
trainer.train()

# Phase 2: Refinement (low LR)
pinn.optimizer.param_groups[0]['lr'] = 0.001
trainer = Trainer(pinn, max_epochs=1500)
trainer.train()
```

## MLflow Integration

Track PINA experiments with MLflow for reproducibility and comparison:

```python
import mlflow
from pina import Trainer
from pina.solver import PINN

# Set experiment
mlflow.set_experiment("pina-poisson-solver")

with mlflow.start_run(run_name="baseline"):
    # Log hyperparameters
    mlflow.log_params({
        "layers": [64, 64, 64],
        "activation": "Tanh",
        "learning_rate": 0.001,
        "n_points": 1000,
        "epochs": 1500
    })

    # Setup and train
    problem.discretise_domain(n=1000, mode="random")
    trainer = Trainer(solver, max_epochs=1500)
    trainer.train()

    # Log final metrics
    mlflow.log_metric("final_loss", trainer.callback_metrics["train_loss"])

    # Log model
    mlflow.pytorch.log_model(solver.model, "pinn_model")
```

### Marimo Dashboard Integration

Create interactive PINA dashboards with marimo:

```python
import marimo as mo
from pina.solver import PINN

# UI controls for hyperparameters
layers = mo.ui.slider(1, 5, value=3, label="Hidden Layers")
neurons = mo.ui.slider(16, 128, value=64, step=16, label="Neurons/Layer")
lr = mo.ui.number(value=0.001, start=0.0001, stop=0.1, label="Learning Rate")

# Train button
train_btn = mo.ui.run_button(label="Train PINN")

# In another cell: run training when button clicked
if train_btn.value:
    model = FeedForward(
        input_dimensions=2,
        output_dimensions=1,
        layers=[neurons.value] * layers.value
    )
    # ... train and visualize
```

### Using context7 for Documentation

Query up-to-date PINA documentation directly:

```
# context7 Library ID (no resolve needed):
# - /mathlab/pina (official docs, 2345 snippets)

# Example: query-docs("/mathlab/pina", "FeedForward model parameters")
```

## When to Use This Skill

✅ **Use PINA when:**
- Solving PDEs with neural networks
- Need to incorporate physics constraints
- Working with inverse problems
- Building neural operators (FNO, DeepONet)
- Reduced order modeling
- Scientific ML research

❌ **Don't use PINA when:**
- Pure data-driven tasks (use standard PyTorch)
- Not dealing with differential equations
- Need classical numerical solvers (FEM, FVM)

## Reference Documentation

Detailed documentation organized by topic:

- **[Problem Types](references/problem_types.md)**: ODE, Poisson, Wave, Inverse problems, custom equations
- **[Neural Operators](references/neural_operators.md)**: FNO, DeepONet, Kernel Neural Operator
- **[Custom Models](references/custom_models.md)**: Hard constraints, Fourier features, periodic embeddings, POD-NN, GNNs
- **[Advanced Solvers](references/advanced_solvers.md)**: SAPINN, supervised solver, custom solvers, training strategies
- **[Visualization](references/visualization.md)**: Plotting techniques, error analysis, animations

## Complete Examples

Ready-to-run example scripts:

- **[Poisson 2D](examples/poisson_2d.py)**: Complete 2D Poisson equation solver with visualization
- **[Wave Equation](examples/wave_equation.py)**: Time-dependent wave equation with animations
- **[FNO Example](examples/fno_example.py)**: Fourier Neural Operator for operator learning
- **[Inverse Problem](examples/inverse_problem.py)**: Learn unknown parameters from data

## Resources

- **Documentation**: https://mathlab.github.io/PINA/
- **GitHub**: https://github.com/mathLab/PINA
- **Paper**: https://joss.theoj.org/papers/10.21105/joss.04813
- **Tutorials**: https://github.com/mathLab/PINA/tree/master/tutorials
