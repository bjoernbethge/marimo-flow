# Advanced Solvers in PINA

## Overview

PINA provides multiple solver types for different scenarios.

## PINN Solver

Standard Physics-Informed Neural Network solver.

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

### PINN Configuration

```python
from pina.loss import LpLoss

pinn = PINN(
    problem=problem,
    model=model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=0.001, weight_decay=1e-5),
    loss=LpLoss(p=2),  # L2 loss (default)
)
```

## Self-Adaptive PINN (SAPINN)

Automatically adjusts loss weights during training.

```python
from pina.solver import SelfAdaptivePINN as SAPINN
from pina.model import FeedForward

sapinn = SAPINN(
    problem=problem,
    model=FeedForward(
        input_dimensions=1,
        output_dimensions=1,
        layers=[100, 100, 100]
    )
)
```

### Why Use SAPINN

- Automatic balancing of multiple loss terms
- Reduces manual hyperparameter tuning
- Better convergence for multi-condition problems
- Handles stiff equations better

### SAPINN Example

```python
from pina import Trainer

# Problem with multiple conditions
problem = MyComplexProblem()
problem.discretise_domain(n=1000, mode="random")

# SAPINN handles weighting automatically
sapinn = SAPINN(
    problem=problem,
    model=FeedForward(input_dimensions=2, output_dimensions=1, layers=[64, 64, 64])
)

trainer = Trainer(sapinn, max_epochs=2000, accelerator="gpu")
trainer.train()
```

## Supervised Solver

For pure data-driven problems without physics constraints.

```python
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem
import torch

# Create data
x_train = torch.rand((100, 1))
y_train = x_train.pow(3)

# Define problem
problem = SupervisedProblem(x_train, y_train)

# Create solver
solver = SupervisedSolver(
    problem=problem,
    model=model,
    use_lt=False  # Don't use LabelTensors
)
```

### Supervised Learning Example

```python
from pina import Trainer

# Generate regression data
x_train = torch.linspace(0, 2*torch.pi, 100).view(-1, 1)
y_train = torch.sin(x_train)

problem = SupervisedProblem(x_train, y_train)
model = FeedForward(1, 1, layers=[32, 32])
solver = SupervisedSolver(problem, model, use_lt=False)

trainer = Trainer(solver, max_epochs=500, batch_size=16)
trainer.train()
```

## Custom Solvers

Create your own solver for specialized needs.

```python
from pina.solver import SupervisedSolverInterface, SingleSolverInterface

class MyCustomSolver(SupervisedSolverInterface, SingleSolverInterface):
    def __init__(self, problem, model, loss=None, optimizer=None,
                 scheduler=None, weighting=None, use_lt=True):
        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt
        )

    def loss_data(self, input, target):
        network_output = self.forward(input)
        return self.loss(network_output, target)

    def optimization_cycle(self, batch):
        condition_loss = {}
        for condition_name, data in batch:
            condition_loss[condition_name] = self.loss_data(
                input=data["input"],
                target=data["target"]
            )
        return condition_loss
```

### Custom Solver Example

```python
class WeightedPINN(SupervisedSolverInterface, SingleSolverInterface):
    def __init__(self, problem, model, weights):
        super().__init__(
            model=model,
            problem=problem
        )
        self.weights = weights  # Custom weights for conditions

    def optimization_cycle(self, batch):
        condition_loss = {}
        for condition_name, data in batch:
            loss = self.loss_data(
                input=data["input"],
                target=data["target"]
            )
            # Apply custom weighting
            condition_loss[condition_name] = self.weights[condition_name] * loss
        return condition_loss

# Usage
weights = {"boundary": 10.0, "physics": 1.0, "initial": 5.0}
solver = WeightedPINN(problem, model, weights)
```

## Training Strategies

### Two-Phase Training

```python
# Phase 1: Rough solution (low epochs, high LR)
pinn = PINN(problem, model, optimizer=TorchOptimizer(torch.optim.Adam, lr=0.01))
trainer = Trainer(pinn, max_epochs=500)
trainer.train()

# Phase 2: Refinement (more epochs, lower LR)
pinn.optimizer.param_groups[0]['lr'] = 0.001
trainer = Trainer(pinn, max_epochs=1500)
trainer.train()
```

### Curriculum Learning

```python
# Start with coarse discretization
problem.discretise_domain(n=100, mode="grid")
trainer = Trainer(solver, max_epochs=500)
trainer.train()

# Increase resolution
problem.discretise_domain(n=500, mode="random")
trainer = Trainer(solver, max_epochs=500)
trainer.train()

# Final high resolution
problem.discretise_domain(n=2000, mode="random")
trainer = Trainer(solver, max_epochs=1000)
trainer.train()
```

### Stochastic Weight Averaging

```python
from lightning.pytorch.callbacks import StochasticWeightAveraging
from pina.callbacks import MetricTracker

trainer = Trainer(
    solver=solver,
    max_epochs=500,
    callbacks=[
        MetricTracker(),
        StochasticWeightAveraging(swa_lrs=0.005)
    ],
    accelerator="cpu"
)
trainer.train()
```

### Gradient Clipping

```python
trainer = Trainer(
    solver=solver,
    max_epochs=1000,
    gradient_clip_val=0.1,  # Clip gradients
    gradient_clip_algorithm="norm"
)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer=solver.optimizer,
    mode='min',
    factor=0.5,
    patience=100
)

pinn = PINN(
    problem=problem,
    model=model,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=0.001),
    scheduler=scheduler
)
```

## Training Configuration

### Basic Configuration

```python
from pina import Trainer
from pina.callbacks import MetricTracker

trainer = Trainer(
    solver=solver,
    max_epochs=1000,
    accelerator="cpu",  # or "gpu"
    enable_model_summary=False,
    callbacks=[MetricTracker()]
)

trainer.train()
```

### Advanced Configuration

```python
trainer = Trainer(
    solver=solver,
    max_epochs=1000,
    accelerator="gpu",
    devices=1,
    logger=True,  # Enable logging
    callbacks=[MetricTracker()],
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    batch_size=32,
    enable_model_summary=True,
    gradient_clip_val=0.1,  # Gradient clipping
    log_every_n_steps=50
)

trainer.train()
```

### Testing and Evaluation

```python
# Test the model
test_results = trainer.test()

# Manual evaluation
with torch.no_grad():
    test_pts = problem.spatial_domain.sample(100, "grid")
    prediction = solver(test_pts)
    true_solution = problem.solution(test_pts)
    error = torch.abs(prediction - true_solution)

    print(f"Mean absolute error: {error.mean():.6f}")
    print(f"Max absolute error: {error.max():.6f}")
```

## Loss Monitoring

Always track losses to diagnose training issues.

```python
from pina.callbacks import MetricTracker
import matplotlib.pyplot as plt

# Train with metric tracking
trainer = Trainer(
    solver=pinn,
    max_epochs=1000,
    callbacks=[MetricTracker(["train_loss", "bound_cond_loss", "phys_cond_loss"])]
)
trainer.train()

# Plot losses
trainer_metrics = trainer.callbacks[0].metrics

for metric, loss in trainer_metrics.items():
    plt.plot(range(len(loss)), loss, label=metric)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.show()
```

## Best Practices

### Domain Discretization

```python
# Start with grid for testing
problem.discretise_domain(n=100, mode="grid", domains="all")

# Use random/LH for training
problem.discretise_domain(n=1000, mode="lh", domains="all")

# Increase points gradually
problem.discretise_domain(n=5000, mode="random", domains="all")
```

### Optimizer Selection

```python
# Adam: Default choice, works well most of the time
optimizer = TorchOptimizer(torch.optim.Adam, lr=0.001)

# LBFGS: Better for small problems, requires more memory
optimizer = TorchOptimizer(torch.optim.LBFGS, lr=1.0)

# AdamW: Better generalization with weight decay
optimizer = TorchOptimizer(torch.optim.AdamW, lr=0.001, weight_decay=1e-4)
```

### When to Use Each Solver

| Solver | Use Case |
|--------|----------|
| PINN | Standard PDE problems |
| SAPINN | Multiple conditions, hard to balance |
| SupervisedSolver | Pure data-driven, no physics |
| Custom | Specialized requirements |
