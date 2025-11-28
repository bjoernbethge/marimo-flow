# PINA Reference - Quickstart & Core Concepts

**Last Updated**: 2025-11-21
**Source Version**: Latest (mathLab/PINA)
**Status**: Current

## What is PINA?

PINA is an open-source Python library for **Scientific Machine Learning (SciML)**. It simplifies development of neural network-based solutions for solving scientific and engineering problems, particularly those involving differential equations and physics-informed constraints.

Built on top of:
- **PyTorch** - Core deep learning framework
- **PyTorch Lightning** - Scalable training abstractions
- **PyTorch Geometric** - Graph neural network support

### Key Strengths

1. **Modular Architecture**: Easily swap, customize, or extend components
2. **Scalable Performance**: Multi-device training with minimal overhead
3. **Flexible Design**: Both high-level abstractions and granular control

### Primary Use Cases

- **Solving Differential Equations** - PDEs, ODEs using Physics-Informed Neural Networks (PINNs)
- **Data-Driven Modeling** - Learn mappings from data with neural networks
- **Neural Operators** - Learn infinite-dimensional function mappings
- **Hybrid Modeling** - Combine neural networks with physics constraints

## Quick Reference

### Installation

```bash
# Basic installation
pip install pina-mathlab

# From source (latest development version)
git clone https://github.com/mathLab/PINA.git
cd PINA
pip install .

# With development/testing extras
pip install "pina-mathlab[extras]"
```

### Verify Installation

```python
import pina
print(pina.__version__)
```

## Core Concepts

### 1. Four-Step Workflow

PINA follows a consistent pipeline for all problems:

**Step 1: Define the Problem**

```python
from pina import Problem
from pina.domain import Domain
import torch

# Create a domain (input space)
domain = Domain()
domain.add("x", 0, 1)      # x from 0 to 1
domain.add("y", 0, 1)      # y from 0 to 1
domain.add("t", 0, 1)      # time from 0 to 1

# Create problem
problem = Problem(domain)
```

**Step 2: Design a Model**

```python
from pina import FeedForward
import torch.nn as nn

# Create a simple neural network
model = FeedForward(
    input_dimensions=3,     # x, y, t
    output_dimensions=1,    # u (solution)
    layers=[64, 64, 64],    # Hidden layer sizes
    activations=nn.Tanh()   # Activation function
)
```

**Step 3: Select a Solver**

```python
from pina.solvers import SupervisedSolver  # For data-driven problems
from pina.solvers import PINNSolver        # For physics-informed problems

# For supervised learning from data
solver = SupervisedSolver(problem=problem, model=model)

# OR for physics-informed (with constraints)
# solver = PINNSolver(problem=problem, model=model)
```

**Step 4: Train**

```python
from pina.trainer import Trainer

trainer = Trainer(
    solver=solver,
    max_epochs=100,
    accelerator="gpu"  # or "cpu"
)

trainer.fit()
```

### 2. Problem Definition

#### Data-Driven Problem

```python
from pina import Problem
from pina.domain import Domain
from pina.operators import Condition

# Define domain
domain = Domain()
domain.add("x", 0, 1)
domain.add("y", 0, 1)

# Create problem
problem = Problem(domain)

# Load training data
import torch
x_train = torch.linspace(0, 1, 100).unsqueeze(-1)
y_train = torch.sin(x_train)  # True values

# Add condition (training data)
problem.add_condition(Condition(
    domain,
    x_train,
    y_train
))

problem
```

#### Physics-Informed Problem (PDE)

```python
from pina import Problem
from pina.domain import Domain
from pina.operators import Condition, Identity
import torch

# Domain
domain = Domain()
domain.add("x", 0, 1)
domain.add("t", 0, 1)

# Problem
problem = Problem(domain)

# Define differential operator (example: heat equation)
# u_t - u_xx = 0
def pde(output, inputs):
    u = output  # Neural network output
    # Compute derivatives (handled by autograd)
    # This is simplified; PINA handles symbolic differentiation
    return u

# Add PDE condition
from pina.operators import Identity
problem.add_condition(Condition(
    domain,
    Identity(),  # The PDE operator
    torch.zeros(1)  # RHS = 0
))

# Add boundary conditions
problem.add_condition(Condition(
    domain.boundary("x==0"),  # Boundary at x=0
    Identity(),
    torch.zeros(1)
))
```

### 3. Models & Architectures

#### FeedForward (MLP)

```python
from pina import FeedForward
import torch.nn as nn

model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    layers=[64, 64, 64],
    activations=nn.Tanh(),
    output_activations=None  # No activation for output
)
```

#### ResNet Architecture

```python
from pina import Resnet

model = Resnet(
    input_dimensions=2,
    output_dimensions=1,
    layers=[64, 64],
    activations=nn.ReLU()
)
```

#### Graph Neural Networks

```python
from pina import GraphNet

model = GraphNet(
    input_dimensions=2,
    output_dimensions=1,
    hidden_dimensions=64,
    num_layers=3
)
```

### 4. Solvers

#### Supervised Solver (Data-Driven Learning)

For learning input-to-output mappings from data:

```python
from pina.solvers import SupervisedSolver
from pina.trainer import Trainer
import torch.optim as optim

solver = SupervisedSolver(
    problem=problem,
    model=model,
    optimizer=optim.Adam(model.parameters(), lr=1e-3),
    loss=torch.nn.MSELoss()
)

trainer = Trainer(solver, max_epochs=50)
trainer.fit()

# Make predictions
x_test = torch.linspace(0, 1, 50).unsqueeze(-1)
predictions = solver(x_test)
```

#### PINN Solver (Physics-Informed)

For solving PDEs with physics constraints:

```python
from pina.solvers import PINNSolver
from pina.trainer import Trainer

solver = PINNSolver(problem=problem, model=model)

trainer = Trainer(solver, max_epochs=100)
trainer.fit()

# Evaluate solution
x_eval = torch.linspace(0, 1, 100).unsqueeze(-1)
solution = solver(x_eval)
```

#### DeepONet Solver (Neural Operators)

For learning infinite-dimensional function mappings:

```python
from pina.solvers import DeepONetSolver
from pina.models import DeepONet

# DeepONet learns operator: input_function -> output_function
solver = DeepONetSolver(problem=problem, model=model)
```

### 5. Training & Evaluation

#### Basic Training

```python
from pina.trainer import Trainer

trainer = Trainer(
    solver=solver,
    max_epochs=100,
    accelerator="gpu",  # "cpu", "gpu", "auto"
    devices=1,
    batch_size=32,
    learning_rate=1e-3
)

# Train
trainer.fit()

# Evaluate
results = trainer.validate()
print(results)
```

#### Checkpointing & Resume

```python
from pina.trainer import Trainer

trainer = Trainer(
    solver=solver,
    max_epochs=100,
    checkpoint_dir="./checkpoints"
)

# Resume from checkpoint
trainer.fit(ckpt_path="./checkpoints/last.ckpt")
```

#### Early Stopping

```python
from pytorch_lightning.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min"
)

trainer = Trainer(solver=solver, callbacks=[early_stop])
```

### 6. Loss Functions & Regularization

#### Custom Loss

```python
import torch.nn as nn

class CustomLoss(nn.Module):
    def forward(self, predictions, targets):
        mse = ((predictions - targets) ** 2).mean()
        l2_reg = (predictions ** 2).mean()
        return mse + 0.01 * l2_reg

solver = PINNSolver(
    problem=problem,
    model=model,
    loss=CustomLoss()
)
```

#### Weighting Different Terms

```python
# In problem definition
# PINNs often balance data fidelity and PDE constraints
solver = PINNSolver(
    problem=problem,
    model=model,
    loss_weights={
        "data": 1.0,      # Data fitting loss weight
        "pde": 0.1        # PDE constraint weight
    }
)
```

## Common Patterns

### Pattern: Solve 1D ODE with Data

Problem: Learn `y(x)` from data points

```python
import torch
from pina import Problem, FeedForward
from pina.domain import Domain
from pina.operators import Condition, Identity
from pina.solvers import SupervisedSolver
from pina.trainer import Trainer

# 1. Define domain
domain = Domain()
domain.add("x", 0, 10)

# 2. Create problem
problem = Problem(domain)

# 3. Generate synthetic data
x_data = torch.linspace(0, 10, 100).unsqueeze(-1)
y_data = torch.sin(x_data)  # y = sin(x)

# 4. Add training data condition
problem.add_condition(Condition(domain, x_data, y_data))

# 5. Create model
model = FeedForward(
    input_dimensions=1,
    output_dimensions=1,
    layers=[64, 64],
    activations=torch.nn.Tanh()
)

# 6. Create solver
solver = SupervisedSolver(problem=problem, model=model)

# 7. Train
trainer = Trainer(solver, max_epochs=100)
trainer.fit()

# 8. Evaluate
x_test = torch.linspace(0, 10, 50).unsqueeze(-1)
predictions = solver(x_test)
```

### Pattern: Solve Heat Equation (PDE)

Problem: Solve 1D heat equation `u_t = u_xx` with initial/boundary conditions

```python
import torch
from pina import Problem, FeedForward
from pina.domain import Domain
from pina.operators import Condition, PDEOperator
from pina.solvers import PINNSolver
from pina.trainer import Trainer

# Domain: x in [0,1], t in [0,1]
domain = Domain()
domain.add("x", 0, 1)
domain.add("t", 0, 1)

# Problem
problem = Problem(domain)

# Define PDE operator (simplified example)
# Heat equation: u_t - u_xx = 0
class HeatEquation(PDEOperator):
    def forward(self, output, inputs):
        # PINA handles automatic differentiation
        u_t = torch.autograd.grad(
            output.sum(), inputs['t'],
            create_graph=True
        )[0]
        # Similar for u_xx (second derivative in x)
        return u_t  # Should equal u_xx

# Add PDE condition
problem.add_condition(Condition(
    domain.sample(1000),
    HeatEquation(),
    torch.zeros(1000, 1)
))

# Add initial condition: u(x, 0) = sin(pi*x)
boundary_t0 = domain.boundary("t==0")
x_vals = boundary_t0.sample(100)
u_initial = torch.sin(torch.pi * x_vals['x'])
problem.add_condition(Condition(boundary_t0, u_initial))

# Add boundary conditions
boundary_x0 = domain.boundary("x==0")
problem.add_condition(Condition(boundary_x0, torch.zeros(100, 1)))

boundary_x1 = domain.boundary("x==1")
problem.add_condition(Condition(boundary_x1, torch.zeros(100, 1)))

# Create and train
model = FeedForward(2, 1, [64, 64], torch.nn.Tanh())
solver = PINNSolver(problem=problem, model=model)
trainer = Trainer(solver, max_epochs=100)
trainer.fit()

# Evaluate
x_eval = torch.linspace(0, 1, 50).unsqueeze(-1)
t_eval = torch.linspace(0, 1, 50).unsqueeze(-1)
X, T = torch.meshgrid(x_eval, t_eval)
inputs = torch.cat([X.flatten().unsqueeze(-1), T.flatten().unsqueeze(-1)], dim=-1)
predictions = solver(inputs).reshape(50, 50)
```

### Pattern: Multi-Output Problem

Problem: Learn multiple outputs from inputs

```python
from pina import Problem, FeedForward
from pina.solvers import SupervisedSolver

# Model with multiple outputs
model = FeedForward(
    input_dimensions=1,
    output_dimensions=3,  # Three outputs: y1, y2, y3
    layers=[64, 64]
)

# Problem with multi-output data
x_data = torch.linspace(0, 1, 100).unsqueeze(-1)
y_data = torch.cat([
    torch.sin(x_data),
    torch.cos(x_data),
    torch.exp(x_data)
], dim=-1)

problem = Problem(domain)
problem.add_condition(Condition(domain, x_data, y_data))

solver = SupervisedSolver(problem=problem, model=model)
```

## Best Practices

### ✅ DO: Use Appropriate Model for Problem Type

```python
# GOOD - FeedForward for simple data-driven
model = FeedForward(input_dimensions=2, output_dimensions=1)

# GOOD - More complex models for PINNs
model = Resnet(input_dimensions=3, output_dimensions=1, layers=[128, 128])
```

### ✅ DO: Normalize Inputs/Outputs

```python
# GOOD - Neural networks work better with normalized data
x_normalized = (x_data - x_data.mean()) / x_data.std()
y_normalized = (y_data - y_data.mean()) / y_data.std()

# Add condition with normalized data
problem.add_condition(Condition(domain, x_normalized, y_normalized))
```

### ✅ DO: Use Appropriate Activations

```python
# GOOD - Tanh or Sine activations work well for PINNs
model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    activations=torch.nn.Tanh()
)

# GOOD - ReLU for data-driven problems
model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    activations=torch.nn.ReLU()
)
```

### ✅ DO: Monitor Training with Callbacks

```python
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    dirpath="./checkpoints",
    monitor="val_loss",
    save_top_k=3,
    mode="min"
)

trainer = Trainer(solver=solver, callbacks=[checkpoint])
```

### ❌ DON'T: Use Too Many Layers for PINNs

```python
# BAD - Too deep networks can struggle with PINNs
model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    layers=[256] * 10  # Too deep
)

# GOOD - Moderate depth typically works better
model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    layers=[64, 64, 64, 64]
)
```

### ✅ DO: Use Multi-Device Training for Large Problems

```python
trainer = Trainer(
    solver=solver,
    accelerator="gpu",
    devices=[0, 1],  # Multiple GPUs
    strategy="ddp"   # Distributed Data Parallel
)
```

## Common Issues & Solutions

### Issue: Loss Not Decreasing

**Problem**: Training loss plateaus or increases.

**Solutions**:
1. Reduce learning rate
2. Use different activation (Tanh better for PINNs)
3. Normalize input/output data
4. Reduce network complexity
5. Increase number of collocation/data points

```python
# Reduce learning rate
solver = PINNSolver(
    problem=problem,
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4)
)
```

### Issue: OOM (Out of Memory)

**Problem**: GPU memory exhausted.

**Solutions**:
1. Reduce batch size
2. Use smaller network
3. Sample fewer collocation points
4. Use mixed precision training

```python
trainer = Trainer(
    solver=solver,
    batch_size=16,  # Reduce batch size
    precision="16-mixed"  # Mixed precision
)
```

### Issue: Poor Generalization

**Problem**: Model overfits to training data.

**Solutions**:
1. Add regularization
2. Use more diverse training data
3. Reduce model complexity
4. Add physics constraints (PINNs)

```python
# Use PINN with physics constraints instead of pure supervised
solver = PINNSolver(
    problem=problem,
    model=model,
    loss_weights={"data": 1.0, "pde": 0.5}
)
```

## API Reference - Quick Lookup

### Domain

```python
domain = Domain()
domain.add("x", 0, 1)        # Add input dimension
domain.add("t", 0, 10)       # Add time dimension
domain.boundary("x==0")       # Get boundary
domain.sample(1000)           # Sample points
```

### Models

```python
from pina import FeedForward, Resnet, GraphNet

FeedForward(input_dimensions, output_dimensions, layers, activations)
Resnet(input_dimensions, output_dimensions, layers, activations)
GraphNet(input_dimensions, output_dimensions, hidden_dimensions)
```

### Solvers

```python
from pina.solvers import SupervisedSolver, PINNSolver, DeepONetSolver

SupervisedSolver(problem, model, optimizer, loss)
PINNSolver(problem, model, loss_weights)
DeepONetSolver(problem, model)
```

### Conditions

```python
from pina.operators import Condition, Identity

Condition(domain, operator, target)
# operator: Identity() for supervised, custom for PDEs
# target: target values or zero for PDE RHS
```

### Trainer

```python
from pina.trainer import Trainer

trainer = Trainer(
    solver=solver,
    max_epochs=100,
    accelerator="gpu",
    batch_size=32,
    checkpoint_dir="./checkpoints"
)

trainer.fit()
trainer.validate()
```

## Additional Resources

- **Official Docs**: https://mathlab.github.io/PINA/
- **GitHub**: https://github.com/mathLab/PINA
- **PyTorch Ecosystem**: Part of PyTorch ecosystem
- **Examples**: https://github.com/mathLab/PINA/tree/main/examples
- **Contact**: pina.mathlab@gmail.com

## Related Technologies

### PyTorch Integration

PINA is built on PyTorch, so you can use any PyTorch features:

```python
import torch
import torch.nn as nn

# Custom PyTorch module
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

# Use in PINA
model = CustomModel()
solver = SupervisedSolver(problem=problem, model=model)
```

### Integration with Polars/Pandas

Load data from files using Polars or Pandas:

```python
import polars as pl
import torch

# Load with Polars
df = pl.read_csv("data.csv")
x_data = torch.from_numpy(df["x"].to_numpy()).float().unsqueeze(-1)
y_data = torch.from_numpy(df["y"].to_numpy()).float().unsqueeze(-1)

# Add to problem
problem.add_condition(Condition(domain, x_data, y_data))
```
