# Custom Models in PINA

## Overview

PINA supports various custom model architectures for different PDE scenarios.

## Hard Constraints

Enforce boundary conditions exactly by building them into the network architecture.

```python
import torch
import torch.nn as nn

class HardConstraintModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim)
        )

    def forward(self, x):
        # Hard constraint: u(0, y) = u(1, y) = u(x, 0) = u(x, 1) = 0
        hard_constraint = (
            x.extract(["x"]) * (1 - x.extract(["x"])) *
            x.extract(["y"]) * (1 - x.extract(["y"]))
        )
        return hard_constraint * self.layers(x)
```

### Benefits of Hard Constraints

- Boundary conditions satisfied exactly
- Reduces loss complexity
- Improves convergence
- No penalty weighting needed for boundaries

### Example: 1D Boundary Conditions

```python
class HardBC1D(nn.Module):
    def forward(self, x):
        # Boundary condition: u(0) = 0, u(1) = 0
        bc_term = x.extract(["x"]) * (1 - x.extract(["x"]))
        return bc_term * self.nn(x)
```

## Fourier Feature Embedding

For multi-scale problems with varying frequencies.

```python
from pina.model.block import FourierFeatureEmbedding
from pina.model import FeedForward
import torch

class MultiscaleFourierNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding1 = FourierFeatureEmbedding(
            input_dimension=1,
            output_dimension=100,
            sigma=1   # Low frequency
        )
        self.embedding2 = FourierFeatureEmbedding(
            input_dimension=1,
            output_dimension=100,
            sigma=10  # High frequency
        )
        self.layers = FeedForward(
            input_dimensions=200,
            output_dimensions=1,
            layers=[100, 100]
        )

    def forward(self, x):
        e1 = self.embedding1(x)
        e2 = self.embedding2(x)
        return self.layers(torch.cat([e1, e2], dim=-1))
```

### When to Use Fourier Features

- Multi-scale problems
- High-frequency solutions
- Spectral bias issues
- Complex oscillatory behavior

### Multiple Scales Example

```python
class TripleScaleFourierNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Low, medium, high frequency
        self.embed_low = FourierFeatureEmbedding(input_dim, 50, sigma=1)
        self.embed_mid = FourierFeatureEmbedding(input_dim, 50, sigma=5)
        self.embed_high = FourierFeatureEmbedding(input_dim, 50, sigma=20)

        self.net = FeedForward(
            input_dimensions=150,
            output_dimensions=output_dim,
            layers=[128, 128]
        )

    def forward(self, x):
        e_low = self.embed_low(x)
        e_mid = self.embed_mid(x)
        e_high = self.embed_high(x)
        combined = torch.cat([e_low, e_mid, e_high], dim=-1)
        return self.net(combined)
```

## Periodic Boundary Embedding

For periodic problems and domains.

```python
from pina.model.block import PeriodicBoundaryEmbedding

model = torch.nn.Sequential(
    PeriodicBoundaryEmbedding(input_dimension=1, periods=2),
    FeedForward(
        input_dimensions=3,  # 3 * input_dimension
        output_dimensions=1,
        layers=[64, 64]
    )
)
```

### Periodic Embedding Details

The embedding transforms input `x` to `[sin(2πx/L), cos(2πx/L), x]` where `L` is the period.

### Example: Periodic 2D Problem

```python
class Periodic2DNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.periodic_x = PeriodicBoundaryEmbedding(
            input_dimension=1,
            periods=1  # Period in x
        )
        self.periodic_y = PeriodicBoundaryEmbedding(
            input_dimension=1,
            periods=1  # Period in y
        )
        self.net = FeedForward(
            input_dimensions=6,  # 3 for x + 3 for y
            output_dimensions=1,
            layers=[64, 64]
        )

    def forward(self, x):
        x_embed = self.periodic_x(x.extract(['x']))
        y_embed = self.periodic_y(x.extract(['y']))
        combined = torch.cat([x_embed, y_embed], dim=-1)
        return self.net(combined)
```

## POD-NN (Proper Orthogonal Decomposition)

Reduced order modeling with POD basis.

```python
from pina.model.block import PODBlock
from pina.model import FeedForward
import torch.nn as nn

class PODNN(nn.Module):
    def __init__(self, pod_rank, layers, func):
        super().__init__()
        self.pod = PODBlock(pod_rank)
        self.nn = FeedForward(
            input_dimensions=1,
            output_dimensions=pod_rank,
            layers=layers,
            func=func
        )

    def forward(self, x):
        coefficients = self.nn(x)
        return self.pod.expand(coefficients)

    def fit_pod(self, snapshot_data):
        """Fit POD basis to snapshot data.

        Args:
            snapshot_data: Tensor of shape (n_snapshots, spatial_dim)
        """
        self.pod.fit(snapshot_data)

# Usage
model = PODNN(pod_rank=20, layers=[10, 10, 10], func=torch.nn.Tanh)
model.fit_pod(training_snapshots)
```

### POD-NN Workflow

1. Collect high-fidelity solution snapshots
2. Perform POD to extract basis functions
3. Train network to predict POD coefficients
4. Reconstruct solution using learned coefficients

### Complete POD-NN Example

```python
# Generate snapshots (e.g., from FEM solver)
snapshots = torch.randn(1000, 10000)  # 1000 snapshots, 10000 spatial points

# Create and fit POD-NN
model = PODNN(pod_rank=50, layers=[20, 20, 20], func=torch.nn.Tanh)
model.fit_pod(snapshots)

# Use with PINN
from pina.solver import PINN

solver = PINN(problem=problem, model=model)
trainer = Trainer(solver, max_epochs=1000)
trainer.train()
```

## Graph Neural Networks

For problems on irregular geometries and graphs.

```python
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=256):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)
```

### GNN with PyG Datasets

```python
from torch_geometric.datasets import QM9
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem

dataset = QM9(root='./data')
problem = SupervisedProblem(dataset[0], dataset[0].y)
solver = SupervisedSolver(
    problem=problem,
    model=GNN(in_features=11, out_features=19),
    use_lt=False
)
```

## Custom PyTorch Models

Any PyTorch model can be used with PINA.

```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, output_dim)
        )

    def forward(self, x):
        return self.layers(x)

model = CustomModel(input_dim=2, output_dim=1)
```

## Best Practices

### Choose the Right Model

```python
# Simple problems: FeedForward
model = FeedForward(input_dimensions=2, output_dimensions=1, layers=[64, 64])

# Multi-scale: Fourier Features
model = MultiscaleFourierNet()

# Periodic: Periodic Embedding
model = PeriodicBoundaryEmbedding(...)

# ROM: POD-NN
model = PODNN(...)

# Irregular geometry: GNN
model = GNN(...)
```

### Network Architecture Guidelines

```python
# Start simple
model = FeedForward(input_dimensions=2, output_dimensions=1, layers=[20, 20])

# Gradually increase complexity
model = FeedForward(input_dimensions=2, output_dimensions=1, layers=[64, 64, 64])

# Very deep for complex problems
model = FeedForward(input_dimensions=2, output_dimensions=1,
                    layers=[100, 100, 100, 100])
```

### Activation Functions

```python
# Default: Tanh (smooth, bounded)
model = FeedForward(..., func=torch.nn.Tanh)

# Alternative: Softplus (smooth, positive)
model = FeedForward(..., func=torch.nn.Softplus)

# Alternative: SiLU/Swish (modern, effective)
model = FeedForward(..., func=torch.nn.SiLU)
```
