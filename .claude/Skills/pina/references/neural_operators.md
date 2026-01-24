# Neural Operators in PINA

## Overview

Neural operators learn mappings between function spaces, making them ideal for parameterized PDEs and multi-query scenarios.

## Fourier Neural Operator (FNO)

FNO learns operators in Fourier space for efficient operator learning.

```python
from pina.model import FNO
from pina.solver import SupervisedSolver
import torch.nn as nn

# Define lifting and projecting networks
lifting_net = nn.Linear(1, 24)
projecting_net = nn.Linear(24, 1)

# Create FNO
model = FNO(
    lifting_net=lifting_net,
    projecting_net=projecting_net,
    n_modes=8,        # Number of Fourier modes
    dimensions=2,      # 2D problem
    inner_size=24,     # Hidden dimension
    padding=8          # Padding for FFT
)

# Use with SupervisedSolver
solver = SupervisedSolver(problem=problem, model=model, use_lt=False)
```

### FNO Architecture

- **Lifting**: Projects input to higher dimension
- **Fourier Layers**: Operates in frequency domain
- **Projection**: Maps back to output dimension

### When to Use FNO

- Multi-query PDE problems
- Parameterized equations
- Need fast inference
- Regular grid data

## DeepONet

DeepONet uses two networks (branch and trunk) to learn operators.

```python
from pina.model import DeepONet

model = DeepONet(
    trunk_layer_sizes=[10, 20, 30],
    branch_layer_sizes=[5, 10, 15],
    activation='relu'
)
```

### DeepONet Architecture

- **Branch Network**: Encodes input function
- **Trunk Network**: Encodes evaluation points
- **Output**: Inner product of branch and trunk outputs

### When to Use DeepONet

- Irregular geometries
- Point cloud data
- Function-to-function mappings
- Transfer learning scenarios

## Kernel Neural Operator

General framework for operator learning with integral kernels.

```python
from pina.model import KernelNeuralOperator
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class Processor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.kernel = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # Implement integral kernel operation
        return self.kernel(x)

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = KernelNeuralOperator(
    lifting_operator=Encoder(input_dim=2, hidden_dim=64),
    integral_kernels=Processor(hidden_dim=64),
    projection_operator=Decoder(hidden_dim=64, output_dim=1)
)
```

### When to Use Kernel Neural Operator

- Custom operator learning architectures
- Need explicit kernel design
- Research and experimentation

## Complete FNO Example

```python
from pina import Trainer
from pina.model import FNO
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem
import torch
import torch.nn as nn

# Generate training data (e.g., solution snapshots)
n_samples = 1000
input_data = torch.randn(n_samples, 64, 64, 1)  # Initial conditions
output_data = torch.randn(n_samples, 64, 64, 1)  # Solutions

# Create problem
problem = SupervisedProblem(input_data, output_data)

# Define networks
lifting_net = nn.Sequential(
    nn.Linear(1, 32),
    nn.ReLU(),
    nn.Linear(32, 64)
)

projecting_net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

# Create FNO model
fno = FNO(
    lifting_net=lifting_net,
    projecting_net=projecting_net,
    n_modes=16,
    dimensions=2,
    inner_size=64,
    padding=8
)

# Create solver
solver = SupervisedSolver(
    problem=problem,
    model=fno,
    use_lt=False
)

# Train
trainer = Trainer(
    solver=solver,
    max_epochs=100,
    batch_size=32,
    accelerator="gpu"
)
trainer.train()
```

## Best Practices

### Data Preparation

```python
# Normalize data
mean = input_data.mean()
std = input_data.std()
input_data = (input_data - mean) / std

# Use consistent grid sizes
# FNO works best with powers of 2: 64x64, 128x128, 256x256
```

### Hyperparameter Selection

```python
# Start with moderate settings
fno = FNO(
    n_modes=8,      # Increase for more complex functions
    inner_size=32,  # Increase for more capacity
    padding=8       # Typically 8-16
)

# Gradually increase complexity if needed
fno = FNO(
    n_modes=16,
    inner_size=64,
    padding=12
)
```

### Training Tips

- Use large batch sizes (32-128) for stable training
- Start with learning rate ~0.001
- Monitor validation loss for overfitting
- Use data augmentation when possible

## Comparison

| Feature | FNO | DeepONet | Kernel NO |
|---------|-----|----------|-----------|
| Grid Type | Regular | Any | Any |
| Speed | Fast | Medium | Medium |
| Flexibility | Medium | High | Very High |
| Ease of Use | Easy | Medium | Hard |

## Resources

- FNO Paper: "Fourier Neural Operator for Parametric PDEs"
- DeepONet Paper: "Learning nonlinear operators via DeepONet"
- PINA FNO Tutorial: https://github.com/mathLab/PINA/tree/master/tutorials
