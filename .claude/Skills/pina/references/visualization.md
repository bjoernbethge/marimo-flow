# Visualization in PINA

## Overview

Effective visualization is crucial for understanding PDE solutions and diagnosing training issues.

## Plot Solutions

### 1D Solutions

```python
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def plot_solution_1d(solver, n_points=256):
    """Plot 1D solution comparison."""
    # Sample points
    pts = solver.problem.spatial_domain.sample(n_points, "grid")

    # Get predictions
    predicted = solver(pts).extract("u").detach()
    true = solver.problem.solution(pts).detach()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # True solution
    axes[0].plot(pts.extract(["x"]), true, label="True", color="blue")
    axes[0].set_title("True Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].legend()
    axes[0].grid(True)

    # PINN solution
    axes[1].plot(pts.extract(["x"]), predicted, label="PINN", color="green")
    axes[1].set_title("PINN Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u")
    axes[1].legend()
    axes[1].grid(True)

    # Absolute error
    diff = torch.abs(true - predicted)
    axes[2].plot(pts.extract(["x"]), diff, label="Error", color="red")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("|u_true - u_pred|")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

# Usage
plot_solution_1d(solver)
```

### 2D Solutions

```python
import numpy as np

@torch.no_grad()
def plot_solution_2d(solver, n_points=100):
    """Plot 2D solution as heatmap."""
    # Sample grid
    pts = solver.problem.spatial_domain.sample(n_points, "grid")

    # Get predictions
    predicted = solver(pts).extract("u").detach()
    true = solver.problem.solution(pts).detach()

    # Reshape to 2D grid
    x = pts.extract(["x"]).reshape(n_points, n_points)
    y = pts.extract(["y"]).reshape(n_points, n_points)
    pred_grid = predicted.reshape(n_points, n_points)
    true_grid = true.reshape(n_points, n_points)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # True solution
    im0 = axes[0].contourf(x, y, true_grid, levels=20, cmap='viridis')
    axes[0].set_title("True Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    # PINN solution
    im1 = axes[1].contourf(x, y, pred_grid, levels=20, cmap='viridis')
    axes[1].set_title("PINN Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    # Error
    error = torch.abs(true_grid - pred_grid)
    im2 = axes[2].contourf(x, y, error, levels=20, cmap='hot')
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

# Usage
plot_solution_2d(solver)
```

### Time-Dependent Solutions

```python
from matplotlib.animation import FuncAnimation

@torch.no_grad()
def animate_solution(solver, n_spatial=50, n_time=100):
    """Animate time-dependent solution."""
    # Create spatial-temporal grid
    x = torch.linspace(0, 1, n_spatial)
    y = torch.linspace(0, 1, n_spatial)
    t_vals = torch.linspace(0, 1, n_time)

    fig, ax = plt.subplots(figsize=(8, 6))

    def update(frame):
        ax.clear()
        t = t_vals[frame]

        # Create grid at time t
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        tt = torch.full_like(xx, t)

        # Stack into input
        pts = torch.stack([xx.flatten(), yy.flatten(), tt.flatten()], dim=1)
        pts_lt = solver.problem.input_pts_to_labels(pts)

        # Predict
        u = solver(pts_lt).extract("u").reshape(n_spatial, n_spatial)

        # Plot
        im = ax.contourf(xx, yy, u, levels=20, cmap='viridis')
        ax.set_title(f"Solution at t = {t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return [im]

    anim = FuncAnimation(fig, update, frames=n_time, interval=50, blit=False)
    plt.show()

    return anim

# Usage
anim = animate_solution(solver)
```

## Plot Training Metrics

### Basic Loss Plot

```python
def plot_losses(trainer):
    """Plot training losses."""
    trainer_metrics = trainer.callbacks[0].metrics

    plt.figure(figsize=(10, 6))
    for metric, loss in trainer_metrics.items():
        plt.plot(range(len(loss)), loss, label=metric)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("Training Losses")
    plt.tight_layout()
    plt.show()

# Usage
plot_losses(trainer)
```

### Multi-Condition Loss Plot

```python
def plot_condition_losses(trainer):
    """Plot individual condition losses."""
    metrics = trainer.callbacks[0].metrics

    # Separate total loss from condition losses
    total_loss = metrics.get('train_loss', [])
    condition_losses = {k: v for k, v in metrics.items() if k != 'train_loss'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Total loss
    axes[0].plot(range(len(total_loss)), total_loss, label="Total Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Total Loss")
    axes[0].grid(True)
    axes[0].legend()

    # Condition losses
    for metric, loss in condition_losses.items():
        axes[1].plot(range(len(loss)), loss, label=metric)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].set_title("Condition Losses")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

# Usage
plot_condition_losses(trainer)
```

## Error Analysis

### Pointwise Error Distribution

```python
@torch.no_grad()
def plot_error_distribution(solver, n_points=1000):
    """Plot error distribution histogram."""
    # Sample points
    pts = solver.problem.spatial_domain.sample(n_points, "random")

    # Compute error
    predicted = solver(pts).extract("u").detach()
    true = solver.problem.solution(pts).detach()
    error = (predicted - true).numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(error, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].grid(True)

    # Q-Q plot
    from scipy import stats
    stats.probplot(error, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

# Usage
plot_error_distribution(solver)
```

### Spatial Error Map

```python
@torch.no_grad()
def plot_spatial_error(solver, n_points=100):
    """Plot spatial distribution of errors."""
    # Sample grid
    pts = solver.problem.spatial_domain.sample(n_points, "grid")

    # Compute error
    predicted = solver(pts).extract("u").detach()
    true = solver.problem.solution(pts).detach()
    error = torch.abs(true - predicted)

    # Reshape
    x = pts.extract(["x"]).reshape(n_points, n_points)
    y = pts.extract(["y"]).reshape(n_points, n_points)
    error_grid = error.reshape(n_points, n_points)

    # Plot
    plt.figure(figsize=(10, 8))
    im = plt.contourf(x, y, error_grid, levels=20, cmap='hot')
    plt.colorbar(im, label="Absolute Error")
    plt.title("Spatial Error Distribution")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

# Usage
plot_spatial_error(solver)
```

## Diagnostic Plots

### Residual Analysis

```python
@torch.no_grad()
def plot_residuals(solver, n_points=256):
    """Plot PDE residuals."""
    # Sample points
    pts = solver.problem.spatial_domain.sample(n_points, "grid")
    pts.requires_grad = True

    # Compute residuals
    output = solver(pts)
    residuals = {}

    for condition_name, condition in solver.problem.conditions.items():
        if hasattr(condition, 'equation'):
            res = condition.equation(pts, output)
            residuals[condition_name] = res.detach().abs()

    # Plot
    fig, axes = plt.subplots(1, len(residuals), figsize=(5*len(residuals), 5))
    if len(residuals) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, residuals.items()):
        x = pts.extract(["x"]).detach()
        ax.plot(x, res, label=name)
        ax.set_xlabel("x")
        ax.set_ylabel("|Residual|")
        ax.set_title(f"Residual: {name}")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

# Usage
plot_residuals(solver)
```

### Convergence Analysis

```python
def plot_convergence(solvers_history):
    """Plot convergence for different network sizes or discretizations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, metrics in solvers_history.items():
        losses = metrics['train_loss']
        ax.plot(range(len(losses)), losses, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)
    ax.set_title("Convergence Comparison")
    plt.tight_layout()
    plt.show()

# Usage example
# solvers_history = {
#     "n=100": trainer1.callbacks[0].metrics,
#     "n=500": trainer2.callbacks[0].metrics,
#     "n=1000": trainer3.callbacks[0].metrics
# }
# plot_convergence(solvers_history)
```

## Best Practices

### Always Visualize

```python
# After training, always plot:
# 1. Solution comparison
plot_solution_1d(solver)  # or plot_solution_2d

# 2. Training losses
plot_losses(trainer)

# 3. Error distribution
plot_error_distribution(solver)

# 4. Spatial error map (for 2D)
plot_spatial_error(solver)
```

### Save Plots

```python
# Save figures for documentation
fig = plot_solution_1d(solver)
plt.savefig('solution_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Interactive Plots

```python
# For Jupyter notebooks
%matplotlib widget

# For interactive 3D plots
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
plt.show()
```

## Complete Visualization Pipeline

```python
def full_visualization(solver, trainer):
    """Complete visualization suite."""
    print("=" * 50)
    print("PINA Solution Visualization")
    print("=" * 50)

    # 1. Solution plots
    print("\n1. Plotting solutions...")
    plot_solution_1d(solver)

    # 2. Training metrics
    print("\n2. Plotting training metrics...")
    plot_losses(trainer)

    # 3. Error analysis
    print("\n3. Analyzing errors...")
    plot_error_distribution(solver)

    # 4. Compute statistics
    with torch.no_grad():
        pts = solver.problem.spatial_domain.sample(1000, "random")
        pred = solver(pts).extract("u")
        true = solver.problem.solution(pts)
        error = torch.abs(pred - true)

        print("\n" + "=" * 50)
        print("Error Statistics")
        print("=" * 50)
        print(f"Mean Absolute Error: {error.mean():.6e}")
        print(f"Max Absolute Error:  {error.max():.6e}")
        print(f"Std Absolute Error:  {error.std():.6e}")
        print(f"Relative Error:      {(error / true.abs()).mean():.6e}")

# Usage
full_visualization(solver, trainer)
```
