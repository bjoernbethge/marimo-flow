"""
Complete 2D Poisson Equation Solver with PINA

Solves: ∇²u = -sin(πx)sin(πy) on [-2, 2] × [-2, 2]
Boundary: u = 0 on all boundaries
"""

import torch
from pina import Trainer
from pina.problem import SpatialProblem
from pina.domain import CartesianDomain
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.operator import laplacian
from pina.model import FeedForward
from pina.solver import PINN
from pina.callbacks import MetricTracker
import matplotlib.pyplot as plt


def poisson_equation(input_, output_):
    """PDE residual: ∇²u - f = 0"""
    force_term = (
        torch.sin(input_.extract(["x"]) * torch.pi) *
        torch.sin(input_.extract(["y"]) * torch.pi)
    )
    laplacian_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return laplacian_u - force_term


class Poisson2D(SpatialProblem):
    """2D Poisson problem definition."""
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


@torch.no_grad()
def plot_solution(solver, n_points=100):
    """Visualize 2D solution."""
    # Sample grid
    pts = solver.problem.spatial_domain.sample(n_points, "grid")

    # Get predictions
    predicted = solver(pts).extract("u").detach()

    # Reshape to 2D
    x = pts.extract(["x"]).reshape(n_points, n_points)
    y = pts.extract(["y"]).reshape(n_points, n_points)
    u_grid = predicted.reshape(n_points, n_points)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Solution
    im0 = axes[0].contourf(x, y, u_grid, levels=20, cmap='viridis')
    axes[0].set_title("PINN Solution")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    # 3D view
    from mpl_toolkits.mplot3d import Axes3D
    ax = fig.add_subplot(122, projection='3d')
    ax.plot_surface(x, y, u_grid, cmap='viridis')
    ax.set_title("3D View")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")

    plt.tight_layout()
    plt.show()


def main():
    """Main execution."""
    print("=" * 60)
    print("2D Poisson Equation Solver with PINA")
    print("=" * 60)

    # 1. Define problem
    print("\n1. Defining problem...")
    problem = Poisson2D()

    # 2. Discretize domain
    print("2. Discretizing domain...")
    problem.discretise_domain(n=1000, mode="random", domains="all")

    # 3. Create model
    print("3. Creating neural network...")
    model = FeedForward(
        input_dimensions=2,
        output_dimensions=1,
        layers=[64, 64, 64],
        func=torch.nn.Tanh
    )

    # 4. Create PINN solver
    print("4. Creating PINN solver...")
    solver = PINN(problem=problem, model=model)

    # 5. Train
    print("5. Training...")
    trainer = Trainer(
        solver=solver,
        max_epochs=1500,
        accelerator="cpu",  # Change to "gpu" if available
        callbacks=[MetricTracker()],
        enable_model_summary=False
    )
    trainer.train()

    # 6. Plot results
    print("\n6. Plotting results...")
    plot_solution(solver)

    # Plot training losses
    metrics = trainer.callbacks[0].metrics
    plt.figure(figsize=(10, 6))
    for metric, loss in metrics.items():
        plt.plot(range(len(loss)), loss, label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("Training Losses")
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
