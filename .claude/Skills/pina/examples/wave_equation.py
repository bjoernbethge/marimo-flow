"""
Wave Equation Solver with PINA

Solves: ∂²u/∂t² = ∇²u on [0,1]×[0,1]×[0,1]
Initial: u(x,y,0) = sin(πx)sin(πy)
Boundary: u = 0 on all spatial boundaries
"""

import torch
from pina import Trainer
from pina.problem import TimeDependentProblem, SpatialProblem
from pina.domain import CartesianDomain
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.operator import grad, laplacian
from pina.model import FeedForward
from pina.solver import PINN
from pina.callbacks import MetricTracker
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def wave_equation(input_, output_):
    """Wave PDE residual: ∂²u/∂t² - ∇²u = 0"""
    u_t = grad(output_, input_, components=["u"], d=["t"])
    u_tt = grad(u_t, input_, components=["dudt"], d=["t"])
    nabla_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return nabla_u - u_tt


def initial_condition(input_, output_):
    """Initial condition: u(x,y,0) = sin(πx)sin(πy)"""
    u_expected = (
        torch.sin(torch.pi * input_.extract(["x"])) *
        torch.sin(torch.pi * input_.extract(["y"]))
    )
    return output_.extract(["u"]) - u_expected


class WaveEquation(TimeDependentProblem, SpatialProblem):
    """Time-dependent wave equation problem."""
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


@torch.no_grad()
def plot_snapshots(solver, time_steps=[0.0, 0.25, 0.5, 0.75, 1.0], n_spatial=50):
    """Plot solution at different time steps."""
    fig, axes = plt.subplots(1, len(time_steps), figsize=(4*len(time_steps), 4))

    for idx, t in enumerate(time_steps):
        # Create spatial grid at time t
        x = torch.linspace(0, 1, n_spatial)
        y = torch.linspace(0, 1, n_spatial)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        tt = torch.full_like(xx, t)

        # Stack into input
        pts_raw = torch.stack([xx.flatten(), yy.flatten(), tt.flatten()], dim=1)
        pts = solver.problem.input_pts_to_labels(pts_raw)

        # Predict
        u = solver(pts).extract("u").reshape(n_spatial, n_spatial)

        # Plot
        im = axes[idx].contourf(xx, yy, u, levels=20, cmap='RdBu_r')
        axes[idx].set_title(f"t = {t:.2f}")
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("y")
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def animate_solution(solver, n_spatial=50, n_time=100):
    """Create animation of wave propagation."""
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
        pts_raw = torch.stack([xx.flatten(), yy.flatten(), tt.flatten()], dim=1)
        pts = solver.problem.input_pts_to_labels(pts_raw)

        # Predict
        u = solver(pts).extract("u").reshape(n_spatial, n_spatial)

        # Plot
        im = ax.contourf(xx, yy, u, levels=20, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(f"Wave Equation: t = {t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return [im]

    anim = FuncAnimation(fig, update, frames=n_time, interval=50, blit=False)
    plt.show()

    return anim


def main():
    """Main execution."""
    print("=" * 60)
    print("Wave Equation Solver with PINA")
    print("=" * 60)

    # 1. Define problem
    print("\n1. Defining problem...")
    problem = WaveEquation()

    # 2. Discretize domain
    print("2. Discretizing domain...")
    problem.discretise_domain(n=2000, mode="random", domains="all")

    # 3. Create model
    print("3. Creating neural network...")
    model = FeedForward(
        input_dimensions=3,  # x, y, t
        output_dimensions=1,
        layers=[100, 100, 100],
        func=torch.nn.Tanh
    )

    # 4. Create PINN solver
    print("4. Creating PINN solver...")
    solver = PINN(problem=problem, model=model)

    # 5. Train
    print("5. Training...")
    trainer = Trainer(
        solver=solver,
        max_epochs=2000,
        accelerator="cpu",  # Change to "gpu" if available
        callbacks=[MetricTracker()],
        enable_model_summary=False
    )
    trainer.train()

    # 6. Visualize
    print("\n6. Plotting results...")

    # Snapshots at different times
    plot_snapshots(solver)

    # Training losses
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

    # Animate (optional)
    print("\n7. Creating animation...")
    anim = animate_solution(solver)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
