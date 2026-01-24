"""
Inverse Problem Example with PINA

Learn unknown parameters from observed data.

Problem: Solve Poisson equation with unknown parameters
∇²u = f(x, y; μ₁, μ₂)
where μ₁, μ₂ are unknown parameters to be learned from data.
"""

import torch
from pina import Trainer
from pina.problem import SpatialProblem, InverseProblem
from pina.domain import CartesianDomain
from pina.condition import Condition
from pina.equation import Equation, FixedValue
from pina.operator import laplacian
from pina.model import FeedForward
from pina.solver import PINN
from pina.callbacks import MetricTracker
import matplotlib.pyplot as plt


def generate_observation_data(n_points=100):
    """
    Generate synthetic observation data.

    True parameters: μ₁ = 0.5, μ₂ = -0.3
    """
    print("Generating observation data...")

    # True parameters
    mu1_true = 0.5
    mu2_true = -0.3

    # Sample points
    x = torch.rand(n_points) * 4 - 2  # [-2, 2]
    y = torch.rand(n_points) * 4 - 2  # [-2, 2]

    # True solution (synthetic)
    u_true = (
        torch.sin(mu1_true * torch.pi * x) *
        torch.sin(mu2_true * torch.pi * y)
    )

    # Stack into input tensor
    input_data = torch.stack([x, y], dim=1)

    print(f"Generated {n_points} observation points")
    print(f"True parameters: μ₁={mu1_true}, μ₂={mu2_true}")

    return input_data, u_true.unsqueeze(1), mu1_true, mu2_true


def poisson_equation_parametric(input_, output_):
    """
    Parametric Poisson equation.

    The parameters μ₁, μ₂ are stored in the problem.unknown_parameters
    and will be learned during training.
    """
    # Extract parameters (these are learnable)
    mu1 = output_.extract(["mu1"])
    mu2 = output_.extract(["mu2"])

    # Force term with parameters
    force_term = (
        torch.sin(mu1 * torch.pi * input_.extract(["x"])) *
        torch.sin(mu2 * torch.pi * input_.extract(["y"]))
    )

    # Laplacian
    laplacian_u = laplacian(output_, input_, components=["u"], d=["x", "y"])

    return laplacian_u - force_term


class PoissonInverse(SpatialProblem, InverseProblem):
    """Inverse problem: learn unknown parameters."""
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-2, 2], "y": [-2, 2]})

    # Define parameter domain
    unknown_parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})

    domains = {
        "border": CartesianDomain({"x": [-2, 2], "y": 2}) |
                  CartesianDomain({"x": [-2, 2], "y": -2}) |
                  CartesianDomain({"x": 2, "y": [-2, 2]}) |
                  CartesianDomain({"x": -2, "y": [-2, 2]}),
        "interior": CartesianDomain({"x": [-2, 2], "y": [-2, 2]})
    }

    # Conditions will be set dynamically with data
    conditions = {}

    def add_data_condition(self, data_input, data_output):
        """Add data condition for inverse problem."""
        self.conditions = {
            "border": Condition(domain="border", equation=FixedValue(0.0)),
            "interior": Condition(domain="interior", equation=Equation(poisson_equation_parametric)),
            "data": Condition(input_points=data_input, output_points=data_output)
        }


@torch.no_grad()
def plot_solution(solver, n_points=100):
    """Plot learned solution."""
    # Sample grid
    pts = solver.problem.spatial_domain.sample(n_points, "grid")

    # Get predictions
    predicted = solver(pts).extract("u").detach()

    # Reshape
    x = pts.extract(["x"]).reshape(n_points, n_points)
    y = pts.extract(["y"]).reshape(n_points, n_points)
    u_grid = predicted.reshape(n_points, n_points)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(x, y, u_grid, levels=20, cmap='viridis')
    ax.set_title("Learned Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_parameter_history(history):
    """Plot parameter learning history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # μ₁ history
    axes[0].plot(history['mu1'], 'o-', label='Learned μ₁')
    axes[0].axhline(history['mu1_true'], color='r', linestyle='--', label='True μ₁')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("μ₁")
    axes[0].set_title("Parameter μ₁ Learning")
    axes[0].legend()
    axes[0].grid(True)

    # μ₂ history
    axes[1].plot(history['mu2'], 's-', label='Learned μ₂')
    axes[1].axhline(history['mu2_true'], color='r', linestyle='--', label='True μ₂')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("μ₂")
    axes[1].set_title("Parameter μ₂ Learning")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main execution."""
    print("=" * 60)
    print("Inverse Problem: Learning Unknown Parameters")
    print("=" * 60)

    # 1. Generate observation data
    print("\n1. Generating observation data...")
    data_input, data_output, mu1_true, mu2_true = generate_observation_data(n_points=100)

    # 2. Define inverse problem
    print("\n2. Defining inverse problem...")
    problem = PoissonInverse()
    problem.add_data_condition(data_input, data_output)

    # 3. Discretize spatial domains
    print("\n3. Discretizing domain...")
    problem.discretise_domain(n=500, mode="random", domains=["border", "interior"])

    # 4. Create model
    print("\n4. Creating neural network...")
    model = FeedForward(
        input_dimensions=2,
        output_dimensions=1,
        layers=[64, 64, 64],
        func=torch.nn.Tanh
    )

    # 5. Create PINN solver for inverse problem
    print("\n5. Creating PINN solver...")
    solver = PINN(problem=problem, model=model)

    # Initialize parameters (random initial guess)
    solver.problem.unknown_parameters = torch.nn.Parameter(
        torch.rand(2) * 0.2 - 0.1  # Random in [-0.1, 0.1]
    )

    print(f"Initial parameter guess: μ₁={solver.problem.unknown_parameters[0]:.4f}, "
          f"μ₂={solver.problem.unknown_parameters[1]:.4f}")

    # 6. Train
    print("\n6. Training (learning parameters)...")

    # Track parameter evolution
    parameter_history = {
        'mu1': [],
        'mu2': [],
        'mu1_true': mu1_true,
        'mu2_true': mu2_true
    }

    # Custom callback to track parameters
    class ParameterTracker:
        def on_train_epoch_end(self, trainer, solver):
            params = solver.problem.unknown_parameters.detach()
            parameter_history['mu1'].append(params[0].item())
            parameter_history['mu2'].append(params[1].item())

    trainer = Trainer(
        solver=solver,
        max_epochs=2000,
        accelerator="cpu",  # Change to "gpu" if available
        callbacks=[MetricTracker(), ParameterTracker()],
        enable_model_summary=False
    )
    trainer.train()

    # 7. Results
    print("\n7. Results...")
    learned_params = solver.problem.unknown_parameters.detach()

    print(f"\nLearned Parameters:")
    print(f"  μ₁ = {learned_params[0]:.6f}  (true: {mu1_true})")
    print(f"  μ₂ = {learned_params[1]:.6f}  (true: {mu2_true})")

    print(f"\nParameter Errors:")
    print(f"  |μ₁ - μ₁_true| = {abs(learned_params[0] - mu1_true):.6f}")
    print(f"  |μ₂ - μ₂_true| = {abs(learned_params[1] - mu2_true):.6f}")

    # 8. Visualize
    print("\n8. Visualizing results...")

    # Parameter learning history
    plot_parameter_history(parameter_history)

    # Learned solution
    plot_solution(solver)

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

    print("\n" + "=" * 60)
    print("Inverse problem solved!")
    print("=" * 60)


if __name__ == "__main__":
    main()
