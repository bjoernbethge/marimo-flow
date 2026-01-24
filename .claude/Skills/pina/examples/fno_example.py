"""
Fourier Neural Operator (FNO) Example with PINA

Learns operator mapping from initial conditions to solutions
for parameterized PDEs.
"""

import torch
import torch.nn as nn
from pina import Trainer
from pina.model import FNO
from pina.solver import SupervisedSolver
from pina.problem.zoo import SupervisedProblem
from pina.callbacks import MetricTracker
import matplotlib.pyplot as plt


def generate_darcy_flow_data(n_samples=100, grid_size=64):
    """
    Generate synthetic Darcy flow data.

    Input: Permeability field (n_samples, grid_size, grid_size, 1)
    Output: Pressure field (n_samples, grid_size, grid_size, 1)
    """
    print(f"Generating {n_samples} Darcy flow samples...")

    # Generate random permeability fields
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    input_data = []
    output_data = []

    for i in range(n_samples):
        # Random permeability field
        k = 1 + 0.5 * torch.sin(2 * torch.pi * (torch.rand(1) * xx + torch.rand(1)))
        k = k * torch.sin(2 * torch.pi * (torch.rand(1) * yy + torch.rand(1)))

        # Simplified pressure solution (for demonstration)
        # In practice, solve Darcy's law: -∇·(k∇p) = f
        p = torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
        p = p / k  # Simplified inverse relationship

        input_data.append(k.unsqueeze(-1))
        output_data.append(p.unsqueeze(-1))

    input_data = torch.stack(input_data)
    output_data = torch.stack(output_data)

    print(f"Input shape: {input_data.shape}")
    print(f"Output shape: {output_data.shape}")

    return input_data, output_data


def plot_predictions(fno_solver, test_input, test_output, n_samples=3):
    """Plot FNO predictions vs ground truth."""
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))

    with torch.no_grad():
        for idx in range(n_samples):
            # Input
            inp = test_input[idx:idx+1]
            true_out = test_output[idx]
            pred_out = fno_solver.forward(inp)[0]

            # Convert to numpy
            inp_np = inp[0, :, :, 0].cpu().numpy()
            true_np = true_out[:, :, 0].cpu().numpy()
            pred_np = pred_out[:, :, 0].cpu().numpy()

            # Plot input
            im0 = axes[idx, 0].imshow(inp_np, cmap='viridis')
            axes[idx, 0].set_title(f"Sample {idx+1}: Input (Permeability)")
            axes[idx, 0].axis('off')
            plt.colorbar(im0, ax=axes[idx, 0])

            # Plot true output
            im1 = axes[idx, 1].imshow(true_np, cmap='RdBu_r')
            axes[idx, 1].set_title("True Output (Pressure)")
            axes[idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[idx, 1])

            # Plot predicted output
            im2 = axes[idx, 2].imshow(pred_np, cmap='RdBu_r')
            axes[idx, 2].set_title("FNO Prediction")
            axes[idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[idx, 2])

    plt.tight_layout()
    plt.show()


def plot_error_analysis(fno_solver, test_input, test_output):
    """Analyze prediction errors."""
    with torch.no_grad():
        predictions = fno_solver.forward(test_input)
        errors = torch.abs(predictions - test_output)

        mean_error = errors.mean(dim=[1, 2, 3])
        max_error = errors.max(dim=1)[0].max(dim=1)[0].max(dim=1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Error distribution
    axes[0].hist(mean_error.cpu().numpy(), bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Mean Absolute Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].grid(True)

    # Error vs sample
    axes[1].plot(mean_error.cpu().numpy(), 'o-', label='Mean Error')
    axes[1].plot(max_error.cpu().numpy(), 's-', label='Max Error')
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Error per Sample")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Main execution."""
    print("=" * 60)
    print("Fourier Neural Operator (FNO) with PINA")
    print("=" * 60)

    # Configuration
    n_train = 800
    n_test = 200
    grid_size = 64
    n_modes = 12
    hidden_dim = 32

    # 1. Generate data
    print("\n1. Generating training data...")
    train_input, train_output = generate_darcy_flow_data(n_train, grid_size)

    print("\n2. Generating test data...")
    test_input, test_output = generate_darcy_flow_data(n_test, grid_size)

    # 2. Create problem
    print("\n3. Creating supervised problem...")
    problem = SupervisedProblem(train_input, train_output)

    # 3. Define FNO model
    print("\n4. Creating FNO model...")

    # Lifting network
    lifting_net = nn.Sequential(
        nn.Linear(1, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )

    # Projection network
    projecting_net = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1)
    )

    # FNO
    fno = FNO(
        lifting_net=lifting_net,
        projecting_net=projecting_net,
        n_modes=n_modes,
        dimensions=2,
        inner_size=hidden_dim,
        padding=8
    )

    print(f"FNO parameters: {sum(p.numel() for p in fno.parameters()):,}")

    # 4. Create solver
    print("\n5. Creating solver...")
    solver = SupervisedSolver(
        problem=problem,
        model=fno,
        use_lt=False
    )

    # 5. Train
    print("\n6. Training FNO...")
    trainer = Trainer(
        solver=solver,
        max_epochs=100,
        batch_size=16,
        accelerator="cpu",  # Change to "gpu" if available
        callbacks=[MetricTracker()],
        enable_model_summary=False
    )
    trainer.train()

    # 6. Visualize results
    print("\n7. Visualizing results...")

    # Training loss
    metrics = trainer.callbacks[0].metrics
    plt.figure(figsize=(10, 6))
    for metric, loss in metrics.items():
        plt.plot(range(len(loss)), loss, label=metric)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.title("FNO Training Loss")
    plt.tight_layout()
    plt.show()

    # Predictions
    plot_predictions(solver.model, test_input, test_output, n_samples=3)

    # Error analysis
    plot_error_analysis(solver.model, test_input, test_output)

    # 7. Compute test metrics
    print("\n8. Computing test metrics...")
    with torch.no_grad():
        predictions = solver.model(test_input)
        mae = torch.abs(predictions - test_output).mean()
        mse = ((predictions - test_output) ** 2).mean()
        relative_error = torch.abs(predictions - test_output).mean() / test_output.abs().mean()

        print(f"\nTest Metrics:")
        print(f"  Mean Absolute Error: {mae:.6f}")
        print(f"  Mean Squared Error:  {mse:.6f}")
        print(f"  Relative Error:      {relative_error:.6f}")

    print("\n" + "=" * 60)
    print("FNO training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
