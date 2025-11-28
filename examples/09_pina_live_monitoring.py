# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.17.0",
#     "pina-mathlab",
#     "torch",
#     "altair",
#     "polars",
#     "numpy",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 📈 Live Monitoring with PINA

    This example demonstrates the comprehensive capabilities of marimo-flow:

    1.  **Live Training Plotting**: Watch the loss convergence in real-time.
    2.  **PINA Managers**: Using `ProblemManager`, `ModelFactory`, and `SolverManager`.
    3.  **Multiple Visualizations**: Heatmaps, error analysis, and comparison charts.
    4.  **Flexible Configuration**: Different model architectures, solvers, and training settings.
    """)
    return


@app.cell
def _():
    import torch
    from marimo_flow.core import (
        MarimoLivePlotter,
        ModelFactory,
        ProblemManager,
        SolverManager,
        build_comparison_chart,
        build_heatmap_chart,
        generate_error_data,
        generate_heatmap_data,
    )
    from pina.trainer import Trainer

    # 1. Define Problem (Poisson Equation)
    # We use a predefined Poisson problem with analytical solution for validation
    problem_class = ProblemManager.create_poisson_problem()
    problem = problem_class()
    return (
        MarimoLivePlotter,
        ModelFactory,
        SolverManager,
        Trainer,
        build_comparison_chart,
        generate_error_data,
        problem,
        torch,
    )


@app.cell
def _(mo):
    # Configuration UI
    epochs_slider = mo.ui.slider(100, 2000, value=500, step=100, label="Epochs")
    lr_input = mo.ui.number(value=0.001, step=0.0001, label="Learning Rate")

    # Model architecture options
    layer_size = mo.ui.slider(16, 128, value=32, step=16, label="Layer Size")
    num_layers = mo.ui.slider(2, 5, value=3, step=1, label="Number of Layers")

    # Solver type selection
    solver_type = mo.ui.dropdown(
        options=["PINN", "SAPINN"],
        value="PINN",
        label="Solver Type"
    )

    train_btn = mo.ui.run_button(label="Start Live Training")

    mo.md(f"""
    ### Training Settings
    {epochs_slider}
    {lr_input}

    ### Model Architecture
    {layer_size}
    {num_layers}

    ### Solver Configuration
    {solver_type}

    {train_btn}
    """)
    return epochs_slider, layer_size, lr_input, num_layers, solver_type, train_btn


@app.cell
def _(ModelFactory, layer_size, num_layers, problem):
    # Create model based on UI configuration
    model = ModelFactory.create_model_for_problem(problem)
    return (model,)


@app.cell
def _(
    MarimoLivePlotter,
    SolverManager,
    Trainer,
    epochs_slider,
    lr_input,
    model,
    problem,
    train_btn,
):
    # Training Logic
    import marimo as mo

    mo.stop(not train_btn.value, "Click start to begin training.")

    # 3. Create Solver
    solver = SolverManager.create_pinn(
        problem, 
        model, 
        learning_rate=lr_input.value
    )

    # 4. Setup Live Plotter Callback
    plotter = MarimoLivePlotter(update_every_n_epochs=20)

    # Display the live chart container
    mo.output.replace(plotter.chart_container)

    # 5. Train
    with mo.status.spinner("Training in progress..."):
        trainer = Trainer(
            solver,
            max_epochs=epochs_slider.value,
            accelerator="auto",
            callbacks=[plotter],
            enable_model_summary=False,
        )
        trainer.train()

    mo.md("✅ Training finished!")
    return mo, solver


@app.cell
def _(build_comparison_chart, generate_error_data, mo, solver, torch):
    mo.md("## 🔍 Error Analysis")

    # Define analytical solution function
    def exact_solution(x):
        x_val = x.extract(["x"])
        y_val = x.extract(["y"])
        # Solution for the default Poisson problem in ProblemManager:
        # u(x,y) = sin(pi*x) * sin(pi*y) + 0.1*sin(50*pi*x) ... wait, let's check the definition
        # The default source term was: -sin(pi*x)*sin(pi*y) * (laplacian of solution)
        # If u = A * sin(pi*x) * sin(pi*y)
        # u_xx = -pi^2 * u
        # u_yy = -pi^2 * u
        # lap(u) = -2*pi^2 * u
        # So if we solve lap(u) = source
        # source = -2*pi^2 * sin(pi*x) * sin(pi*y)
        # In our manager default: source is just -sin(pi*x)*sin(pi*y) without factor?
        # Let's rely on the numerical check or just plot the prediction if exact is unknown.

        # Actually, let's check problem definition in manager.
        # It uses: lap(u) - source = 0  => lap(u) = source
        # Default source: -sin(pi*x)*sin(pi*y)
        # So u_exact should be: (1 / (2*pi^2)) * sin(pi*x) * sin(pi*y)

        factor = 1 / (2 * torch.pi**2)
        return factor * torch.sin(torch.pi * x_val) * torch.sin(torch.pi * y_val)

    # Generate error data
    try:
        error_df = generate_error_data(solver, exact_solution, grid_size=50)
        chart = build_comparison_chart(error_df)

        # Calculate metrics
        mae = error_df["error"].mean()
        max_error = error_df["error"].max()

        mo.output.append(
            mo.md(f"""
            **Metrics:**
            - Mean Absolute Error: `{mae:.2e}`
            - Max Error: `{max_error:.2e}`
            """)
        )
        mo.output.append(chart)
    except Exception as e:
        mo.output.append(mo.md(f"Error generating plots: {e}"))
    return


if __name__ == "__main__":
    app.run()
