import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Physics-Informed Neural Networks with PINA""")
    return


@app.cell
def _():
    """Import PINA and PyTorch"""
    import torch
    from pina import Condition, LabelTensor, Problem
    from pina.geometry import CartesianDomain
    from pina.model import FeedForward
    from pina.solvers import PINNSolver
    from pina.trainer import Trainer
    from pina.equation import Laplacian
    
    return (
        CartesianDomain,
        Condition,
        FeedForward,
        LabelTensor,
        Laplacian,
        PINNSolver,
        Problem,
        Trainer,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""## 1. Define the Problem (Poisson Equation)""")
    return


@app.cell
def _(CartesianDomain, Condition, Laplacian, Problem, torch):
    class PoissonProblem(Problem):
        def __init__(self):
            super().__init__()
            self.spatial_domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
            
            # Equation: Laplacian(u) = -sin(pi*x)*sin(pi*y)
            self.equations = [
                Laplacian(
                    "u", 
                    lambda x: -torch.sin(torch.pi * x.extract(["x"])) * torch.sin(torch.pi * x.extract(["y"]))
                )
            ]
            
            # Boundary conditions
            self.conditions = {
                "gamma1": Condition(
                    location=CartesianDomain({"x": [0, 1], "y":  1}),
                    equation=lambda x: x.extract(["u"]) # u=0 on boundary
                ),
                "gamma2": Condition(
                    location=CartesianDomain({"x": [0, 1], "y": 0}),
                    equation=lambda x: x.extract(["u"])
                ),
                "gamma3": Condition(
                    location=CartesianDomain({"x": 1, "y": [0, 1]}),
                    equation=lambda x: x.extract(["u"])
                ),
                "gamma4": Condition(
                    location=CartesianDomain({"x": 0, "y": [0, 1]}),
                    equation=lambda x: x.extract(["u"])
                ),
                "D": Condition(
                    location=CartesianDomain({"x": [0, 1], "y": [0, 1]}),
                    equation=self.equations[0]
                )
            }
            
    problem = PoissonProblem()
    return PoissonProblem, problem


@app.cell
def _(mo):
    mo.md(r"""## 2. Define the Model and Solver""")
    return


@app.cell
def _(FeedForward, PINNSolver, problem):
    # Simple MLP
    model = FeedForward(
        layers=[2, 20, 20, 20, 1],
        output_variables=["u"],
        input_variables=["x", "y"]
    )
    
    solver = PINNSolver(
        problem=problem,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
    )
    return model, solver


@app.cell
def _(mo):
    """Training control"""
    train_button = mo.ui.run_button(label="🚀 Train PINN")
    mo.md(f"{train_button}")
    return (train_button,)


@app.cell
def _(Trainer, mo, solver, train_button):
    mo.stop(not train_button.value, "Click 'Train PINN' to start training")
    
    trainer = Trainer(
        solver=solver,
        max_epochs=1000,
        accelerator='auto',
        callbacks=[]
    )
    
    with mo.status.spinner("Training PINN..."):
        trainer.train()
        
    print("✅ Training complete")
    return (trainer,)


@app.cell
def _(mo):
    mo.md(r"""## 3. Visualize Solution""")
    return


@app.cell
def _(mo, solver, torch):
    import altair as alt
    import polars as pl
    import numpy as np
    
    # Create grid for visualization
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    
    # Predict
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)
    pts.requires_grad = True
    input_tensor = LabelTensor(pts, labels=["x", "y"])
    
    with torch.no_grad():
        u_pred = solver.model(input_tensor).numpy().flatten()
    
    # Plot
    df = pl.DataFrame({
        "x": X.ravel(),
        "y": Y.ravel(),
        "u": u_pred
    })
    
    chart = alt.Chart(df.to_pandas()).mark_rect().encode(
        x='x:Q',
        y='y:Q',
        color='u:Q',
        tooltip=['x', 'y', 'u']
    ).properties(
        title="PINN Solution u(x,y)"
    )
    
    return X, Y, alt, chart, df, input_tensor, np, pl, pts, u_pred, x, y


@app.cell
def _(chart):
    chart.display()
    return


if __name__ == "__main__":
    app.run()

