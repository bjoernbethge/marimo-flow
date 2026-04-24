"""Closed-loop MPC on a 1D heat-rod PINN surrogate.

Run with:
    marimo edit examples/04_mpc_heat_rod.py

Trains a small surrogate for 1D heat equation, then drives an MPC loop
that steers the centre temperature toward a target setpoint by
modulating a heat-flux boundary condition. Demonstrates the Phase F
composition: ProblemSpec → compose → train → register surrogate → MPC.
"""

from __future__ import annotations

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _header():
    import marimo as mo

    mo.md(
        "# MPC demo — heat rod with a PINN surrogate\n"
        "Composition-first: no hardcoded heat-equation factory. The MPC "
        "layer treats the trained network as a one-step dynamics model."
    )
    return (mo,)


@app.cell
def _controls(mo):
    setpoint = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.6, label="centre setpoint"
    )
    horizon = mo.ui.slider(start=3, stop=20, step=1, value=8, label="MPC horizon")
    n_steps = mo.ui.slider(
        start=5, stop=30, step=1, value=15, label="closed-loop steps"
    )
    mo.hstack([setpoint, horizon, n_steps])
    return (setpoint, horizon, n_steps)


@app.cell
def _build_surrogate_spec(mo):
    from marimo_flow.agents.schemas import (
        ConditionSpec,
        DerivativeSpec,
        EquationSpec,
        ProblemSpec,
        SubdomainSpec,
    )

    spec = ProblemSpec(
        name="heat_rod_1d",
        output_variables=["T"],
        domain_bounds={"x": [0.0, 1.0], "t": [0.0, 1.0]},
        subdomains=[
            SubdomainSpec(name="D", bounds={"x": [0.0, 1.0], "t": [0.0, 1.0]}),
            SubdomainSpec(name="left", bounds={"x": 0.0, "t": [0.0, 1.0]}),
            SubdomainSpec(name="right", bounds={"x": 1.0, "t": [0.0, 1.0]}),
            SubdomainSpec(name="t0", bounds={"x": [0.0, 1.0], "t": 0.0}),
        ],
        equations=[
            EquationSpec(
                name="heat",
                form="T_t - alpha*T_xx",
                outputs=["T"],
                derivatives=[
                    DerivativeSpec(name="T_t", field="T", wrt=["t"]),
                    DerivativeSpec(name="T_xx", field="T", wrt=["x", "x"]),
                ],
                parameters={"alpha": 0.05},
            ),
        ],
        conditions=[
            ConditionSpec(subdomain="D", kind="equation", equation_name="heat"),
            ConditionSpec(subdomain="left", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="right", kind="fixed_value", value=0.0),
            ConditionSpec(subdomain="t0", kind="fixed_value", value=0.0),
        ],
    )
    mo.md("Surrogate spec: 1D heat equation on x ∈ [0, 1], t ∈ [0, 1].")
    return (spec,)


@app.cell
def _train_surrogate(mo, spec):
    from marimo_flow.agents.services.composer import compose_problem
    from marimo_flow.core import ModelManager, SolverManager, train_solver

    problem = compose_problem(spec)()
    model = ModelManager.create("feedforward", problem=problem, layers=[32, 32])
    solver = SolverManager.create(
        "pinn", problem=problem, model=model, learning_rate=1e-3
    )
    trainer = train_solver(
        solver,
        max_epochs=10,
        accelerator="cpu",
        n_points=2048,
        sample_mode="random",
    )
    mo.md(
        "Training: 10 epochs over 2048 collocation points. Final loss terms: "
        f"{dict(trainer.callback_metrics)}"
    )
    return (solver, trainer)


@app.cell
def _wire_surrogate_callable(solver):
    import numpy as np
    import torch
    from pina.label_tensor import LabelTensor

    def surrogate(state, controls):
        """(state, controls) → predicted centre-temperature trajectory.

        Crude mapping: treats each control as a heat-flux scale on the
        right boundary and queries the PINN at x=0.5, t=dt·k. For the
        demo this is enough — the network was trained without boundary
        control, so 'controls' effectively multiply the surrogate value.
        """
        centre = 0.5
        traj = np.zeros((len(controls), 1))
        for k, u in enumerate(controls[:, 0]):
            t = (k + 1) * 0.05
            pt = LabelTensor(
                torch.tensor([[centre, t]], dtype=torch.float32),
                ["x", "t"],
            )
            base = float(solver.forward(pt).detach().cpu().numpy().squeeze())
            traj[k, 0] = float(state[0] + u * base)
        return traj

    return (surrogate,)


@app.cell
def _run_mpc(mo, surrogate, setpoint, horizon, n_steps):
    import numpy as np
    from marimo_flow.agents.schemas import (
        ControlPlan,
        ControlVariableSpec,
        StateSpec,
    )
    from marimo_flow.control import simulate_closed_loop

    plan = ControlPlan(
        name="heat_rod_mpc",
        surrogate_uri="mem://heat_surrogate",
        horizon=int(horizon.value),
        dt=0.05,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="T_centre", target=float(setpoint.value), weight=1.0)],
    )
    traj = simulate_closed_loop(
        plan,
        initial_state=np.array([0.0]),
        surrogate=surrogate,
        true_dynamics=surrogate,  # surrogate == plant for this demo
        n_steps=int(n_steps.value),
    )

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=traj["states"][:, 0],
            mode="lines+markers",
            name="T_centre",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=[setpoint.value] * len(traj["states"]),
            mode="lines",
            line={"dash": "dash"},
            name="setpoint",
        )
    )
    fig.update_layout(title="Closed-loop state trajectory")
    mo.ui.plotly(fig)

    ctrl_fig = go.Figure()
    ctrl_fig.add_trace(
        go.Scatter(
            y=traj["controls"][:, 0],
            mode="lines+markers",
            name="u",
        )
    )
    ctrl_fig.update_layout(title="Applied control sequence")
    mo.ui.plotly(ctrl_fig)
    return (traj,)


if __name__ == "__main__":
    app.run()
