"""Tests for the MPC control layer — pure-Python surrogate, no PINN."""

from __future__ import annotations

import numpy as np

from marimo_flow.agents.schemas import (
    ControlPlan,
    ControlVariableSpec,
    StateSpec,
)
from marimo_flow.control import run_mpc_step, simulate_closed_loop


def _scalar_surrogate(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Toy linear dynamics: x_{k+1} = 0.9 x_k + u_k."""
    traj = np.zeros((len(controls), 1))
    x = float(state[0])
    for i, u in enumerate(controls[:, 0]):
        x = 0.9 * x + float(u)
        traj[i, 0] = x
    return traj


def test_mpc_step_drives_state_toward_target():
    plan = ControlPlan(
        name="thermostat",
        surrogate_uri="mem://scalar",
        horizon=5,
        dt=1.0,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="x", target=0.0, weight=1.0)],
    )
    state_now = np.array([2.0])
    ctrl, info = run_mpc_step(plan, state_now, _scalar_surrogate)
    assert ctrl.shape == (5, 1)
    # Cost reduces the state toward 0 — first control should be negative.
    assert ctrl[0, 0] <= 0.0
    assert info["success"]


def test_closed_loop_converges_on_linear_plant():
    plan = ControlPlan(
        name="thermostat",
        surrogate_uri="mem://scalar",
        horizon=5,
        dt=1.0,
        controls=[ControlVariableSpec(name="u", low=-1.0, high=1.0)],
        states=[StateSpec(name="x", target=0.0, weight=1.0)],
    )
    traj = simulate_closed_loop(
        plan,
        initial_state=np.array([2.0]),
        surrogate=_scalar_surrogate,
        true_dynamics=_scalar_surrogate,
        n_steps=10,
    )
    assert traj["states"].shape == (11, 1)
    assert abs(traj["states"][-1, 0]) < abs(traj["states"][0, 0])
