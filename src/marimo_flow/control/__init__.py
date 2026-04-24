"""Closed-loop control on top of a trained PINN surrogate.

Scope (Phase F):

* :func:`run_mpc_step` — single rolling-horizon optimisation given a
  state estimate, a ``ControlPlan``, and a callable surrogate.
* :func:`simulate_closed_loop` — drive a toy dynamics model forward
  under the MPC's selected controls so the demo notebook can report a
  closed-loop trajectory.

Kept deliberately small — scipy.optimize SLSQP is the inner QP engine,
no casadi / cvxpy / do-mpc. For anything beyond scalar-dynamics
examples escalate to Lead.
"""

from marimo_flow.control.mpc import run_mpc_step, simulate_closed_loop

__all__ = ["run_mpc_step", "simulate_closed_loop"]
