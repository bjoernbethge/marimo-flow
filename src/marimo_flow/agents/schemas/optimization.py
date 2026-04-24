"""OptimizationPlan + DesignVariableSpec + ConstraintSpec.

The Design agent wraps the PINN forward/inverse problem in an outer
optimisation loop: change the design variables, re-train (or reuse) the
surrogate, compare against the objective. ``method`` picks the
optimiser; ``constraints`` are handled through the ``design_toolset``
penalty / augmented-Lagrangian helpers.

Scope (Phase D): scalar design variables and inequality constraints.
Shape / topology as distributed fields is a later iteration — that
needs either level-set coupling or SIMP-style material interpolation.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OptimizationMethod = Literal[
    "optuna_tpe",
    "scipy_slsqp",
    "penalty",
    "augmented_lagrangian",
]


ConstraintOp = Literal["<=", ">=", "=="]


class DesignVariableSpec(BaseModel):
    """One scalar design variable the outer loop will sweep.

    ``parameter_path`` picks the target inside the ProblemSpec. Format:

    * ``"equations[0].parameters.nu"`` — parameter on the first equation.
    * ``"domain_bounds.x[1]"`` — upper end of the x interval.
    * ``"conditions[3].value"`` — the lid speed on a Dirichlet wall.

    The design toolset walks the dotted/indexed path and replaces the
    scalar at each optimisation step before re-composing.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    parameter_path: str = Field(
        description="dotted path into the ProblemSpec, e.g. "
        "'equations[0].parameters.nu'"
    )
    low: float
    high: float
    initial: float | None = Field(
        default=None,
        description="optimiser starting point; defaults to (low+high)/2",
    )


class ConstraintSpec(BaseModel):
    """An inequality / equality constraint evaluated from the surrogate.

    ``expression`` is a sympy-style expression over the trained
    solver's output fields and the design variables (by ``name``). The
    design toolset lambdifies it and evaluates at the configured
    ``evaluation_points`` — average, max, or min aggregation set by
    ``aggregation``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    expression: str = Field(
        description="sympy expression over output fields + design vars"
    )
    op: ConstraintOp = "<="
    rhs: float = 0.0
    aggregation: Literal["mean", "max", "min"] = "max"
    penalty_weight: float = Field(
        default=1.0,
        description="λ multiplier in the penalty / aug-Lagrangian loss",
    )


class OptimizationPlan(BaseModel):
    """Recipe for a design-space sweep over a PINN surrogate.

    ``objective_expression`` follows the same rules as ``ConstraintSpec.expression``
    — same sympy dialect, same aggregation options via ``objective_aggregation``.
    The optimiser minimises the aggregated value (flip the sign for
    maximisation).

    ``reuse_surrogate`` controls the re-training policy. ``True`` =
    train once then query the surrogate at every outer step (fast but
    less accurate when the PDE is non-linear in the design variable).
    ``False`` = re-train from scratch per trial (slow but correct).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    objective_expression: str
    objective_aggregation: Literal["mean", "max", "min"] = "mean"
    design_variables: list[DesignVariableSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    method: OptimizationMethod = "optuna_tpe"
    n_trials: int = 20
    reuse_surrogate: bool = True
    notes: str | None = None
