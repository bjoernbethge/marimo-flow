"""FunctionToolset for the Design agent (Phase D).

Tools the Design agent calls to run a design-space sweep over a trained
PINN surrogate: apply overrides, evaluate constraints, run the chosen
optimiser. The outer loop is intentionally shallow — Optuna is the
default engine; SLSQP is available when the design variables are few
and the problem is smooth.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import (
    ConstraintSpec,
    DesignVariableSpec,
    OptimizationPlan,
    ProblemSpec,
)
from marimo_flow.agents.services.design import (
    ConstraintAggregator,
    apply_design_overrides,
)

design_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="design")


@design_toolset.tool
def build_optimization_plan(
    ctx: RunContext[FlowDeps],  # noqa: ARG001 — dep not needed
    name: str,
    objective_expression: str,
    design_variables: list[dict[str, Any]],
    constraints: list[dict[str, Any]] | None = None,
    method: str = "optuna_tpe",
    n_trials: int = 20,
    reuse_surrogate: bool = True,
    objective_aggregation: str = "mean",
) -> dict[str, Any]:
    """Validate + persist an OptimizationPlan.

    Returns the serialised plan so the caller can hand it straight to
    ``run_design_sweep`` — the tools don't share state implicitly.
    """
    plan = OptimizationPlan(
        name=name,
        objective_expression=objective_expression,
        objective_aggregation=objective_aggregation,  # type: ignore[arg-type]
        design_variables=[DesignVariableSpec(**dv) for dv in design_variables],
        constraints=[ConstraintSpec(**c) for c in (constraints or [])],
        method=method,  # type: ignore[arg-type]
        n_trials=n_trials,
        reuse_surrogate=reuse_surrogate,
    )
    return plan.model_dump()


@design_toolset.tool
def apply_overrides(
    ctx: RunContext[FlowDeps],  # noqa: ARG001
    spec: dict[str, Any],
    design_variables: list[dict[str, Any]],
    values: dict[str, float],
) -> dict[str, Any]:
    """Return a new ProblemSpec with the chosen design-variable values applied."""
    ps = ProblemSpec.model_validate(spec)
    dvs = [DesignVariableSpec(**dv) for dv in design_variables]
    updated = apply_design_overrides(ps, dvs, values)
    return updated.model_dump()


@design_toolset.tool
def evaluate_constraints(
    ctx: RunContext[FlowDeps],  # noqa: ARG001
    constraints: list[dict[str, Any]],
    design_variable_names: list[str],
    field_samples: dict[str, list[float]],
    design_values: dict[str, float],
    method: str = "penalty",
) -> dict[str, Any]:
    """Run the constraint aggregator over a trained-surrogate sample set.

    ``field_samples`` is a dict of ``output-field-name → list[float]``
    (e.g. ``{"u": [...], "ux": [...]}``) evaluated at test points.
    Returns residuals per constraint + the aggregated penalty.
    """
    cspecs = [ConstraintSpec(**c) for c in constraints]
    agg = ConstraintAggregator(cspecs, design_variable_names)
    if method == "penalty":
        residuals, total = agg.evaluate_penalty(field_samples, design_values)
    elif method == "augmented_lagrangian":
        residuals, total = agg.evaluate_augmented_lagrangian(
            field_samples, design_values
        )
    else:
        raise ModelRetry(
            f"unknown constraint method {method!r}; use "
            "'penalty' or 'augmented_lagrangian'"
        )
    return {"residuals": residuals, "penalty": total}


@design_toolset.tool
def run_design_sweep(
    ctx: RunContext[FlowDeps],  # noqa: ARG001 — side effects via caller
    plan: dict[str, Any],
    objective_fn_registry_key: str,
) -> dict[str, Any]:
    """Drive an Optuna / SLSQP sweep given a pre-registered objective fn.

    The objective function must be placed by the caller into
    ``deps.registry[objective_fn_registry_key]`` with signature::

        objective(design_values: dict[str, float]) -> float

    It encodes: compose(overrides) → (re-)train → sample surrogate →
    aggregate constraints → return scalar objective. The Design agent
    should construct this function once per plan and register it before
    calling this tool.

    Returns the best trial's ``(params, value)``.
    """
    try:
        plan_model = OptimizationPlan.model_validate(plan)
    except Exception as exc:  # noqa: BLE001
        raise ModelRetry(f"plan validation failed: {exc}") from exc

    fn = ctx.deps.registry.get(objective_fn_registry_key)
    if fn is None or not callable(fn):
        raise ModelRetry(
            f"no callable objective registered under {objective_fn_registry_key!r}"
        )

    if plan_model.method == "optuna_tpe":
        return _sweep_optuna(plan_model, fn)
    if plan_model.method == "scipy_slsqp":
        return _sweep_scipy(plan_model, fn)
    if plan_model.method in {"penalty", "augmented_lagrangian"}:
        # Those are constraint-handling policies, not global optimisers —
        # bounce back through Optuna with the chosen policy baked into fn.
        return _sweep_optuna(plan_model, fn)
    raise ModelRetry(f"unknown optimiser {plan_model.method!r}")


def _sweep_optuna(
    plan: OptimizationPlan, fn: Callable[[dict[str, float]], float]
) -> dict[str, Any]:
    import optuna

    def trial_fn(trial: optuna.Trial) -> float:
        values = {
            dv.name: trial.suggest_float(dv.name, dv.low, dv.high)
            for dv in plan.design_variables
        }
        return float(fn(values))

    study = optuna.create_study(direction="minimize")
    study.optimize(trial_fn, n_trials=plan.n_trials, show_progress_bar=False)
    return {
        "best_params": dict(study.best_trial.params),
        "best_value": float(study.best_value),
        "n_trials_completed": len(study.trials),
    }


def _sweep_scipy(
    plan: OptimizationPlan, fn: Callable[[dict[str, float]], float]
) -> dict[str, Any]:
    from scipy.optimize import minimize

    names = [dv.name for dv in plan.design_variables]
    x0 = [
        dv.initial if dv.initial is not None else 0.5 * (dv.low + dv.high)
        for dv in plan.design_variables
    ]
    bounds = [(dv.low, dv.high) for dv in plan.design_variables]

    def scalar_fn(x: list[float]) -> float:
        return float(fn(dict(zip(names, x, strict=True))))

    result = minimize(
        scalar_fn,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": plan.n_trials},
    )
    return {
        "best_params": dict(zip(names, [float(v) for v in result.x], strict=True)),
        "best_value": float(result.fun),
        "n_trials_completed": int(result.nit),
    }
