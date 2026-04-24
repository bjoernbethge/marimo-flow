"""Design-agent helpers: spec-path mutation + constraint handler.

Two orthogonal concerns live here:

* ``apply_design_overrides`` walks a ``ProblemSpec`` and replaces
  scalars at the paths declared in ``DesignVariableSpec.parameter_path``.
  No surrogate call, no side effects on the source spec.
* ``ConstraintAggregator`` evaluates a set of ``ConstraintSpec`` against
  a trained PINN surrogate's ``(input, output)`` samples and folds them
  into either a penalty term (``loss += λ·max(0, g)²``) or an
  augmented-Lagrangian term (outer-loop λ updates).

Kept deliberately thin — the actual optimiser live in the
``design_toolset``; these services just supply the pure-function
building blocks so tests can poke them without spinning up Optuna.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import sympy

from marimo_flow.agents.schemas import (
    ConstraintSpec,
    DesignVariableSpec,
    OptimizationPlan,
    ProblemSpec,
)

_PATH_SEGMENT = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[(\d+)\])?")


def apply_design_overrides(
    spec: ProblemSpec,
    design_variables: list[DesignVariableSpec],
    values: dict[str, float],
) -> ProblemSpec:
    """Return a deep copy of ``spec`` with each design variable replaced.

    Unknown variable names in ``values`` are silently ignored — callers
    (optimisers) often pass the full trial dict even when some variables
    are held fixed. Missing variables in ``values`` leave the spec
    unchanged.
    """
    new_spec = spec.model_copy(deep=True)
    for dv in design_variables:
        if dv.name not in values:
            continue
        _set_by_path(new_spec, dv.parameter_path, float(values[dv.name]))
    return new_spec


def _set_by_path(obj: Any, path: str, value: float) -> None:
    """Walk dotted/indexed ``path`` into ``obj`` and assign ``value`` at the leaf."""
    segments = [m for m in _PATH_SEGMENT.finditer(path) if m.group(1)]
    if not segments:
        raise ValueError(f"empty / unparseable path: {path!r}")
    cursor: Any = obj
    for idx, match in enumerate(segments):
        attr = match.group(1)
        index = match.group(3)
        is_last = idx == len(segments) - 1
        container = _get_attr_or_item(cursor, attr)
        if index is not None:
            pos = int(index)
            if is_last:
                if isinstance(container, dict):
                    container[pos] = value  # type: ignore[index]
                else:
                    container[pos] = value
                return
            cursor = container[pos]
        else:
            if is_last:
                _set_attr_or_item(cursor, attr, value)
                return
            cursor = container


def _get_attr_or_item(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj[key]
    return getattr(obj, key)


def _set_attr_or_item(obj: Any, key: str, value: float) -> None:
    if isinstance(obj, dict):
        obj[key] = value
        return
    setattr(obj, key, value)


class ConstraintAggregator:
    """Evaluate ``ConstraintSpec`` residuals against surrogate outputs.

    Each constraint's ``expression`` is lambdified once at construction;
    callers supply ``(values_by_name, design_values)`` at evaluation
    time. The aggregator returns a dict of raw residuals plus the
    total penalty (or aug-Lagrangian) term that gets added to the
    optimiser's objective.
    """

    def __init__(
        self,
        constraints: list[ConstraintSpec],
        design_variable_names: list[str],
    ):
        self._constraints = list(constraints)
        self._design_names = list(design_variable_names)
        self._compiled: list[tuple[list[str], Callable[..., float]]] = []
        for c in self._constraints:
            symbols, fn = _compile_expression(c.expression, self._design_names)
            self._compiled.append((symbols, fn))
        self._multipliers: dict[str, float] = {c.name: 0.0 for c in self._constraints}

    def evaluate_penalty(
        self,
        field_samples: dict[str, list[float]],
        design_values: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Return (per-constraint residuals aggregated, total penalty)."""
        residuals: dict[str, float] = {}
        total = 0.0
        for cspec, (symbols, fn) in zip(
            self._constraints, self._compiled, strict=False
        ):
            agg = _evaluate_aggregated(cspec, symbols, fn, field_samples, design_values)
            residuals[cspec.name] = agg
            gap = _constraint_gap(agg, cspec.op, cspec.rhs)
            if gap > 0.0:
                total += cspec.penalty_weight * gap * gap
        return residuals, total

    def evaluate_augmented_lagrangian(
        self,
        field_samples: dict[str, list[float]],
        design_values: dict[str, float],
    ) -> tuple[dict[str, float], float]:
        """Augmented-Lagrangian variant: μ·g² + λ·g with λ-update hook."""
        residuals: dict[str, float] = {}
        total = 0.0
        for cspec, (symbols, fn) in zip(
            self._constraints, self._compiled, strict=False
        ):
            agg = _evaluate_aggregated(cspec, symbols, fn, field_samples, design_values)
            residuals[cspec.name] = agg
            gap = _constraint_gap(agg, cspec.op, cspec.rhs)
            lam = self._multipliers[cspec.name]
            total += lam * gap + cspec.penalty_weight * gap * gap
        return residuals, total

    def update_multipliers(self, residuals: dict[str, float]) -> None:
        """Adam-style λ bump: λ += 2·μ·g on active constraints."""
        for cspec in self._constraints:
            g = _constraint_gap(residuals[cspec.name], cspec.op, cspec.rhs)
            if g > 0.0:
                self._multipliers[cspec.name] += 2.0 * cspec.penalty_weight * g

    @property
    def multipliers(self) -> dict[str, float]:
        return dict(self._multipliers)


def _compile_expression(
    expression: str, design_names: list[str]
) -> tuple[list[str], Callable[..., float]]:
    locals_: dict[str, sympy.Symbol] = {}
    expr = sympy.sympify(expression, locals=locals_)
    symbols = sorted({s.name for s in expr.free_symbols})
    for n in symbols:
        locals_[n] = sympy.Symbol(n)
    fn = sympy.lambdify(
        [locals_[n] for n in symbols],
        expr,
        modules=["math"],
    )
    return symbols, fn


def _evaluate_aggregated(
    cspec: ConstraintSpec,
    symbols: list[str],
    fn: Callable[..., float],
    field_samples: dict[str, list[float]],
    design_values: dict[str, float],
) -> float:
    n = _sample_count(field_samples)
    if n == 0:
        return 0.0
    totals: list[float] = []
    for i in range(n):
        args = []
        for name in symbols:
            if name in field_samples:
                args.append(float(field_samples[name][i]))
            elif name in design_values:
                args.append(float(design_values[name]))
            else:
                raise ValueError(
                    f"constraint '{cspec.name}' references unknown symbol {name!r}"
                )
        totals.append(float(fn(*args)))
    if cspec.aggregation == "mean":
        return sum(totals) / len(totals)
    if cspec.aggregation == "max":
        return max(totals)
    if cspec.aggregation == "min":
        return min(totals)
    raise ValueError(f"unknown aggregation {cspec.aggregation!r}")


def _constraint_gap(residual: float, op: str, rhs: float) -> float:
    """Return the amount by which the constraint is violated (>= 0)."""
    if op == "<=":
        return max(0.0, residual - rhs)
    if op == ">=":
        return max(0.0, rhs - residual)
    if op == "==":
        return abs(residual - rhs)
    raise ValueError(f"unknown constraint op {op!r}")


def _sample_count(field_samples: dict[str, list[float]]) -> int:
    counts = [len(v) for v in field_samples.values()]
    return counts[0] if counts else 0


__all__ = [
    "ConstraintAggregator",
    "apply_design_overrides",
]


# Silence unused-import warning; OptimizationPlan re-exported for toolsets.
_ = OptimizationPlan
