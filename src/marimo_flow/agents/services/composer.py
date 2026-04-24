"""compose_problem — turn a typed ``ProblemSpec`` into a live PINA Problem.

Core of the composition-first architecture. Agents build a
``ProblemSpec`` from primitives (``EquationSpec`` + ``SubdomainSpec`` +
``ConditionSpec``) and hand it here. The composer:

1. Splits ``domain_bounds`` into spatial + temporal axes (``"t"`` if
   present promotes to ``TimeDependentProblem``).
2. Builds a ``CartesianDomain`` per subdomain from its ``bounds`` dict.
3. Compiles each ``EquationSpec.form`` through ``sympy.sympify`` +
   ``sympy.lambdify`` into a torch callable, wiring derivatives through
   ``pina.operator.grad`` / ``laplacian``.
4. Emits a ``pina.Condition`` per ``ConditionSpec`` (``FixedValue`` for
   ``fixed_value``, ``Equation`` lookup for ``equation``).
5. Dynamically constructs a ``pina.Problem`` subclass with those class
   attributes.

No hardcoded equation catalog. Any PDE expressible in sympy over
derivatives, outputs, input variables and scalar parameters is
reachable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import sympy
import torch
from pina import Condition
from pina.domain import CartesianDomain
from pina.equation import Equation, FixedValue
from pina.operator import grad, laplacian
from pina.problem import SpatialProblem, TimeDependentProblem

from marimo_flow.agents.schemas import (
    ConditionSpec,
    DerivativeSpec,
    EquationSpec,
    ProblemSpec,
    SubdomainSpec,
)

_INPUT_VAR_CANDIDATES: tuple[str, ...] = ("x", "y", "z", "t")


def compose_problem(spec: ProblemSpec) -> type:
    """Assemble a ``pina.Problem`` subclass from a ``ProblemSpec``.

    Returns a class (not an instance) — PINA problem classes carry
    their structure as class attributes, matching the existing
    ``ProblemManager`` return shape so downstream toolsets
    (``ModelManager``, ``SolverManager``) keep working unchanged.
    """
    spatial_bounds, temporal_bounds = _split_domain(spec.domain_bounds)
    spatial_domain = CartesianDomain(spatial_bounds) if spatial_bounds else None
    temporal_domain = (
        CartesianDomain({"t": temporal_bounds}) if temporal_bounds else None
    )

    subdomains: dict[str, CartesianDomain] = {
        sd.name: _subdomain_to_cartesian(sd) for sd in spec.subdomains
    }

    equations: dict[str, Equation] = {
        eq.name: build_equation(eq) for eq in spec.equations
    }

    conditions: dict[str, Condition] = {}
    for cond in spec.conditions:
        conditions[cond.subdomain] = _compile_condition(cond, equations)

    attrs: dict[str, Any] = {
        "output_variables": list(spec.output_variables),
        "domains": subdomains,
        "conditions": conditions,
    }
    if spatial_domain is not None:
        attrs["spatial_domain"] = spatial_domain

    if spec.time_dependent:
        if temporal_domain is None:
            raise ValueError(
                "time_dependent=True but no 't' axis in domain_bounds"
            )
        attrs["temporal_domain"] = temporal_domain
        base = (TimeDependentProblem, SpatialProblem)
    else:
        base = (SpatialProblem,)

    class_name = spec.name or "ComposedProblem"
    return type(class_name, base, attrs)


def build_equation(spec: EquationSpec) -> Equation:
    """Compile a symbolic ``EquationSpec`` into a ``pina.Equation``.

    The returned ``Equation`` wraps a torch callable
    ``residual(input_, output_)`` that:

    * extracts each output field via ``output_.extract([field])``;
    * computes each declared derivative via PINA's ``grad`` /
      ``laplacian`` operators;
    * substitutes the declared parameters as Python scalars;
    * evaluates the sympy form through ``lambdify`` to a torch
      expression that autograd can differentiate through.
    """
    symbol_names, torch_fn = _build_lambda(spec)

    def residual(input_: Any, output_: Any) -> Any:
        values: dict[str, Any] = {}

        # Output fields referenced by bare name in the form (e.g. "u").
        for out_var in spec.outputs:
            if out_var in symbol_names:
                values[out_var] = output_.extract([out_var])

        # Input variables (x/y/z/t) if they appear in the form.
        for var in _INPUT_VAR_CANDIDATES:
            if var in symbol_names:
                values[var] = input_.extract([var])

        # Parameters — scalar constants.
        for pname, pval in spec.parameters.items():
            if pname in symbol_names:
                values[pname] = float(pval)

        # Derivatives via PINA operators.
        for deriv in spec.derivatives:
            if deriv.name in symbol_names:
                values[deriv.name] = _compute_derivative(input_, output_, deriv)

        missing = [n for n in symbol_names if n not in values]
        if missing:
            raise ValueError(
                f"equation '{spec.name}' is missing inputs for symbols: "
                f"{', '.join(missing)}"
            )

        return torch_fn(**values)

    return Equation(residual)


# --- internals -----------------------------------------------------


def _split_domain(
    bounds: dict[str, list[float]],
) -> tuple[dict[str, list[float]], list[float] | None]:
    spatial: dict[str, list[float]] = {}
    temporal: list[float] | None = None
    for axis, interval in bounds.items():
        if axis == "t":
            temporal = [float(v) for v in interval]
        else:
            spatial[axis] = [float(v) for v in interval]
    return spatial, temporal


def _subdomain_to_cartesian(sd: SubdomainSpec) -> CartesianDomain:
    """Convert a ``SubdomainSpec`` to ``CartesianDomain`` literal bounds.

    Scalars pin the axis (wall / initial slice); lists of length 2
    become intervals.
    """
    bounds: dict[str, float | list[float]] = {}
    for axis, val in sd.bounds.items():
        if isinstance(val, list | tuple):
            if len(val) == 1:
                bounds[axis] = float(val[0])
            elif len(val) == 2:
                bounds[axis] = [float(val[0]), float(val[1])]
            else:
                raise ValueError(
                    f"subdomain '{sd.name}' axis {axis!r}: bounds must have 1 or 2 values"
                )
        else:
            bounds[axis] = float(val)
    return CartesianDomain(bounds)


def _compile_condition(
    cond: ConditionSpec, equations: dict[str, Equation]
) -> Condition:
    if cond.kind == "fixed_value":
        if cond.value is None:
            raise ValueError(
                f"condition on '{cond.subdomain}' kind='fixed_value' needs value"
            )
        return Condition(domain=cond.subdomain, equation=FixedValue(cond.value))
    if cond.kind == "equation":
        if cond.equation_inline is not None:
            eq = build_equation(cond.equation_inline)
        elif cond.equation_name is not None:
            eq = equations.get(cond.equation_name)
            if eq is None:
                raise ValueError(
                    f"condition on '{cond.subdomain}' references unknown "
                    f"equation {cond.equation_name!r}"
                )
        else:
            raise ValueError(
                f"condition on '{cond.subdomain}' kind='equation' needs "
                "either equation_name or equation_inline"
            )
        return Condition(domain=cond.subdomain, equation=eq)
    raise ValueError(f"unknown condition kind: {cond.kind!r}")


_TORCH_MODULE: dict[str, Any] = {
    # Scalar constants the agent can reference in the form.
    "pi": float(torch.pi),
    "e": float(torch.e),
    # Unary torch ops that keep autograd-tracked tensors intact. numpy
    # on the fallback path would call .numpy() on grad-tracking tensors
    # and crash — always prefer torch.
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "Abs": torch.abs,
    "abs": torch.abs,
    "Min": torch.minimum,
    "Max": torch.maximum,
    "Pow": torch.pow,
}


def _build_lambda(
    spec: EquationSpec,
) -> tuple[list[str], Callable[..., Any]]:
    """Parse ``spec.form`` via sympy and return (symbol_names, torch_fn).

    ``sympy.lambdify`` is pointed at an explicit torch-op dict — this
    keeps autograd working end-to-end. The numpy fallback is poisonous
    for grad-tracking tensors (torch raises on ``.numpy()``), so we
    never use it.
    """
    names: set[str] = set()
    names.update(d.name for d in spec.derivatives)
    names.update(spec.outputs)
    names.update(spec.parameters)
    # Include any axis that appears literally in the form expression.
    for axis in _INPUT_VAR_CANDIDATES:
        if _token_present(spec.form, axis):
            names.add(axis)

    sympy_locals = {n: sympy.Symbol(n) for n in names}
    expr = sympy.sympify(spec.form, locals=sympy_locals)

    # Pull in any sympy-detected free symbols we missed (rare, but keeps
    # diagnostics honest).
    for sym in expr.free_symbols:
        if sym.name not in names:
            names.add(sym.name)
            sympy_locals[sym.name] = sym

    symbol_order = sorted(names)
    torch_fn_positional = sympy.lambdify(
        [sympy_locals[n] for n in symbol_order],
        expr,
        modules=[_TORCH_MODULE],
    )

    def torch_fn(**kwargs: Any) -> Any:
        args = [kwargs[n] for n in symbol_order]
        return torch_fn_positional(*args)

    return symbol_order, torch_fn


def _token_present(expr: str, token: str) -> bool:
    """True iff ``token`` appears as an identifier in the expression."""
    import re

    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", expr) is not None


def _compute_derivative(input_: Any, output_: Any, deriv: DerivativeSpec) -> Any:
    """Dispatch (grad / laplacian / chained grad) based on ``wrt`` shape."""
    if not deriv.wrt:
        raise ValueError(
            f"derivative {deriv.name!r} has empty 'wrt' — nothing to differentiate"
        )
    if len(deriv.wrt) == 1:
        return grad(output_, input_, components=[deriv.field], d=list(deriv.wrt))
    if len(set(deriv.wrt)) == 1:
        # Pure higher-order derivative in a single variable → laplacian.
        return laplacian(
            output_, input_, components=[deriv.field], d=deriv.wrt[:1]
        )
    # Mixed partials — chain grad per axis.
    result = output_
    for axis in deriv.wrt:
        result = grad(result, input_, components=[deriv.field], d=[axis])
    return result
