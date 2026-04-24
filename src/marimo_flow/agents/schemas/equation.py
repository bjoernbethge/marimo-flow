"""EquationSpec + SubdomainSpec + ConditionSpec — composition primitives.

The Problem agent produces these instead of picking a hardcoded kind.
The composer (``services.composer.compose_problem``) turns them into a
live ``pina.Problem`` subclass at runtime.

Typed + explicit so agents can reason about PDEs structurally (which
field, which derivative, which parameter) rather than writing Python.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DerivativeSpec(BaseModel):
    """One symbolic derivative used in an equation.

    Example for ``u_t`` (∂u/∂t): ``name="u_t"``, ``field="u"``, ``wrt=["t"]``.
    For ``u_xx`` (∂²u/∂x²): ``wrt=["x", "x"]``. For mixed ``u_xy``:
    ``wrt=["x", "y"]``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="label used in the equation form, e.g. 'u_t'")
    field: str = Field(description="output variable being differentiated, e.g. 'u'")
    wrt: list[str] = Field(
        description="input vars to differentiate against in order, e.g. ['x','x']"
    )


class EquationSpec(BaseModel):
    """A symbolic PDE residual (the part set to zero).

    ``form`` is a sympy-parseable expression using the labels declared in
    ``derivatives`` (e.g. ``u_t``, ``u_xx``), the output variable names
    (``u``, ``v``, ``p``), the input variable names (``x``, ``y``, ``z``,
    ``t``), and any names in ``parameters``. The composer compiles the
    form into a torch callable via ``sympy.lambdify`` and wires the
    derivatives through ``pina.operator.grad`` / ``laplacian``.

    Example for 1D viscous Burgers (u_t + u·u_x − ν·u_xx = 0)::

        EquationSpec(
            name="burgers",
            form="u_t + u*u_x - nu*u_xx",
            outputs=["u"],
            derivatives=[
                DerivativeSpec(name="u_t",  field="u", wrt=["t"]),
                DerivativeSpec(name="u_x",  field="u", wrt=["x"]),
                DerivativeSpec(name="u_xx", field="u", wrt=["x","x"]),
            ],
            parameters={"nu": 0.01},
        )
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    form: str = Field(description="sympy expression; residual form (= 0)")
    outputs: list[str] = Field(
        description="output vars referenced in the form, e.g. ['u']"
    )
    derivatives: list[DerivativeSpec] = Field(default_factory=list)
    parameters: dict[str, float] = Field(default_factory=dict)


class SubdomainSpec(BaseModel):
    """A named sub-region of the full domain.

    ``bounds`` is a dict of axis → range. A single-element list or a
    scalar pins the coordinate (boundary wall / initial-time slice);
    a 2-element list is an interval. Example 1D time-dependent
    interior ``D`` on x∈[-1,1], t∈[0,1]::

        SubdomainSpec(name="D", bounds={"x": [-1.0, 1.0], "t": [0.0, 1.0]})

    Left wall (x=-1, t∈[0,1])::

        SubdomainSpec(name="left", bounds={"x": -1.0, "t": [0.0, 1.0]})
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    bounds: dict[str, float | list[float]]


ConditionKind = Literal["fixed_value", "equation"]


class ConditionSpec(BaseModel):
    """Attach an equation (or a fixed scalar) to a named subdomain.

    ``kind="fixed_value"`` + ``value=0.0`` → homogeneous Dirichlet.
    ``kind="equation"`` + ``equation_name="..."`` → points at an
    EquationSpec defined at the ProblemSpec level (either the main PDE
    residual applied on an interior subdomain, or a BC/IC expression).
    ``equation_inline`` is an escape hatch for one-off conditions
    (e.g. a custom IC) that shouldn't clutter the top-level equations list.
    """

    model_config = ConfigDict(extra="forbid")

    subdomain: str
    kind: ConditionKind
    value: float | None = None
    equation_name: str | None = None
    equation_inline: EquationSpec | None = None
