"""ProblemSpec — compositional description of a PINA Problem.

No hardcoded ``kind`` enum. Agents construct a ProblemSpec from
primitives (equations, subdomains, conditions) and hand it to
``services.composer.compose_problem`` which assembles a live
``pina.Problem`` subclass at runtime. Any PDE that PINA's operators
can express is reachable without touching Python.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from marimo_flow.agents.schemas.equation import (
    ConditionSpec,
    EquationSpec,
    SubdomainSpec,
)


class ProblemSpec(BaseModel):
    """Compositional description of a PDE problem.

    The composer reads the fields as follows:

    * ``output_variables`` becomes the problem class attribute.
    * ``domain_bounds`` is the full ambient domain. Presence of ``"t"``
      promotes the problem to ``TimeDependentProblem`` (otherwise
      ``SpatialProblem``).
    * ``subdomains`` declares the named regions that appear in conditions
      (walls, initial-time slice, interior, …).
    * ``equations`` lists the PDE residuals; they are referenced by name
      from ``conditions[*].equation_name``.
    * ``conditions`` maps each subdomain to either an equation (interior
      PDE residual, IC expression) or a ``fixed_value`` (homogeneous
      Dirichlet etc.).
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        default=None,
        description="optional catalog label, e.g. 'burgers_1d_forward'",
    )
    output_variables: list[str]
    domain_bounds: dict[str, list[float]] = Field(
        description="full domain per axis, e.g. {'x': [-1.0, 1.0], 't': [0.0, 1.0]}"
    )
    subdomains: list[SubdomainSpec] = Field(default_factory=list)
    equations: list[EquationSpec] = Field(default_factory=list)
    conditions: list[ConditionSpec] = Field(default_factory=list)
    notes: str | None = None

    @property
    def time_dependent(self) -> bool:
        """True iff ``domain_bounds`` includes a time axis."""
        return "t" in self.domain_bounds
