"""ObservationSpec + UnknownParameterSpec — inverse problem primitives.

Agents add these to a ``ProblemSpec`` when the goal is parameter
identification from data. The composer turns them into:

* a ``pina.problem.InverseProblem`` mixin with ``unknown_parameter_domain``
  populated from the declared parameter bounds;
* a ``data`` condition per ``ObservationSpec`` that wraps a
  ``(input, target)`` ``LabelTensor`` pair.

Observations are never hand-coded into the spec. Either:

* ``source="data_file"`` — the Data agent loads a CSV / Parquet / NPZ at
  ``path`` (axes + field columns); or
* ``source="synthetic"`` — the Data agent runs a forward solve with the
  declared ``true_parameters`` and samples ``n_points`` (+ optional
  Gaussian noise).

The concrete ``points`` + ``values`` arrays are filled in by the Data
agent before the composer runs.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UnknownParameterSpec(BaseModel):
    """One scalar parameter to be inferred from observations.

    Shows up in equation forms as a bare symbol (e.g. ``nu`` in
    ``u_t + u*u_x - nu*u_xx``). The composer routes the symbol through
    ``params_`` instead of ``parameters`` when emitting the torch
    residual, so PINA's inverse trainer can backprop into it.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="symbol as it appears in the equation form")
    low: float = Field(description="lower bound of the initial uniform prior")
    high: float = Field(description="upper bound of the initial uniform prior")


ObservationSource = Literal["data_file", "synthetic", "live_sensor"]


class ObservationSpec(BaseModel):
    """Data attached to the problem as a ``Condition(input=…, target=…)``.

    Two dispatch modes:

    * ``source="data_file"`` → Data agent loads ``path`` (a CSV / Parquet
      / NPZ) and extracts columns for each axis in ``axes`` plus ``field``.
    * ``source="synthetic"`` → Data agent runs a forward solve with
      ``true_parameters`` over ``n_points`` samples, applies Gaussian
      noise ``noise_sigma``, and fills ``points`` + ``values``.

    The composer reads only the materialised ``points`` / ``values``
    fields. Any agent or Data-agent implementation is free to fill them.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="data",
        description="condition key in the compiled problem (e.g. 'data')",
    )
    source: ObservationSource = "data_file"
    path: str | None = Field(
        default=None,
        description="path to the observation file when source=data_file",
    )
    field: str = Field(
        default="u",
        description="output variable measured (must appear in output_variables)",
    )
    axes: list[str] = Field(
        default_factory=list,
        description="input axes present in the data file, in column order",
    )
    n_points: int = Field(
        default=100,
        description="target point count (for synthetic generation or file sampling)",
    )
    noise_sigma: float = Field(
        default=0.0,
        description="Gaussian noise σ added to the measured field",
    )
    true_parameters: dict[str, float] = Field(
        default_factory=dict,
        description="ground-truth parameter values used by synthetic generation",
    )
    points: list[list[float]] | None = Field(
        default=None,
        description="materialised input tuples; shape (n, len(axes))",
    )
    values: list[list[float]] | None = Field(
        default=None,
        description="materialised target values; shape (n, 1) for a scalar field",
    )
