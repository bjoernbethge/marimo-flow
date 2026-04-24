"""NoiseSpec — declarative additive noise term for stochastic PDEs.

The composer reads ``ProblemSpec.noise`` (when set) and wraps each
interior equation with an additive noise term evaluated via Monte
Carlo over ``n_realisations`` batches. The mean residual is what gets
minimised — this is the variational-PINN formulation.

Scope: white noise (δ-correlated in space/time), coloured noise with a
user-supplied correlation length, and fractional Brownian motion as a
soft-dep via the ``fbm`` package (not bundled yet; escalate if needed).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NoiseKind = Literal["white", "colored", "fbm"]


class NoiseSpec(BaseModel):
    """Additive noise term attached to a stochastic PDE.

    The effective residual becomes::

        R(u; x, t, ξ) = F(u) - intensity · ξ(x, t)

    with ``ξ`` drawn per collocation batch. For coloured noise the
    sampler applies a Gaussian convolution in the input coordinates
    with ``correlation_length`` as the kernel width.
    """

    model_config = ConfigDict(extra="forbid")

    kind: NoiseKind = "white"
    intensity: float = Field(
        default=1.0,
        description="scalar multiplier applied to the raw noise sample",
    )
    correlation_length: float | None = Field(
        default=None,
        description="required for kind='colored'; kernel width in axis units",
    )
    hurst: float | None = Field(
        default=None,
        description="Hurst exponent for kind='fbm' (0 < H < 1)",
    )
    n_realisations: int = Field(
        default=8,
        description="Monte-Carlo samples per collocation batch for the "
        "variational formulation",
    )
    seed: int | None = Field(
        default=None,
        description="RNG seed for reproducibility; None = fresh per run",
    )
