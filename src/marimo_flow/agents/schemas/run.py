"""ModelSpec, SolverPlan, RunConfig — typed contracts for the training step.

``ModelKind`` / ``SolverKind`` mirror the registries in
``marimo_flow.core.ModelManager`` and ``SolverManager`` so the toolsets
can compile a spec back into the exact ``Manager.create(...)`` call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelKind = Literal[
    "feedforward",
    "residual",
    "fno",
    "deeponet",
    "mionet",
    "pirate",
    "walrus",
]

SolverKind = Literal[
    "pinn",
    "sapinn",
    "causalpinn",
    "gradientpinn",
    "rbapinn",
    "supervised",
]


class ModelSpec(BaseModel):
    """Neural architecture choice, sized against the registered Problem.

    Common kwargs by kind (see ``ModelManager._REGISTRY``):

    * feedforward / residual / pirate: ``layers``, ``activation``
    * fno: ``n_modes``, ``dimensions``, ``inner_size``, ``n_layers``
    * deeponet: ``branch_net``, ``trunk_net`` (provided via ``kwargs``)
    * walrus: ``checkpoint``, ``freeze_backbone``
    """

    model_config = ConfigDict(extra="forbid")

    kind: ModelKind
    layers: list[int] | None = None
    activation: str | None = None
    kwargs: dict[str, object] = Field(default_factory=dict)
    notes: str | None = None


class SolverPlan(BaseModel):
    """Solver + optimizer plan.

    The solver toolset resolves ``optimizer_type`` (e.g. ``"Adam"``) to
    the corresponding ``torch.optim`` class when calling
    ``SolverManager.create(...)``.
    """

    model_config = ConfigDict(extra="forbid")

    kind: SolverKind
    learning_rate: float = 1e-3
    optimizer_type: str = "Adam"
    loss: str | None = None
    kwargs: dict[str, object] = Field(default_factory=dict)
    notes: str | None = None


class RunConfig(BaseModel):
    """Trainer + discretisation knobs.

    Consumed by ``training_toolset.train(...)``. ``accelerator`` matches
    PyTorch Lightning's ``Trainer(accelerator=...)`` values; ``xpu`` is
    the Intel-GPU path exposed through the repo's XPU tooling.
    """

    model_config = ConfigDict(extra="forbid")

    max_epochs: int = 1000
    accelerator: Literal["auto", "cpu", "gpu", "mps", "xpu"] = "auto"
    n_points: int = 1000
    sample_mode: Literal["random", "grid", "lh"] = "random"
    seed: int | None = None
    notes: str | None = None
