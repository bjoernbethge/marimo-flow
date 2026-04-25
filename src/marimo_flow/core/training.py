"""Training helpers for the PINA demo."""

from __future__ import annotations

from typing import Any

from pina.solver import PINN, SupervisedSolver
from pina.trainer import Trainer

_UNSET = object()


def train_solver(
    solver: PINN | SupervisedSolver,
    max_epochs: int = 1000,
    accelerator: str = "auto",
    callbacks: list[Any] | None = None,
    logger: Any = _UNSET,
    n_points: int = 1000,
    sample_mode: str = "random",
) -> Trainer:
    """Train the provided solver and return the fitted Trainer.

    ``logger`` defaults to ``False`` (no logger). Pass an explicit Lightning
    logger (CSVLogger, MLflowLogger, …) to enable per-step metrics. The
    Lightning default ``logger=True`` would emit a UserWarning about
    ``tensorboardX`` being absent — opt out of that auto-default here.
    """
    solver.problem.discretise_domain(n=n_points, mode=sample_mode, domains="all")
    trainer = Trainer(
        solver=solver,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks or [],
        logger=False if logger is _UNSET else logger,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.train()
    return trainer
