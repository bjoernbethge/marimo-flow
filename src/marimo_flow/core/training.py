"""Training helpers for the PINA demo."""

from __future__ import annotations

from typing import Any

from pina.solver import PINN, SupervisedSolver
from pina.trainer import Trainer


def train_solver(
    solver: PINN | SupervisedSolver,
    max_epochs: int = 1000,
    accelerator: str = "auto",
    callbacks: list[Any] | None = None,
    logger: Any = None,
    n_points: int = 1000,
    sample_mode: str = "random",
) -> Trainer:
    """Train the provided solver and return the fitted Trainer."""
    solver.problem.discretise_domain(n=n_points, mode=sample_mode, domains="all")
    trainer = Trainer(
        solver=solver,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=callbacks or [],
        logger=logger,
        enable_model_summary=False,
        limit_val_batches=0,
    )
    trainer.train()
    return trainer
