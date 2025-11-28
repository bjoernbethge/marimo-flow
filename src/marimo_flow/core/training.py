"""Training helpers for the PINA demo."""

from __future__ import annotations

from pina.solver import PINN, SupervisedSolver
from pina.trainer import Trainer


def train_solver(
    solver: PINN | SupervisedSolver,
    max_epochs: int = 1000,
    accelerator: str = "auto",
) -> Trainer:
    """Train the provided solver and return the fitted Trainer."""

    trainer = Trainer(
        solver=solver,
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[],
    )
    trainer.train()
    return trainer

