"""Public helpers for building PINA demos."""

from .model_factory import create_model_for_problem
from .model_manager import ModelManager
from .problem_manager import ProblemManager
from .solver_manager import SolverManager
from .training import train_solver
from .visualization import (
    build_optuna_history_figure,
    build_optuna_parallel_figure,
    build_optuna_param_importance_figure,
    build_trials_scatter_chart,
    study_trials_dataframe,
)
from .walrus import FoundationModelAdapter

__all__ = [
    "build_optuna_history_figure",
    "build_optuna_parallel_figure",
    "build_optuna_param_importance_figure",
    "build_trials_scatter_chart",
    "create_model_for_problem",
    "FoundationModelAdapter",
    "ModelManager",
    "ProblemManager",
    "SolverManager",
    "study_trials_dataframe",
    "train_solver",
]
