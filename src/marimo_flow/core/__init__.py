"""Public helpers for building PINA demos."""

# Legacy exports (for backward compatibility)
from .modeling import build_model, build_solver
from .problem import PoissonProblem, build_problem
from .training import train_solver
from .visualization import (
    build_comparison_chart,
    build_heatmap_chart,
    generate_error_data,
    generate_heatmap_data,
)
from .walrus import WalrusAdapter

# New manager-based API
from .callbacks import MarimoLivePlotter
from .model_factory import ModelFactory
from .problem_manager import ProblemManager
from .solver_manager import SolverManager

__all__ = [
    # Legacy
    "PoissonProblem",
    "WalrusAdapter",
    "build_comparison_chart",
    "build_heatmap_chart",
    "build_model",
    "build_problem",
    "build_solver",
    "generate_error_data",
    "generate_heatmap_data",
    "train_solver",
    # New managers
    "MarimoLivePlotter",
    "ModelFactory",
    "ProblemManager",
    "SolverManager",
]
