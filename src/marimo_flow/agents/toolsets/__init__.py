"""FunctionToolset modules for the PINA multi-agent team.

Each toolset is a module-level singleton so tools can be tested in isolation
(`ts.call_tool("name", {...}, deps=...)`) and attached to Agents via
`Agent(..., toolsets=[ts, ...])`. Tools access FlowState through
`ctx.deps.state`, which each Node wires in before running the agent.

The four workflow toolsets are thin wrappers over the corresponding
`marimo_flow.core` Manager (Problem/Model/Solver/Training) — no business
logic lives in the toolset beyond MLflow-artifact bookkeeping.
"""

from marimo_flow.agents.toolsets.curator import curator_toolset
from marimo_flow.agents.toolsets.data import data_toolset
from marimo_flow.agents.toolsets.lead import lead_toolset
from marimo_flow.agents.toolsets.model import model_toolset
from marimo_flow.agents.toolsets.problem import problem_toolset
from marimo_flow.agents.toolsets.skills import skills_toolset
from marimo_flow.agents.toolsets.solver import solver_toolset
from marimo_flow.agents.toolsets.training import training_toolset
from marimo_flow.agents.toolsets.validation import validation_toolset

__all__ = [
    "curator_toolset",
    "data_toolset",
    "lead_toolset",
    "model_toolset",
    "problem_toolset",
    "skills_toolset",
    "solver_toolset",
    "training_toolset",
    "validation_toolset",
]
