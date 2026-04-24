"""ProblemNode — defines a PINA Problem and stores it via the registry pattern."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

if TYPE_CHECKING:
    from marimo_flow.agents.nodes.route import RouteNode

PROBLEM_SKILLS = ["pina-problem"]


def _define_problem(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    """Materialise a PINA Problem from `spec`, log it, return its MLflow URI.

    For now we serialise the spec as JSON and log it as an artifact under the
    active MLflow run; the live PINA Problem instance is built lazily by
    the SolverNode (which is the only place it is needed end-to-end).
    """
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    client = mlflow.MlflowClient()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "problem_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        client.log_artifact(state.mlflow_run_id, str(p), artifact_path="problem")
    uri = f"runs:/{state.mlflow_run_id}/problem/problem_spec.json"
    deps.registry[uri] = spec
    state.problem_artifact_uri = uri
    return uri


@dataclass
class ProblemNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]) -> RouteNode:
        from marimo_flow.agents.nodes.route import RouteNode
        from marimo_flow.agents.toolsets.curator import curator_toolset
        from marimo_flow.agents.toolsets.problem import problem_toolset

        model = self.model_override or ctx.deps.model_for("problem")
        ctx.deps.state = ctx.state
        agent = Agent(
            model,
            deps_type=FlowDeps,
            instructions=build_skill_instructions(PROBLEM_SKILLS),
            toolsets=[curator_toolset, problem_toolset],
            retries=3,
        )
        task_title = (
            ctx.state.task_spec.title if ctx.state.task_spec else "(no task_spec)"
        )
        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Task title: {task_title}\n"
            "You COMPOSE problems from primitives — you never pick a hardcoded "
            "kind. Workflow:\n"
            "1. search_presets(family='problem', query=<equation family or "
            "   descriptive term>) — check if a similar composition is already "
            "   in the catalog.\n"
            "2a. If a match exists → describe_preset + clone_preset with any "
            "    parameter overrides, then compose_problem(spec) using the "
            "    merged spec_json.\n"
            "2b. Otherwise → build a ProblemSpec from scratch:\n"
            "    - output_variables (list of field names)\n"
            "    - domain_bounds (axis → [min, max]; include 't' for "
            "      time-dependent)\n"
            "    - subdomains (named walls / initial slices / interior)\n"
            "    - equations (EquationSpec per residual — form is a sympy "
            "      expression over derivatives, parameters, outputs, axes; "
            "      derivatives MUST be declared explicitly with field + wrt)\n"
            "    - conditions (each subdomain → fixed_value OR equation reference)\n"
            "    Call compose_problem(spec) to build + register.\n"
            "3. If the composition succeeded and is worth keeping, call "
            "   register_preset to persist it for future runs.\n"
            "4. Call inspect_problem to sanity-check what you built before handing "
            "   back to the router.",
            deps=ctx.deps,
        )
        ctx.state.last_node = "problem"
        ctx.state.history.setdefault("problem", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
