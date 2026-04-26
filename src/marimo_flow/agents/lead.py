"""Lead agent — single Pydantic-AI Agent wrapping the graph as one tool.

Used by:
  * marimo `mo.ui.chat` (see chat.py)
  * A2A    `agent.to_a2a()`  (see server/a2a.py)
  * AG-UI  `agent.to_ag_ui()` (see server/ag_ui.py)

mlflow.pydantic_ai.autolog() is enabled here so every nested sub-agent
call inside the graph produces traces under the active MLflow run.

The single graph-dispatch tool lives in `toolsets.lead.lead_toolset` as a
module-level `FunctionToolset[FlowDeps]` — the agent consumes it via
`toolsets=[lead_toolset]` and callers pass `deps` at run time.
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
from pydantic_ai import Agent

from marimo_flow.agents.deps import (
    DEFAULT_MLFLOW_ARTIFACT_ROOT,
    DEFAULT_MLFLOW_EXPERIMENT_NAME,
    DEFAULT_MLFLOW_LAYOUT_ROOT,
    DEFAULT_MLFLOW_TRACKING_URI,
    FlowDeps,
    get_model,
    resolve_mlflow_tracking_uri,
)
from marimo_flow.agents.toolsets.lead import lead_toolset

LEAD_INSTRUCTIONS = """\
You are the lead of a PINA (Physics-Informed NN) team.
For any user request that needs the team, call run_pina_workflow(intent).
For trivial chit-chat, answer directly.
"""

_AUTOLOG_ENABLED = False
_TRACKING_URI_APPLIED: str | None = None
_LAYOUT_PREPARED: str | None = None


def _looks_like_local_default(uri: str) -> bool:
    """True when ``uri`` is the baked-in local SQLite default.

    Used to decide whether to prepare ``data/mlflow/`` and pin a
    ``data/mlflow/artifacts/`` artifact root. Remote / user-supplied
    URIs (Postgres, MySQL, MLflow server HTTP) get left alone.
    """
    return uri == DEFAULT_MLFLOW_TRACKING_URI or uri.endswith(
        f"/{DEFAULT_MLFLOW_LAYOUT_ROOT}/db/mlflow.db"
    )


def _ensure_local_layout(uri: str) -> None:
    """For the local SQLite default, materialise ``data/mlflow/{db,artifacts}``
    and pin the marimo-flow experiment's ``artifact_location`` so runs land
    under ``data/mlflow/artifacts/`` instead of MLflow's CWD-relative
    ``./mlruns/`` fallback.
    """
    global _LAYOUT_PREPARED
    if _LAYOUT_PREPARED == uri or not _looks_like_local_default(uri):
        return

    layout = Path(DEFAULT_MLFLOW_LAYOUT_ROOT).resolve()
    (layout / "db").mkdir(parents=True, exist_ok=True)
    artifacts = layout / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    client = mlflow.MlflowClient()
    name = os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_MLFLOW_EXPERIMENT_NAME)
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(
            name, artifact_location=f"file:///{artifacts.as_posix()}"
        )
    # Make this the default for subsequent fluent API calls.
    os.environ["MLFLOW_EXPERIMENT_NAME"] = name
    mlflow.set_experiment(name)
    _LAYOUT_PREPARED = uri


def _ensure_tracking_uri(uri: str | None = None) -> None:
    """Point MLflow at the resolved tracking URI.

    Idempotent within a process for a given URI. Honours an explicit arg
    first, then ``MLFLOW_TRACKING_URI`` env / ``config.yaml`` / the baked-in
    default, in that order (see ``resolve_mlflow_tracking_uri``).

    Side effect: when the URI is the local SQLite default, ensure
    ``data/mlflow/`` exists with a marimo-flow experiment whose artifact
    root is ``data/mlflow/artifacts/`` (mirrors the docker-compose layout).
    """
    global _TRACKING_URI_APPLIED
    target = uri or resolve_mlflow_tracking_uri()
    if target == _TRACKING_URI_APPLIED:
        return
    mlflow.set_tracking_uri(target)
    _TRACKING_URI_APPLIED = target
    _ensure_local_layout(target)


# Re-export as a public symbol so tests / callers can verify layout prep.
__all__ = [
    "build_lead_agent",
    "DEFAULT_MLFLOW_ARTIFACT_ROOT",
    "_ensure_local_layout",
    "_ensure_tracking_uri",
]


def _ensure_autolog() -> None:
    """Enable MLflow autologging for the team.

    `mlflow.pydantic_ai.autolog()` is disabled by default because mlflow
    3.11.1 (latest PyPI release) raises `ValueError: Circular reference
    detected` in `dump_span_attribute_value` when used with pydantic-ai
    >= 1.80 — fix merged upstream in mlflow#22693 (2026-04-21) but not
    yet released. Opt back in via `MLFLOW_PYDANTIC_AI_AUTOLOG=1` once
    mlflow >= 3.11.2 is installed.
    """
    global _AUTOLOG_ENABLED
    if _AUTOLOG_ENABLED:
        return
    if os.environ.get("MLFLOW_PYDANTIC_AI_AUTOLOG") == "1":
        mlflow.pydantic_ai.autolog()
    mlflow.pytorch.autolog()
    _AUTOLOG_ENABLED = True


def build_lead_agent(*, model=None, deps: FlowDeps | None = None) -> Agent:
    """Build the lead agent. ``deps`` is optional; when given, its
    ``mlflow_tracking_uri`` overrides the resolver default. Callers still
    pass ``deps`` to ``agent.run(..., deps=...)`` at run time."""
    _ensure_tracking_uri(deps.mlflow_tracking_uri if deps else None)
    _ensure_autolog()
    model = model or get_model("lead")
    return Agent(
        model,
        deps_type=FlowDeps,
        instructions=LEAD_INSTRUCTIONS,
        toolsets=[lead_toolset],
    )
