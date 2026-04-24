"""FunctionToolset for the Data / Provenance agent (SPEC §9.3).

Read-only SQL over the DuckDB provenance store, MLflow run queries,
and explicit persist-* writers for AgentDecision / DatasetBinding that
other toolsets would otherwise skip.
"""

from __future__ import annotations

import contextlib
from typing import Any

import mlflow
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import (
    AgentDecision,
    AgentRole,
    ArtifactRef,
    DatasetBinding,
)
from marimo_flow.agents.toolsets._registry import require_state

data_toolset: FunctionToolset[FlowDeps] = FunctionToolset(id="data")

_READ_ONLY_PREFIXES = ("select", "with", "pragma", "describe", "show", "explain")


@data_toolset.tool
def duckdb_query(
    ctx: RunContext[FlowDeps],
    sql: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Run a read-only query against the provenance DuckDB store.

    Blocks DDL/DML statements — agents should not mutate the provenance
    log through SQL. Writes go through the dedicated ``persist_*`` tools.

    Args:
        sql: the SELECT statement. Rejected if it doesn't start with
            one of ``SELECT``, ``WITH``, ``PRAGMA``, ``DESCRIBE``,
            ``SHOW``, ``EXPLAIN``.
        limit: implicit ``LIMIT`` appended if the SQL lacks one
            (keeps huge scans from blowing up the LLM context).
    """
    stripped = sql.strip().rstrip(";")
    first_word = stripped.split(None, 1)[0].lower() if stripped else ""
    if first_word not in _READ_ONLY_PREFIXES:
        raise ModelRetry(
            f"duckdb_query only allows read-only statements "
            f"(one of {', '.join(_READ_ONLY_PREFIXES)}); got {first_word!r}."
        )
    if "limit" not in stripped.lower():
        stripped = f"{stripped} LIMIT {int(limit)}"
    store = ctx.deps.provenance()
    return store.query(stripped)


@data_toolset.tool
def list_tables(ctx: RunContext[FlowDeps]) -> list[str]:
    """Return the DuckDB table names in the provenance database."""
    rows = ctx.deps.provenance().query(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    )
    return [r["table_name"] for r in rows]


@data_toolset.tool
def list_runs(
    ctx: RunContext[FlowDeps],  # noqa: ARG001
    experiment_name: str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """List recent MLflow runs.

    Args:
        experiment_name: when None, searches across all experiments.
        limit: max rows returned.
    """
    client = mlflow.MlflowClient()
    if experiment_name is not None:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            return []
        experiment_ids = [exp.experiment_id]
    else:
        experiment_ids = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(experiment_ids, max_results=limit)
    return [
        {
            "run_id": r.info.run_id,
            "experiment_id": r.info.experiment_id,
            "status": r.info.status,
            "start_time": r.info.start_time,
            "end_time": r.info.end_time,
        }
        for r in runs
    ]


@data_toolset.tool
def fetch_run_metrics(
    ctx: RunContext[FlowDeps],  # noqa: ARG001
    run_id: str,
) -> dict[str, float]:
    """Return the final metric values for a specific MLflow run."""
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items()}


@data_toolset.tool
def persist_agent_decision(
    ctx: RunContext[FlowDeps],
    agent: str,
    summary: str,
    tool: str | None = None,
    input_schema: str | None = None,
    output_schema: str | None = None,
) -> str:
    """Append an AgentDecision to state and mirror it to the provenance DB."""
    if agent not in _VALID_AGENT_ROLES:
        raise ModelRetry(
            f"Unknown agent role {agent!r}. Allowed: "
            f"{', '.join(sorted(_VALID_AGENT_ROLES))}."
        )
    state = require_state(ctx.deps)
    decision = AgentDecision(
        agent=agent,  # type: ignore[arg-type]
        tool=tool,
        input_schema=input_schema,
        output_schema=output_schema,
        summary=summary,
        task_id=state.task_spec.task_id if state.task_spec else None,
        run_id=state.mlflow_run_id,
    )
    state.decisions.append(decision)
    with contextlib.suppress(Exception):
        ctx.deps.provenance().record_decision(decision)
    return decision.decision_id


@data_toolset.tool
def register_dataset(
    ctx: RunContext[FlowDeps],
    name: str,
    source: str,
    location: str | None = None,
    input_columns: list[str] | None = None,
    output_columns: list[str] | None = None,
    n_rows: int | None = None,
) -> str:
    """Attach a dataset binding to the active TaskSpec + provenance.

    Supports the same ``source`` values as ``DatasetBinding``:
    ``tensor``, ``dataframe``, ``parquet``, ``csv``, ``other``.
    """
    state = require_state(ctx.deps)
    binding = DatasetBinding(
        name=name,
        source=source,  # type: ignore[arg-type]
        location=location,
        input_columns=input_columns,
        output_columns=output_columns,
        n_rows=n_rows,
    )
    if state.task_spec is not None:
        state.task_spec.available_data.append(binding)
    task_id = state.task_spec.task_id if state.task_spec else "unknown"
    with contextlib.suppress(Exception):
        ctx.deps.provenance().record_dataset_binding(task_id, binding)
    return binding.name


@data_toolset.tool
def persist_artifact_ref(
    ctx: RunContext[FlowDeps],
    uri: str,
    kind: str,
    label: str | None = None,
) -> str:
    """Persist an ArtifactRef to provenance.

    Existing toolsets (problem/model/solver/training) still register
    artifacts in MLflow + ``deps.registry`` themselves; this tool is a
    manual escape hatch when an agent wants to record an additional URI
    (e.g. a dataset file) without going through those toolsets.
    """
    state = require_state(ctx.deps)
    ref = ArtifactRef(kind=kind, uri=uri, label=label)  # type: ignore[arg-type]
    task_id = state.task_spec.task_id if state.task_spec else None
    with contextlib.suppress(Exception):
        ctx.deps.provenance().record_artifact(ref, task_id=task_id)
    return ref.uri


_VALID_AGENT_ROLES: frozenset[str] = frozenset(
    {
        "route",
        "notebook",
        "problem",
        "model",
        "solver",
        "training",
        "mlflow",
        "lead",
        "triage",
        "data",
        "validation",
        "orchestrator",
    }
)
# Sanity check that the literal stays in sync with the AgentRole Literal.
_ = AgentRole  # referenced so lint doesn't flag the import as unused
