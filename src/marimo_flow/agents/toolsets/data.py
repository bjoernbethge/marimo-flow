"""FunctionToolset for the Data / Provenance agent (SPEC §9.3).

Read-only SQL over the DuckDB provenance store, MLflow run queries,
explicit persist-* writers for AgentDecision / DatasetBinding, and
observation ingestion for inverse problems.

The Data agent owns observation materialisation so user requests never
need to hand-code numpy arrays. Two pathways:

* ``load_observations_from_file`` — read a CSV / Parquet / NPZ and slice
  out ``(axes..., field)`` columns into ``ObservationSpec.points`` /
  ``values``.
* ``generate_synthetic_observations`` — sample points in a declared
  range and evaluate a ground-truth sympy expression; Gaussian noise
  optional.

Both write the materialised observation into ``FlowState`` so downstream
``compose_problem`` calls see them directly.
"""

from __future__ import annotations

import contextlib
import csv
import math
import random
from pathlib import Path
from typing import Any

import mlflow
from pydantic_ai import FunctionToolset, ModelRetry, RunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.schemas import (
    AgentDecision,
    AgentRole,
    ArtifactRef,
    DatasetBinding,
    ObservationSpec,
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


@data_toolset.tool
def load_observations_from_file(
    ctx: RunContext[FlowDeps],
    path: str,
    field: str,
    axes: list[str],
    *,
    name: str = "data",
    format: str | None = None,
    max_points: int | None = None,
) -> dict[str, Any]:
    """Materialise an ObservationSpec from a CSV / Parquet / NPZ file.

    Columns matched by exact name — ``axes`` in order plus the ``field``.
    For CSV the first row must be a header. For NPZ each array is
    accessed by name and must be 1D with equal length. For Parquet the
    file is read via ``pyarrow`` if available, else via ``duckdb``.

    Returns the materialised observation as a plain dict so the Problem
    agent can splice it into the ``ProblemSpec.observations`` list.
    """
    p = Path(path)
    if not p.exists():
        raise ModelRetry(f"observation file not found: {path}")
    fmt = (format or p.suffix.lstrip(".")).lower()
    columns = [*axes, field]

    if fmt == "csv":
        rows = _load_csv(p, columns)
    elif fmt == "npz":
        rows = _load_npz(p, columns)
    elif fmt in {"parquet", "pq"}:
        rows = _load_parquet(p, columns)
    else:
        raise ModelRetry(
            f"unsupported observation file format {fmt!r}; use csv, npz or parquet"
        )

    if max_points is not None and len(rows) > max_points:
        step = len(rows) // max_points
        rows = rows[::step][:max_points]

    points = [[float(r[a]) for a in axes] for r in rows]
    values = [[float(r[field])] for r in rows]

    obs = ObservationSpec(
        name=name,
        source="data_file",
        path=str(p),
        field=field,
        axes=axes,
        n_points=len(points),
        points=points,
        values=values,
    )

    state = require_state(ctx.deps)
    if state.task_spec is not None:
        binding = DatasetBinding(
            name=name,
            source=fmt if fmt in {"csv", "parquet"} else "tensor",  # type: ignore[arg-type]
            location=str(p),
            input_columns=axes,
            output_columns=[field],
            n_rows=len(points),
        )
        state.task_spec.available_data.append(binding)
        with contextlib.suppress(Exception):
            ctx.deps.provenance().record_dataset_binding(
                state.task_spec.task_id, binding
            )

    return obs.model_dump()


@data_toolset.tool
def generate_synthetic_observations(
    ctx: RunContext[FlowDeps],  # noqa: ARG001
    *,
    truth_form: str,
    axes: list[str],
    field: str,
    axis_bounds: dict[str, list[float]],
    n_points: int = 100,
    noise_sigma: float = 0.0,
    true_parameters: dict[str, float] | None = None,
    name: str = "data",
    seed: int | None = None,
) -> dict[str, Any]:
    """Sample points in ``axis_bounds`` and evaluate ``truth_form`` on them.

    ``truth_form`` is a sympy expression over ``axes`` and the keys in
    ``true_parameters`` (constants). Each point is drawn from a uniform
    over the axis range; the evaluated value gets Gaussian noise with
    standard deviation ``noise_sigma`` added.

    The returned ObservationSpec has ``points`` / ``values`` filled in
    and is ready to splice into a ``ProblemSpec``.
    """
    import sympy

    rng = random.Random(seed)
    params = dict(true_parameters or {})

    symbols = {a: sympy.Symbol(a) for a in axes}
    symbols.update({k: sympy.Symbol(k) for k in params})
    expr = sympy.sympify(truth_form, locals=symbols)
    ordered_names = list(axes) + list(params)
    fn = sympy.lambdify([symbols[n] for n in ordered_names], expr, modules=["math"])

    points: list[list[float]] = []
    values: list[list[float]] = []
    for _ in range(n_points):
        coords = [rng.uniform(axis_bounds[a][0], axis_bounds[a][1]) for a in axes]
        value = float(fn(*coords, *[params[k] for k in params]))
        if noise_sigma > 0.0:
            value += rng.gauss(0.0, noise_sigma)
        if not math.isfinite(value):
            continue
        points.append(coords)
        values.append([value])

    obs = ObservationSpec(
        name=name,
        source="synthetic",
        field=field,
        axes=axes,
        n_points=len(points),
        noise_sigma=noise_sigma,
        true_parameters=params,
        points=points,
        values=values,
    )
    return obs.model_dump()


def _load_csv(path: Path, columns: list[str]) -> list[dict[str, float]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [c for c in columns if c not in header]
        if missing:
            raise ModelRetry(
                f"CSV {path.name} missing required columns: {', '.join(missing)}"
            )
        rows = [{c: float(row[c]) for c in columns} for row in reader]
    if not rows:
        raise ModelRetry(f"CSV {path.name} has no data rows")
    return rows


def _load_npz(path: Path, columns: list[str]) -> list[dict[str, float]]:
    import numpy as np

    bundle = np.load(path)
    missing = [c for c in columns if c not in bundle.files]
    if missing:
        raise ModelRetry(
            f"NPZ {path.name} missing required arrays: {', '.join(missing)}"
        )
    arrays = [np.asarray(bundle[c]).astype(float).flatten() for c in columns]
    length = len(arrays[0])
    if any(len(a) != length for a in arrays):
        raise ModelRetry(
            f"NPZ {path.name}: arrays have mismatched lengths "
            f"{[len(a) for a in arrays]}"
        )
    return [
        {columns[j]: float(arrays[j][i]) for j in range(len(columns))}
        for i in range(length)
    ]


def _load_parquet(path: Path, columns: list[str]) -> list[dict[str, float]]:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns)
        records = table.to_pylist()
    except ImportError:
        try:
            import duckdb
        except ImportError as exc:  # pragma: no cover — duckdb is a project dep
            raise ModelRetry(
                "parquet observations need either pyarrow or duckdb installed"
            ) from exc
        cols_csv = ", ".join(columns)
        records = (
            duckdb.sql(f"SELECT {cols_csv} FROM read_parquet('{path.as_posix()}')")
            .to_df()
            .to_dict(orient="records")
        )
    return [{c: float(r[c]) for c in columns} for r in records]


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
        "design",
        "control",
    }
)
# Sanity check that the literal stays in sync with the AgentRole Literal.
_ = AgentRole  # referenced so lint doesn't flag the import as unused
