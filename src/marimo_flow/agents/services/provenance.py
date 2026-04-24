"""DuckDB-backed provenance store (SPEC §12).

Mirrors every typed spec and provenance record the agents produce into
a local DuckDB file so runs can be inspected with SQL without having
to crack open MLflow artifacts. MLflow still owns the binary artifacts;
this store owns the *index* over them.

Tables (one row = one record):

* ``tasks``                — TaskSpec
* ``problem_specs``        — ProblemSpec (history, one per version)
* ``model_specs``          — ModelSpec
* ``solver_plans``         — SolverPlan
* ``run_configs``          — RunConfig
* ``dataset_bindings``     — DatasetBinding
* ``artifacts``            — ArtifactRef (MLflow URI → kind/label)
* ``experiments``          — ExperimentRecord (per end-to-end run)
* ``metrics``              — (experiment_id, run_id, name, value)
* ``agent_decisions``      — AgentDecision
* ``handoff_records``      — HandoffRecord
* ``validation_reports``   — ValidationReport
* ``lineage_edges``        — typed edges between artifact URIs

DuckDB 1.5.2 ships transitively via ``marimo[sql]``; no extra project
dependency needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from marimo_flow.agents.schemas import (
    AgentDecision,
    ArtifactRef,
    DatasetBinding,
    ExperimentRecord,
    HandoffRecord,
    ModelSpec,
    PresetRecord,
    ProblemSpec,
    RunConfig,
    SolverPlan,
    TaskSpec,
    ValidationReport,
)

DEFAULT_PROVENANCE_DB_PATH = "provenance.duckdb"

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY,
        title TEXT,
        description TEXT,
        problem_kind TEXT,
        physics_domain TEXT,
        equation_family TEXT,
        review_required BOOLEAN,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS problem_specs (
        task_id TEXT,
        kind TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS model_specs (
        task_id TEXT,
        kind TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS solver_plans (
        task_id TEXT,
        kind TEXT,
        learning_rate DOUBLE,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS run_configs (
        task_id TEXT,
        max_epochs INTEGER,
        accelerator TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS dataset_bindings (
        task_id TEXT,
        name TEXT,
        source TEXT,
        location TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS artifacts (
        uri TEXT PRIMARY KEY,
        task_id TEXT,
        kind TEXT,
        label TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        task_id TEXT,
        run_id TEXT,
        status TEXT,
        created_at TIMESTAMP,
        finished_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS metrics (
        experiment_id TEXT,
        run_id TEXT,
        name TEXT,
        value DOUBLE,
        step INTEGER DEFAULT 0,
        created_at TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS agent_decisions (
        decision_id TEXT PRIMARY KEY,
        task_id TEXT,
        run_id TEXT,
        agent TEXT,
        tool TEXT,
        input_schema TEXT,
        output_schema TEXT,
        summary TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS handoff_records (
        handoff_id TEXT PRIMARY KEY,
        task_id TEXT,
        run_id TEXT,
        from_agent TEXT,
        to_agent TEXT,
        reason TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS validation_reports (
        report_id TEXT PRIMARY KEY,
        task_id TEXT,
        run_id TEXT,
        verdict TEXT,
        rationale TEXT,
        created_at TIMESTAMP,
        payload JSON
    )""",
    """CREATE TABLE IF NOT EXISTS lineage_edges (
        from_uri TEXT,
        to_uri TEXT,
        relation TEXT,
        created_at TIMESTAMP
    )""",
    # Preset catalog — one table per family so queries like
    # "active problem presets tagged 'incompressible'" stay simple.
    """CREATE TABLE IF NOT EXISTS preset_problems (
        name TEXT PRIMARY KEY,
        version INTEGER,
        builder_ref TEXT,
        spec_json JSON,
        description TEXT,
        tags JSON,
        parent_name TEXT,
        author TEXT,
        status TEXT,
        created_at TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS preset_models (
        name TEXT PRIMARY KEY,
        version INTEGER,
        builder_ref TEXT,
        spec_json JSON,
        description TEXT,
        tags JSON,
        parent_name TEXT,
        author TEXT,
        status TEXT,
        created_at TIMESTAMP
    )""",
    """CREATE TABLE IF NOT EXISTS preset_solvers (
        name TEXT PRIMARY KEY,
        version INTEGER,
        builder_ref TEXT,
        spec_json JSON,
        description TEXT,
        tags JSON,
        parent_name TEXT,
        author TEXT,
        status TEXT,
        created_at TIMESTAMP
    )""",
)


_PRESET_TABLE_BY_FAMILY: dict[str, str] = {
    "problem": "preset_problems",
    "model": "preset_models",
    "solver": "preset_solvers",
}


class ProvenanceStore:
    """DuckDB-backed index over the agents' typed records.

    Open one store per ``FlowDeps`` instance — DuckDB is a single-process
    database and sharing a connection inside the same process is fine,
    but a second process cannot open the same file for write. The store
    is created lazily (see ``FlowDeps.provenance()``) so tests that
    don't exercise provenance never touch disk.

    Use ``:memory:`` as the path for ephemeral tests.
    """

    def __init__(self, db_path: str | Path = DEFAULT_PROVENANCE_DB_PATH) -> None:
        self.db_path = str(db_path)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        for stmt in _SCHEMA_STATEMENTS:
            self.conn.execute(stmt)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> ProvenanceStore:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def query(self, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        """Read-only convenience wrapper. Returns a list of row dicts."""
        rel = self.conn.execute(sql, params or [])
        if rel.description is None:
            return []
        cols = [d[0] for d in rel.description]
        return [dict(zip(cols, row, strict=False)) for row in rel.fetchall()]

    # --- writers ---

    def record_task(self, task: TaskSpec) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO tasks
               (task_id, title, description, problem_kind, physics_domain,
                equation_family, review_required, created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                task.task_id,
                task.title,
                task.description,
                task.problem_kind,
                task.physics_domain,
                task.equation_family,
                task.review_required,
                task.created_at,
                task.model_dump_json(),
            ],
        )

    def record_problem_spec(self, task_id: str, spec: ProblemSpec) -> None:
        # ``kind`` column now tracks the ProblemSpec.name (the composition
        # label) — useful as a grouping key without hardcoding a PDE enum.
        self.conn.execute(
            """INSERT INTO problem_specs (task_id, kind, created_at, payload)
               VALUES (?, ?, ?, ?)""",
            [
                task_id,
                spec.name or "composed",
                datetime.now(UTC),
                spec.model_dump_json(),
            ],
        )

    def record_model_spec(self, task_id: str, spec: ModelSpec) -> None:
        self.conn.execute(
            """INSERT INTO model_specs (task_id, kind, created_at, payload)
               VALUES (?, ?, ?, ?)""",
            [task_id, spec.kind, datetime.now(UTC), spec.model_dump_json()],
        )

    def record_solver_plan(self, task_id: str, plan: SolverPlan) -> None:
        self.conn.execute(
            """INSERT INTO solver_plans
               (task_id, kind, learning_rate, created_at, payload)
               VALUES (?, ?, ?, ?, ?)""",
            [
                task_id,
                plan.kind,
                plan.learning_rate,
                datetime.now(UTC),
                plan.model_dump_json(),
            ],
        )

    def record_run_config(self, task_id: str, cfg: RunConfig) -> None:
        self.conn.execute(
            """INSERT INTO run_configs
               (task_id, max_epochs, accelerator, created_at, payload)
               VALUES (?, ?, ?, ?, ?)""",
            [
                task_id,
                cfg.max_epochs,
                cfg.accelerator,
                datetime.now(UTC),
                cfg.model_dump_json(),
            ],
        )

    def record_dataset_binding(self, task_id: str, binding: DatasetBinding) -> None:
        self.conn.execute(
            """INSERT INTO dataset_bindings
               (task_id, name, source, location, created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                task_id,
                binding.name,
                binding.source,
                binding.location,
                datetime.now(UTC),
                binding.model_dump_json(),
            ],
        )

    def record_artifact(self, ref: ArtifactRef, task_id: str | None = None) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO artifacts
               (uri, task_id, kind, label, created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                ref.uri,
                task_id,
                ref.kind,
                ref.label,
                ref.created_at,
                ref.model_dump_json(),
            ],
        )

    def record_experiment(self, exp: ExperimentRecord) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO experiments
               (experiment_id, task_id, run_id, status, created_at,
                finished_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                exp.experiment_id,
                exp.task_id,
                exp.run_id,
                exp.status,
                exp.created_at,
                exp.finished_at,
                exp.model_dump_json(),
            ],
        )

    def record_metric(
        self,
        *,
        experiment_id: str | None,
        run_id: str | None,
        name: str,
        value: float,
        step: int = 0,
    ) -> None:
        self.conn.execute(
            """INSERT INTO metrics
               (experiment_id, run_id, name, value, step, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [experiment_id, run_id, name, float(value), step, datetime.now(UTC)],
        )

    def record_decision(self, decision: AgentDecision) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO agent_decisions
               (decision_id, task_id, run_id, agent, tool, input_schema,
                output_schema, summary, created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                decision.decision_id,
                decision.task_id,
                decision.run_id,
                decision.agent,
                decision.tool,
                decision.input_schema,
                decision.output_schema,
                decision.summary,
                decision.created_at,
                decision.model_dump_json(),
            ],
        )

    def record_handoff(self, handoff: HandoffRecord) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO handoff_records
               (handoff_id, task_id, run_id, from_agent, to_agent, reason,
                created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                handoff.handoff_id,
                handoff.task_id,
                handoff.run_id,
                handoff.from_agent,
                handoff.to_agent,
                handoff.reason,
                handoff.created_at,
                handoff.model_dump_json(),
            ],
        )

    def record_validation_report(self, report: ValidationReport) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO validation_reports
               (report_id, task_id, run_id, verdict, rationale,
                created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [
                report.report_id,
                report.task_id,
                report.run_id,
                report.verdict,
                report.rationale,
                report.created_at,
                report.model_dump_json(),
            ],
        )

    def record_lineage_edge(self, *, from_uri: str, to_uri: str, relation: str) -> None:
        self.conn.execute(
            """INSERT INTO lineage_edges
               (from_uri, to_uri, relation, created_at)
               VALUES (?, ?, ?, ?)""",
            [from_uri, to_uri, relation, datetime.now(UTC)],
        )

    # --- preset catalog ---

    def upsert_preset(self, record: PresetRecord) -> None:
        """Insert-or-replace a PresetRecord in the family-appropriate table."""
        table = _PRESET_TABLE_BY_FAMILY[record.family]
        import json as _json

        self.conn.execute(
            f"""INSERT OR REPLACE INTO {table}
               (name, version, builder_ref, spec_json, description,
                tags, parent_name, author, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                record.name,
                record.version,
                record.builder_ref,
                _json.dumps(record.spec_json),
                record.description,
                _json.dumps(record.tags),
                record.parent_name,
                record.author,
                record.status,
                record.created_at,
            ],
        )

    def list_presets(
        self,
        family: str,
        *,
        status: str | None = "active",
    ) -> list[dict[str, Any]]:
        table = _PRESET_TABLE_BY_FAMILY[family]
        if status is None:
            return self.query(f"SELECT * FROM {table} ORDER BY name")
        return self.query(
            f"SELECT * FROM {table} WHERE status = ? ORDER BY name",
            [status],
        )

    def get_preset(self, family: str, name: str) -> dict[str, Any] | None:
        table = _PRESET_TABLE_BY_FAMILY[family]
        rows = self.query(f"SELECT * FROM {table} WHERE name = ?", [name])
        return rows[0] if rows else None

    def set_preset_status(self, family: str, name: str, status: str) -> None:
        table = _PRESET_TABLE_BY_FAMILY[family]
        self.conn.execute(
            f"UPDATE {table} SET status = ? WHERE name = ?", [status, name]
        )
