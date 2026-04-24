"""marimo dashboard over the DuckDB provenance store (SPEC §10 / §17.7).

Run with:
    marimo edit examples/02_provenance_dashboard.py

Surfaces task state, agent decisions, experiment outcomes, and
validation verdicts side-by-side so a human can approve/reject runs
without touching MLflow directly.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _header():
    import marimo as mo

    mo.md(
        "# PINA agents — provenance dashboard\n"
        "Reads the DuckDB store written by the multi-agent team. Point "
        "`provenance_db_path` at your local file (default: `provenance.duckdb`)."
    )
    return (mo,)


@app.cell
def _db_picker(mo):
    db_path = mo.ui.text(
        value="provenance.duckdb",
        label="Provenance DuckDB path",
        full_width=True,
    )
    db_path
    return (db_path,)


@app.cell
def _open_store(db_path):
    from marimo_flow.agents.services.provenance import ProvenanceStore

    store = ProvenanceStore(db_path.value)
    return (store,)


@app.cell
def _tasks(mo, store):
    import pandas as pd

    rows = store.query(
        "SELECT task_id, title, problem_kind, equation_family, "
        "review_required, created_at FROM tasks ORDER BY created_at DESC"
    )
    df = pd.DataFrame(rows)
    mo.md("## Tasks")
    mo.ui.table(df) if len(df) else mo.md("_No tasks recorded yet._")
    return (df,)


@app.cell
def _experiments(mo, store):
    import pandas as pd

    rows = store.query(
        "SELECT experiment_id, task_id, run_id, status, created_at, "
        "finished_at FROM experiments ORDER BY created_at DESC"
    )
    df = pd.DataFrame(rows)
    mo.md("## Experiments")
    mo.ui.table(df) if len(df) else mo.md("_No experiments recorded yet._")
    return (df,)


@app.cell
def _decisions(mo, store):
    import pandas as pd

    rows = store.query(
        "SELECT created_at, agent, tool, summary, task_id "
        "FROM agent_decisions ORDER BY created_at DESC LIMIT 100"
    )
    df = pd.DataFrame(rows)
    mo.md("## Recent agent decisions")
    mo.ui.table(df) if len(df) else mo.md("_No decisions recorded yet._")
    return (df,)


@app.cell
def _validation(mo, store):
    import pandas as pd

    rows = store.query(
        "SELECT created_at, task_id, run_id, verdict, rationale "
        "FROM validation_reports ORDER BY created_at DESC LIMIT 50"
    )
    df = pd.DataFrame(rows)
    mo.md("## Validation verdicts")
    mo.ui.table(df) if len(df) else mo.md("_No validation reports yet._")
    return (df,)


@app.cell
def _handoffs(mo, store):
    import pandas as pd

    rows = store.query(
        "SELECT created_at, from_agent, to_agent, reason, task_id "
        "FROM handoff_records ORDER BY created_at DESC LIMIT 50"
    )
    df = pd.DataFrame(rows)
    mo.md("## Handoffs")
    mo.ui.table(df) if len(df) else mo.md("_No handoffs recorded yet._")
    return (df,)


@app.cell
def _preset_family_picker(mo):
    family = mo.ui.dropdown(
        options=["problem", "model", "solver"],
        value="problem",
        label="Preset family",
    )
    include_deprecated = mo.ui.checkbox(
        value=False, label="include deprecated"
    )
    mo.md("## Preset-Bibliothek")
    mo.hstack([family, include_deprecated])
    return (family, include_deprecated)


@app.cell
def _presets(mo, store, family, include_deprecated):
    import pandas as pd

    table_map = {
        "problem": "preset_problems",
        "model": "preset_models",
        "solver": "preset_solvers",
    }
    table = table_map[family.value]
    where = "" if include_deprecated.value else "WHERE status = 'active'"
    rows = store.query(
        f"SELECT name, description, status, author, tags, parent_name, "
        f"       created_at "
        f"FROM {table} {where} "
        f"ORDER BY created_at DESC"
    )
    df = pd.DataFrame(rows)
    if len(df):
        mo.ui.table(df)
    else:
        mo.md(
            f"_No {family.value} compositions yet — they get authored "
            "by the agents as they go._"
        )
    return (df,)


if __name__ == "__main__":
    app.run()
