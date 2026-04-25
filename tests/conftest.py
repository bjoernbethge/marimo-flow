"""Global pytest fixtures.

The autouse fixtures here make the suite independent of:

* ``.env`` files in the working tree
* ``config.yaml`` / ``marimo-flow.yaml`` at the repo root
* User-set ``MARIMO_FLOW_*`` / provider auth env vars

Individual tests may still opt into a specific environment with ``monkeypatch``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def _mlflow_session_db(tmp_path_factory):
    """One SQLite tracking DB shared across the whole test session.

    Per-fixture file backends emit a FutureWarning ("filesystem tracking
    backend is deprecated as of Feb 2026"). Switching to SQLite per-test
    runs Alembic migrations on every fresh DB and turns a 27 s suite into
    a 113 s suite. A session-scoped DB lets Alembic run once.
    """
    db = tmp_path_factory.mktemp("mlflow") / "tracking.db"
    return f"sqlite:///{db.as_posix()}"


@pytest.fixture(autouse=True)
def _isolate_marimo_flow_config(monkeypatch, tmp_path, _mlflow_session_db):
    """Run every test in a scratch CWD with no config/env spill-in.

    We chdir into ``tmp_path`` so the config resolver can't find a
    repo-local ``config.yaml``/``.env`` and wipe the env vars that would
    otherwise steer ``DEFAULT_MODELS``. Tests that need specific config
    opt back in via ``monkeypatch.chdir(...)`` or ``monkeypatch.setenv(...)``.
    """
    monkeypatch.chdir(tmp_path)
    for var in (
        "MARIMO_FLOW_CONFIG",
        "MARIMO_FLOW_DOTENV",
        "MARIMO_MCP_URL",
        "MLFLOW_PYDANTIC_AI_AUTOLOG",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _mlflow_session_db)
    # Per-role model overrides inspected by resolve_models.
    for role in (
        "ROUTE",
        "NOTEBOOK",
        "PROBLEM",
        "MODEL",
        "SOLVER",
        "TRAINING",
        "MLFLOW",
        "LEAD",
    ):
        monkeypatch.delenv(f"MARIMO_FLOW_MODEL_{role}", raising=False)
    # Reset the dotenv-once flag so the next test can load its own .env.
    import marimo_flow.agents.deps as _deps

    monkeypatch.setattr(_deps, "_DOTENV_LOADED", False, raising=False)
