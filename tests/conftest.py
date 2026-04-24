"""Global pytest fixtures.

The autouse fixtures here make the suite independent of:

* ``.env`` files in the working tree
* ``config.yaml`` / ``marimo-flow.yaml`` at the repo root
* User-set ``MARIMO_FLOW_*`` / provider auth env vars

Individual tests may still opt into a specific environment with ``monkeypatch``.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_marimo_flow_config(monkeypatch, tmp_path):
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
        "MLFLOW_TRACKING_URI",
        "MARIMO_MCP_URL",
        "MLFLOW_PYDANTIC_AI_AUTOLOG",
    ):
        monkeypatch.delenv(var, raising=False)
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
