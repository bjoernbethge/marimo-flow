"""Smoke tests for the marimo-flow CLI."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from marimo_flow.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("solve", "lab", "config-show"):
        assert name in result.output


def test_config_show_prints_defaults(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["models"]["lead"].startswith("ollama:")
    assert payload["mlflow_tracking_uri"].startswith("sqlite:")


def test_config_show_picks_up_env_override(monkeypatch):
    monkeypatch.setenv("MARIMO_FLOW_MODEL_LEAD", "anthropic:claude-sonnet-4-6")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["models"]["lead"] == "anthropic:claude-sonnet-4-6"


def test_solve_rejects_malformed_model_override():
    # "--model lead" without "=" should error out before running.
    result = runner.invoke(app, ["solve", "--model", "lead", "do something"])
    assert result.exit_code == 2
    assert "role=spec" in result.output
