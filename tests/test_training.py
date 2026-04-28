"""Tests for marimo_flow.core.training.

Covers `_resolve_default_root_dir`: the helper that pins Lightning's
output directory to either the active MLflow run's artifact dir (local
file stores) or `data/mlflow/lightning/` as a fallback — keeping
`checkpoints/` and `lightning_logs/` out of the working directory.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import mlflow
import pytest

from marimo_flow.core.training import (
    _FALLBACK_LIGHTNING_ROOT,
    _resolve_default_root_dir,
)


def test_resolve_default_root_dir_no_active_run(monkeypatch):
    monkeypatch.setattr(mlflow, "active_run", lambda: None)
    assert _resolve_default_root_dir() == _FALLBACK_LIGHTNING_ROOT


def test_resolve_default_root_dir_local_file_store(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    fake_run = MagicMock()
    fake_run.info.artifact_uri = artifact_dir.as_uri()  # file:///...
    monkeypatch.setattr(mlflow, "active_run", lambda: fake_run)
    resolved = _resolve_default_root_dir()
    assert resolved == artifact_dir


def test_resolve_default_root_dir_remote_tracking_falls_back(monkeypatch):
    """Remote artifact stores can't be used as Lightning's local dir."""
    fake_run = MagicMock()
    fake_run.info.artifact_uri = "s3://bucket/runs/abc"
    monkeypatch.setattr(mlflow, "active_run", lambda: fake_run)
    assert _resolve_default_root_dir() == _FALLBACK_LIGHTNING_ROOT


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path quirk")
def test_resolve_default_root_dir_strips_windows_leading_slash(monkeypatch):
    """`urlparse('file:///C:/foo').path == '/C:/foo'` — leading slash trips Path."""
    fake_run = MagicMock()
    fake_run.info.artifact_uri = "file:///C:/Users/test/artifacts"
    monkeypatch.setattr(mlflow, "active_run", lambda: fake_run)
    resolved = _resolve_default_root_dir()
    # Should be a Windows-shaped path, not /C:/...
    assert str(resolved)[0].isalpha() and ":" in str(resolved)
