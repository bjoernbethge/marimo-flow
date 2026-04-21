"""MLflowStatePersistence — pydantic-graph snapshot backend backed by MLflow.

Snapshots are written as JSON artifacts under the active MLflow run.
Live composition is handled by the in-memory parent backend; we override
loading/saving so that resuming a chat = resuming an MLflow run.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path

import mlflow
from pydantic_graph import BaseNode, End
from pydantic_graph.persistence.in_mem import FullStatePersistence


class MLflowStatePersistence(FullStatePersistence):
    """JSON snapshots logged as MLflow artifacts under `agent_state/`."""

    ARTIFACT_DIR = "agent_state"

    def __init__(
        self, *, run_id: str, client: mlflow.MlflowClient | None = None
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.client = client or mlflow.MlflowClient()

    def _to_jsonable(self, state) -> dict:
        return asdict(state) if is_dataclass(state) else dict(state)

    def _log_state(self, label: str, state) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{label}.json"
            path.write_text(json.dumps(self._to_jsonable(state), default=str, indent=2))
            self.client.log_artifact(
                self.run_id, str(path), artifact_path=self.ARTIFACT_DIR
            )

    async def snapshot_node(self, state, next_node: BaseNode) -> None:
        await super().snapshot_node(state, next_node)
        self._log_state(f"node-{next_node.__class__.__name__}", state)

    async def snapshot_node_if_new(
        self, snapshot_id: str, state, next_node: BaseNode
    ) -> None:
        await super().snapshot_node_if_new(snapshot_id, state, next_node)
        self._log_state(f"node-{snapshot_id}", state)

    async def snapshot_end(self, state, end: End) -> None:
        await super().snapshot_end(state, end)
        self._log_state("end", state)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        async with super().record_run(snapshot_id):
            yield
