"""Tests for MLflowStatePersistence — uses MLflow file:// store in tmp dir."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pytest
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState


@dataclass
class StartNode(BaseNode[FlowState, None, str]):
    async def run(self, ctx: GraphRunContext[FlowState, None]) -> End[str]:
        ctx.state.user_intent = "test-ran"
        return End("done")


@pytest.fixture
def tmp_mlflow():
    mlflow.set_experiment("agents-test")
    with mlflow.start_run() as run:
        yield run.info.run_id


@pytest.mark.asyncio
async def test_persistence_records_run_to_mlflow(tmp_mlflow):
    graph = Graph(nodes=(StartNode,), state_type=FlowState)
    persistence = MLflowStatePersistence(run_id=tmp_mlflow)
    persistence.set_graph_types(graph)
    state = FlowState()
    result = await graph.run(StartNode(), state=state, persistence=persistence)
    assert result.output == "done"
    snapshots = await persistence.load_all()
    assert len(snapshots) >= 1
