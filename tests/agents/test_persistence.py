"""Tests for MLflowStatePersistence — uses MLflow file:// store in tmp dir."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pytest
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from marimo_flow.agents.persistence import MLflowStatePersistence, _safe_filename
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


def test_safe_filename_strips_windows_reserved_chars():
    # pydantic-graph snapshot IDs use the form "<NodeName>:<uuid4().hex>"
    # — the colon is invalid on NTFS and would crash log_artifact on Windows.
    snap_id = "TriageNode:88d3940e1ff04924bd719c09550efa8e"
    safe = _safe_filename(f"node-{snap_id}")
    assert ":" not in safe
    assert safe == "node-TriageNode_88d3940e1ff04924bd719c09550efa8e"
    # Plain labels stay untouched.
    assert _safe_filename("end") == "end"
    assert _safe_filename("node-TriageNode") == "node-TriageNode"
    # Other reserved chars also get folded.
    assert _safe_filename('a<b>c"d|e?f*g/h\\i') == "a_b_c_d_e_f_g_h_i"
