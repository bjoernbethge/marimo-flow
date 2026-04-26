"""Tests for MLflowStatePersistence — uses MLflow file:// store in tmp dir."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@pytest.mark.asyncio
async def test_persistence_writes_artifact_files_to_disk(tmp_mlflow):
    """Regression test for [Errno 22] on Windows.

    pydantic_graph.nodes.generate_snapshot_id returns
    ``'{NodeName}:{uuid4().hex}'``. The colon is reserved on NTFS, so
    until commit 79e4251 ``client.log_artifact`` crashed with
    ``[Errno 22] Invalid argument`` the moment the lead chat ran on
    Windows with a local-filesystem MLflow store.
    """
    persistence = MLflowStatePersistence(run_id=tmp_mlflow)
    state = FlowState()
    node = StartNode()
    snap_id = node.get_snapshot_id()
    # Sanity: pydantic-graph really does inject a colon — that's the bug
    # vector. If pydantic-graph ever changes its ID format, drop this test
    # but keep _safe_filename as a defence-in-depth.
    assert ":" in snap_id

    # Drives both code paths in _log_state:
    #   - snapshot_node_if_new → _log_state(f"node-{snap_id}")  (colon!)
    #   - super().snapshot_node → _log_state(f"node-{Class.__name__}")
    await persistence.snapshot_node_if_new(snap_id, state, node)
    await persistence.snapshot_end(state, End("done"))

    # Resolve the on-disk artifact directory from the active run and
    # check what actually got written.
    run = mlflow.get_run(tmp_mlflow)
    artifact_uri = run.info.artifact_uri
    assert artifact_uri.startswith("file:")
    artifact_root = Path(artifact_uri[len("file:///") :]) / "agent_state"
    files = sorted(p.name for p in artifact_root.iterdir())
    assert files, "no artifacts written"
    # Strict: no Windows-reserved char ever leaks onto disk.
    for f in files:
        assert not any(c in f for c in '<>:"/\\|?*'), f"reserved char in {f}"
    # The colon-bearing snap_id must be present in sanitized form.
    assert any(snap_id.replace(":", "_") in f for f in files), files


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
