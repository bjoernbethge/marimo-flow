"""Tests for the lead agent wrapper."""

from __future__ import annotations

from pydantic_ai.models.test import TestModel

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


async def test_lead_agent_exposes_run_pina_workflow_tool(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    deps = FlowDeps()
    agent = build_lead_agent(model=TestModel(), deps=deps)
    # pydantic-ai 1.84: registered function tools live on agent._function_toolset.tools
    tool_names = list(agent._function_toolset.tools.keys())  # noqa: SLF001
    assert "run_pina_workflow" in tool_names


async def test_lead_agent_skips_workflow_when_model_responds_directly(monkeypatch):
    """If the LLM produces a text reply (not a tool call), the workflow stays untouched."""
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    deps = FlowDeps()
    agent = build_lead_agent(
        model=TestModel(custom_output_text="hi, no workflow needed", call_tools=[]),
        deps=deps,
    )
    result = await agent.run("just say hi")
    assert "hi" in result.output.lower()
