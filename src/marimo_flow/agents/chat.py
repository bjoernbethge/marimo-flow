"""marimo chat adapter — bridges mo.ui.chat to the lead agent's stream."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def lead_chat(*, model=None, deps: FlowDeps | None = None) -> Callable:
    """Return an async-generator function suitable for `mo.ui.chat(...)`.

    The returned callable has the signature `(messages, config) -> AsyncIterator[str]`
    expected by marimo's chat component. Yields delta chunks (not accumulated text)
    so streaming bandwidth scales with new tokens, not total length.
    """
    agent = build_lead_agent(model=model, deps=deps)

    async def _chat(messages, config) -> AsyncIterator[str]:  # noqa: ARG001
        user_text = messages[-1].content
        async with agent.run_stream(user_text) as run:
            previous = ""
            async for accumulated in run.stream_output():
                delta = accumulated[len(previous) :]
                if delta:
                    yield delta
                previous = accumulated

    return _chat
