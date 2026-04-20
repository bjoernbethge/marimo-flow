"""FlowDeps — in-memory side-channel for live objects + per-role model config.

`get_model()` returns a pydantic-ai OpenAIModel pointed at Ollama's
OpenAI-compatible endpoint (default: http://localhost:11434/v1).
Cloud-backed Ollama models use the ':cloud' suffix and are routed by
Ollama itself — no extra proxy needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_MODELS: dict[str, str] = {
    "route": "gemma4-fast:latest",
    "notebook": "qwen3-coder:480b-cloud",
    "problem": "deepseek-v3.2:cloud",
    "model": "qwen3.5:397b-cloud",
    "solver": "qwen3-coder:480b-cloud",
    "mlflow": "gpt-oss:20b-cloud",
    "lead": "kimi-k2.5:cloud",
}


def get_model(role: str, *, override: str | None = None, base_url: str | None = None) -> OpenAIModel:
    name = override or DEFAULT_MODELS[role]
    url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    provider = OpenAIProvider(base_url=url, api_key="ollama")
    return OpenAIModel(name, provider=provider)


@dataclass
class FlowDeps:
    """In-memory deps. Not persisted — recreated per session.

    `registry` maps MLflow artifact URI -> live object (PINA Problem,
    torch model, Trainer). FlowState only holds the URIs.
    """

    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    registry: dict[str, Any] = field(default_factory=dict)
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    marimo_mcp_url: str = "http://127.0.0.1:2718/mcp/server"
