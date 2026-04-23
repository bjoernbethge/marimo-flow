"""FlowDeps — in-memory side-channel for live objects + per-role model config.

Model specs are provider-prefixed strings (e.g. ``"ollama:qwen3-coder:480b-cloud"``,
``"anthropic:claude-sonnet-4-6"``, ``"openai:gpt-5"``). Resolution is delegated
to ``pydantic_ai.models.infer_model`` which supports the full pydantic-ai
provider catalogue: openai, anthropic, google-gla, google-vertex, groq, mistral,
cohere, bedrock, huggingface, ollama, deepseek, openrouter, vercel, azure,
cerebras, xai, moonshotai, fireworks, together, heroku, github, litellm, nebius,
ovhcloud, alibaba, sambanova, outlines, sentence-transformers, voyageai.

Per-role model specs can be customised three ways (highest wins):

1. Env var ``MARIMO_FLOW_MODEL_<ROLE>=<provider>:<model>``
2. YAML file — first of these that exists at CWD is used:
   ``$MARIMO_FLOW_CONFIG`` → ``config.yaml`` → ``config.yml`` →
   ``marimo-flow.yaml`` → ``marimo-flow.yml``
3. ``DEFAULT_MODELS`` (Ollama Cloud defaults shipped with the repo).

Provider auth uses each provider's standard env var (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GROQ_API_KEY``, ``OLLAMA_BASE_URL``, ...). The yaml
file may also carry an ``env:`` block to seed env vars without exporting them
in every shell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic_ai.models import Model, infer_model

if TYPE_CHECKING:
    from marimo_flow.agents.state import FlowState

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_MODELS: dict[str, str] = {
    "route":    "ollama:gemma4:31b-cloud",
    "notebook": "ollama:qwen3-coder:480b-cloud",
    "problem":  "ollama:qwen3-coder:480b-cloud",
    "model":    "ollama:qwen3.5:cloud",
    "solver":   "ollama:qwen3-coder:480b-cloud",
    "training": "ollama:qwen3-coder:480b-cloud",
    "mlflow":   "ollama:gpt-oss:20b-cloud",
    "lead":     "ollama:kimi-k2.5:cloud",
}

_CONFIG_CANDIDATES = (
    "config.yaml",
    "config.yml",
    "marimo-flow.yaml",
    "marimo-flow.yml",
)


def _config_path() -> Path | None:
    env = os.environ.get("MARIMO_FLOW_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    for name in _CONFIG_CANDIDATES:
        p = Path(name)
        if p.is_file():
            return p
    return None


def load_config() -> dict[str, Any]:
    """Parse the marimo-flow yaml config if present. Empty dict otherwise."""
    p = _config_path()
    if p is None:
        return {}
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _apply_env_block(config: dict[str, Any]) -> None:
    """Seed os.environ from config['env'] without overriding existing values."""
    env_block = config.get("env") or {}
    for key, value in env_block.items():
        if key not in os.environ and value is not None:
            os.environ[key] = str(value)


def resolve_models(config: dict[str, Any] | None = None) -> dict[str, str]:
    """Role → spec map, overlaying DEFAULT_MODELS with config and env vars."""
    if config is None:
        config = load_config()
    _apply_env_block(config)
    models = dict(DEFAULT_MODELS)
    yaml_models = config.get("models") or {}
    if isinstance(yaml_models, dict):
        models.update({str(k): str(v) for k, v in yaml_models.items()})
    for role in list(models):
        env_key = f"MARIMO_FLOW_MODEL_{role.upper()}"
        if env_key in os.environ:
            models[role] = os.environ[env_key]
    return models


def _ensure_ollama_base_url() -> None:
    """pydantic_ai.OllamaProvider reads OLLAMA_BASE_URL; seed a local default."""
    if "OLLAMA_BASE_URL" not in os.environ:
        os.environ["OLLAMA_BASE_URL"] = DEFAULT_OLLAMA_BASE_URL


def get_model(role: str, *, override: str | None = None) -> Model:
    """Resolve a role to a pydantic-ai ``Model``.

    ``override`` is a spec string that short-circuits the role lookup,
    typically passed as ``override=ctx.deps.models[role]`` from a Node so
    each FlowDeps instance can hold its own per-run model assignment.
    """
    _ensure_ollama_base_url()
    spec = override or resolve_models()[role]
    return infer_model(spec)


@dataclass
class FlowDeps:
    """In-memory deps. Not persisted — recreated per session.

    ``registry`` maps MLflow artifact URI → live object (PINA Problem, torch
    model, Trainer). ``FlowState`` only holds the URIs.

    ``state`` is wired in by each Node before calling
    ``agent.run(..., deps=ctx.deps)`` so FunctionToolset tools can reach
    the current FlowState via ``ctx.deps.state``.

    ``models`` is populated from ``resolve_models()``  (DEFAULT_MODELS overlaid
    by ``config.yaml`` and ``MARIMO_FLOW_MODEL_<ROLE>`` env vars). You can
    override an individual role at runtime with
    ``deps.models[role] = "<provider>:<model>"``.
    """

    models: dict[str, str] = field(default_factory=resolve_models)
    registry: dict[str, Any] = field(default_factory=dict)
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    marimo_mcp_url: str = "http://127.0.0.1:2718/mcp/server"
    state: FlowState | None = None
