# CLAUDE.md

## Project: marimo-flow

Reactive Python notebooks with MLflow tracking and PINA physics-informed neural networks.

## Commands

- `uv sync` - Install dependencies
- `marimo edit examples/<name>.py` - Open notebook
- `mlflow ui` - Start MLflow dashboard
- `ruff format . && ruff check --fix .` - Format/lint

## MCP Servers

- **marimo**: start manually with `marimo edit --mcp --no-token --port 2718 --headless`
- **mlflow**: `mlflow mcp run` (set `MLFLOW_TRACKING_URI`)
- VS Code config: `.vscode/mcp.json` (uses `servers` key, not `mcpServers`)
- No `.mcp.json` for Claude Code — MCP is wired in some other way here.

## context7 Library IDs

- `/marimo-team/marimo` - marimo docs
- `/mlflow/mlflow` - mlflow docs
- `/mathlab/pina` - PINA docs

## Cross-Platform Hooks

Pattern: `bash -c '...' 2>/dev/null || powershell ...`

## Code Style

- Use `uv` workflow (uv add, uv sync, uv tool install)
- No god classes - split into smaller modules
- No unnecessary wrappers - use core library features directly
- Skills in `.claude/Skills/` are git-tracked

## Agents (`src/marimo_flow/agents/`)

Multi-agent PINA team:
- Orchestration: `pydantic-graph` (Graph + BaseNode + GraphRunContext)
- Persistence + tracing: MLflow (`mlflow.pydantic_ai.autolog()` + `mlflow.pytorch.autolog()`)
- LLMs: Ollama Cloud via `pydantic_ai.models.openai.OpenAIChatModel(base_url=...)`; defaults per role in `agents.deps.DEFAULT_MODELS`
- Each sub-agent loads its skill from `.claude/Skills/<name>/SKILL.md` via `build_skill_instructions()` — lazy, no message-history bloat
- Lead agent (`build_lead_agent`) exposed three ways: `mo.ui.chat`, `agent.to_a2a()`, `agent.to_ag_ui()`
- State (`FlowState`) holds **only** MLflow URIs; live PINA/torch objects live in `FlowDeps.registry`
- Sub-nodes use `_define_*(spec, deps, state)` private helpers — monkeypatch these in tests rather than relying on `TestModel` auto-arg generation
- Tests for nodes that use MCP toolsets (Notebook, MLflow): stub `build_*_mcp` with `pydantic_ai.toolsets.FunctionToolset()` to avoid live server connections, and pin `TestModel(call_tools=[])`
