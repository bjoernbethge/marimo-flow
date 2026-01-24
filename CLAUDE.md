# CLAUDE.md

## Project: marimo-flow

Reactive Python notebooks with MLflow tracking and PINA physics-informed neural networks.

## Commands

- `uv sync` - Install dependencies
- `marimo edit examples/<name>.py` - Open notebook
- `mlflow ui` - Start MLflow dashboard
- `ruff format . && ruff check --fix .` - Format/lint

## MCP Servers

- **marimo**: `marimo edit --mcp --no-token --port 2718 --headless`
- **mlflow**: `mlflow mcp run` (set MLFLOW_TRACKING_URI)
- VS Code config: `.vscode/mcp.json` (uses `servers` key, not `mcpServers`)
- Claude Code config: `.mcp.json` (uses `mcpServers` key)

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
