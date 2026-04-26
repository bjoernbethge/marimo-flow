# Quick Setup Guide for marimo-flow

Get started with marimo-flow in 5 minutes — no shell-scripts, no
platform-specific tooling, just `uv` and a couple of long-running
services.

## Prerequisites

- **Python 3.11+**
- **uv** — Fast Python package installer ([install](https://github.com/astral-sh/uv))
- **Git**

For the full Docker stack (CPU / CUDA / XPU images, single-command
start), see [README → Quick Start with Docker](README.md#with-docker-recommended).
This guide covers the **bare-metal local-dev path** for hacking on the
package itself.

## 1. Install

```bash
git clone https://github.com/synapticore-io/marimo-flow.git
cd marimo-flow
uv sync
```

`uv sync` materialises `.venv/` and installs marimo, mlflow, PINA,
torch, and the agent toolchain. First run takes 1–2 minutes.

## 2. Start the services

You need two long-running processes. Open two terminals (or use a tool
like `tmux` / `pm2` / Windows Terminal tabs).

**Terminal A — MLflow tracking + UI** (`http://localhost:5000`):

```bash
uv run mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root data/mlflow/artifacts \
  --serve-artifacts
```

**Terminal B — marimo notebook UI + MCP** (`http://localhost:2718`):

```bash
uv run marimo edit examples/ --mcp --no-token --headless --port 2718
```

The `--mcp` flag exposes marimo's MCP server at
`http://localhost:2718/mcp/server` so AI-assisted IDEs (Claude Code,
Cursor, etc.) can introspect cells and metrics.

`uv run` activates the project's `.venv/` automatically — no manual
`source .venv/bin/activate` needed.

## 3. Open a notebook

Browse to <http://localhost:2718> and open one of:

- `examples/01_pina_poisson_solver.py` — Poisson PDE with a baseline
  PINN or the Walrus foundation model. MLflow auto-tracks every run.
- `examples/02_provenance_dashboard.py` — DuckDB review surface over
  the agent provenance store + 3D preset preview.
- `examples/03_navier_stokes_3d_cavity.py` — 3D lid-driven cavity
  composed end-to-end from a declarative `ProblemSpec` (no hardcoded
  Navier-Stokes factory).
- `examples/04_mpc_heat_rod.py` — closed-loop MPC on a 1D heat-rod
  PINN surrogate via `marimo_flow.control`.
- `examples/lab.py` — multi-agent PINA team chat (needs Ollama on
  `http://localhost:11434`).

## Common commands

```bash
# Run the test suite (222 tests, ~25 s on CPU)
uv run pytest

# Format + lint
uv run ruff format . && uv run ruff check --fix .

# Build a wheel
uv build

# Stop services: Ctrl+C in each terminal
# Or kill by port:
#   Linux/macOS: lsof -ti :5000 | xargs kill
#   Windows:     netstat -ano | findstr :5000  → taskkill /PID <pid> /F
```

## MCP setup for AI-assisted IDEs

| Tool | Where the config lives | Notes |
|---|---|---|
| **marimo** (in-notebook chat) | `.marimo.toml` | already in the repo, includes `marimo` + `context7` presets and the `mlflow` MCP server. |
| **VS Code / Claude Code** | `.vscode/mcp.json` | already in the repo. Lists `marimo` (HTTP), `mlflow` (stdio), `context7` (stdio), `serena` (stdio). |
| **Cursor / Claude Desktop / others** | their respective config file | use `.vscode/mcp.json` as a reference and adapt for the schema your tool expects. |

For the marimo MCP server to be reachable from external IDEs, terminal
B above must be running with `--mcp`. The MLflow MCP server is started
on demand by the IDE (stdio command: `mlflow mcp run` with
`MLFLOW_TRACKING_URI=http://localhost:5000`).

For details see [`docs/mcp-setup.md`](docs/mcp-setup.md).

## Where data lives

| Path | Contents |
|---|---|
| `data/mlflow/db/mlflow.db` | SQLite tracking DB (experiments, runs, metrics, params) |
| `data/mlflow/artifacts/<run_id>/artifacts/` | logged model artifacts, JSON state snapshots |
| `provenance.duckdb` | DuckDB provenance store for the agent team (override via `MARIMO_FLOW_PROVENANCE_DB`) |

The lead agent (`marimo_flow.agents.lead`) auto-creates
`data/mlflow/{db,artifacts}/` on first import and pins a `marimo-flow`
experiment so artifacts always land under `data/mlflow/artifacts/`,
not the legacy CWD-relative `./mlruns/` fallback.

## Troubleshooting

**MLflow says "filesystem tracking backend is deprecated"** — you're
on a `file://` URI. The SQLite default above silences it.

**Port already in use** — kill the previous process (see "Common
commands") or change `--port`.

**Ollama not found** (when running `examples/lab.py`) — install
[Ollama](https://ollama.com), then `ollama pull qwen2.5:7b` (or any
model from `marimo_flow.agents.deps.DEFAULT_MODELS`).

**`uv sync` is slow** — the first run materialises ~1.5 GB of torch +
Lightning + meshio + transformers. Subsequent runs hit the cache and
finish in seconds.

## What's next

- [README.md](README.md) — full project documentation, Docker variants,
  multi-agent team architecture.
- [docs/roadmap.md](docs/roadmap.md) — phases A-0 → F status with
  pointers to every shipped feature.
- [CLAUDE.md](CLAUDE.md) — guidance for AI agents working in the repo
  (Claude Code, Cursor, etc.).
- [CONTRIBUTING.md](CONTRIBUTING.md) — development workflow, code
  style, test expectations.
