# Marimo Flow 🌊


[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Marimo](https://img.shields.io/badge/Marimo-Latest-orange?logo=python&logoColor=white)](https://marimo.io)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue?logo=mlflow&logoColor=white)](https://mlflow.org)
[![MCP](https://img.shields.io/badge/MCP-Enabled-green?logo=anthropic&logoColor=white)](https://docs.marimo.io/guides/editor_features/mcp/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![Version](https://img.shields.io/badge/Version-0.3.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/synapticore-io/marimo-flow)
[![Contributing](https://img.shields.io/badge/Contributing-Welcome-brightgreen.svg)](CONTRIBUTING.md)

---

*Like marimo algae drifting in crystal waters, your data flows and evolves – each cell a living sphere of computation, gently touching others, creating ripples of reactive change. In this digital ocean, data streams like currents, models grow like organic formations, and insights emerge naturally from the depths. Let your ML experiments flow freely, tracked and nurtured, as nature intended.*



<div align="center">

https://github.com/user-attachments/assets/3bc24463-ff42-44a7-ae61-5d500d29688c



</div>


## Why Marimo Flow is Powerful 🚀

**Marimo Flow** combines reactive notebook development with AI-powered assistance and robust ML experiment tracking:

- **🤖 AI-First Development with MCP**: Model Context Protocol (MCP) integration brings live documentation, code examples, and AI assistance directly into your notebooks - access up-to-date library docs for Marimo, Polars, Plotly, and more without leaving your workflow
- **🔄 Reactive Execution**: Marimo's dataflow graph ensures your notebooks are always consistent - change a parameter and watch your entire pipeline update automatically
- **📊 Seamless ML Pipeline**: MLflow integration tracks every experiment, model, and metric without breaking your flow
- **🎯 Interactive Development**: Real-time parameter tuning with instant feedback and beautiful visualizations

This combination eliminates the reproducibility issues of traditional notebooks while providing AI-enhanced, enterprise-grade experiment tracking.

## Features ✨

### 🤖 AI-Powered Development (MCP)
- **Model Context Protocol Integration**: Live documentation and AI assistance in your notebooks
- **Context7 Server**: Access up-to-date docs for any Python library without leaving marimo
- **Marimo MCP Server**: Specialized assistance for marimo patterns and best practices
- **Local LLM Support**: Ollama integration for privacy-focused AI code completion

### 📊 ML Development Workflow
- **📓 Reactive Notebooks**: Git-friendly `.py` notebooks with automatic dependency tracking
- **🔬 MLflow Tracking**: Complete ML lifecycle management with model registry
- **🎯 Interactive Development**: Real-time parameter tuning with instant visual feedback
- **💾 SQLite Backend**: Lightweight, file-based storage for experiments

### 🐳 Deployment
- **Docker**: docker-compose setup with CPU, CUDA, and XPU image variants
- **🧠 PINA Integration**: Physics-informed neural networks with Walrus foundation model
- **📚 MCP-Powered Docs**: Live documentation via Context7 and Marimo MCP servers

## Quick Start 🏃‍♂️

### With Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/synapticore-io/marimo-flow.git
cd marimo-flow

# Build and start services
docker compose -f docker/docker-compose.yaml up --build -d

# Access services
# Marimo: http://localhost:2718
# MLflow: http://localhost:5000

# View logs
docker compose -f docker/docker-compose.yaml logs -f

# Stop services
docker compose -f docker/docker-compose.yaml down
```

#### Docker Image Variants

| Variant | Image Tag | Use Case |
|---------|-----------|----------|
| **CPU** | `ghcr.io/synapticore-io/marimo-flow:latest` | No GPU (lightweight) |
| **CUDA** | `ghcr.io/synapticore-io/marimo-flow:cuda` | NVIDIA GPUs |
| **XPU** | `ghcr.io/synapticore-io/marimo-flow:xpu` | Intel Arc/Data Center GPUs |

```bash
# NVIDIA GPU (requires nvidia-docker)
docker compose -f docker/docker-compose.cuda.yaml up -d

# Intel GPU (requires Intel GPU drivers)
docker compose -f docker/docker-compose.xpu.yaml up -d
```

### Local Development

```bash
# Install dependencies
uv sync

# Start MLflow server (in background or separate terminal)
uv run mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root ./data/mlflow/artifacts \
  --serve-artifacts

# Start Marimo (in another terminal)
uv run marimo edit examples/
```

## Example Notebooks 📚

All notebooks live in `examples/` and can be opened with `uv run marimo edit examples/<file>.py`.

- **`01_pina_poisson_solver.py`** – Solve the Poisson equation with baseline PINNs or the Walrus foundation model. Training is tracked in MLflow with integrated Optuna sweep analytics and experiment history. Uses `marimo_flow.core` directly.
- **`lab.py`** – PINA multi-agent team chat demo (see [PINA Multi-Agent Team](#pina-multi-agent-team-marimo_flowagents-)). Requires Ollama running locally.

## Project Structure 📁

```
marimo-flow/
├── examples/                    # Marimo notebooks
│   ├── 01_pina_poisson_solver.py   # Poisson PINN demo (uses core/)
│   └── lab.py                      # PINA team chat demo (uses agents/)
├── src/marimo_flow/             # Installable package
│   ├── core/                    # PINA solvers, training, visualization
│   └── agents/                  # Multi-agent team (pydantic-graph + MLflow)
├── tests/                       # Pytest suite (48 passing, 1 xfailed)
├── docs/                        # Project documentation (see docs/INDEX.md)
├── docker/                      # Dockerfiles + compose (CPU, CUDA, XPU)
├── data/mlflow/                 # MLflow storage (artifacts, db)
└── pyproject.toml               # Dependencies
```

### Two Workflows

| Workflow | Import | Use Case |
|----------|--------|----------|
| **Classic** (`core/`) | `from marimo_flow.core import ...` | You know the PDE, pick a solver, log to MLflow. See `examples/01_pina_poisson_solver.py`. |
| **Agents** (`agents/`) | `from marimo_flow.agents import lead_chat, FlowDeps` | Describe the problem in natural language; a multi-agent team composes Problem + Model + Solver. See `examples/lab.py`. |

Both write to the same MLflow backend (`data/mlflow/`). The two packages do not depend on each other — pick whichever matches the task.

### The `marimo_flow.core` Package

```python
from marimo_flow.core import (
    ProblemManager,           # Define PDE problems and domains
    SolverManager,            # Configure PINN / SAPINN solvers
    FoundationModelAdapter,   # Walrus foundation model adapter
    create_model_for_problem, # Build a PINA neural-network model for a Problem
    train_solver,             # Run training via PINA's Trainer
    build_optuna_history_figure,
    build_optuna_param_importance_figure,
    build_optuna_parallel_figure,
    build_trials_scatter_chart,
    study_trials_dataframe,
)
```

## MCP (Model Context Protocol) Integration 🔌

marimo and AI-assisted IDEs share MCP servers for live documentation and notebook operations. For the full configuration reference see [`docs/mcp-setup.md`](docs/mcp-setup.md).

### marimo (in-notebook AI)

Pre-configured in `.marimo.toml`:

```toml
[mcp]
presets = ["marimo", "context7"]

[mcp.mcpServers.mlflow]
command = "mlflow"
args = ["mcp", "run"]

[mcp.mcpServers.mlflow.env]
MLFLOW_TRACKING_URI = "http://localhost:5000"

[ai.ollama]
model = "gpt-oss:20b-cloud"
base_url = "http://localhost:11434/v1"
```

The Docker container uses a separate `docker/.marimo.toml` without MCP presets — containerized sessions run only the notebook UI; MCP servers run on the host and are reached over `host.docker.internal`.

### VS Code / Claude Code

Four MCP servers configured in `.vscode/mcp.json`:

| Server | Purpose |
|--------|---------|
| **marimo** | Notebook inspection, linting (HTTP on port 2718) |
| **mlflow** | Trace search, feedback, evaluation (stdio via `mlflow mcp run`) |
| **context7** | Live library documentation (stdio via npx) |
| **serena** | Semantic code search (stdio via uvx) |

Start the marimo MCP server (required for the `marimo` tool):

```bash
uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"
marimo edit --mcp --no-token --port 2718 --headless
```

### Claude Code skills & hooks

Three domain skills in `.claude/Skills/` (`marimo`, `mlflow`, `pina`) provide expert guidance and pre-resolved context7 library IDs (`/marimo-team/marimo`, `/mlflow/mlflow`, `/mathlab/pina`).

Automated cross-platform hooks in `.claude/settings.json`:

| Hook | Trigger | Action |
|------|---------|--------|
| **PostToolUse** | Edit/Write `.py` files | Auto-format with ruff |
| **PreToolUse** | Edit `uv.lock` | Block (protection) |

## Configuration ⚙️

### Environment Variables

Docker setup (configured in `docker/docker-compose.yaml`):
- `MLFLOW_BACKEND_STORE_URI`: `sqlite:////app/data/mlflow/db/mlflow.db`
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: `/app/data/mlflow/artifacts`
- `MLFLOW_HOST`: `0.0.0.0` (allows external access)
- `MLFLOW_PORT`: `5000`
- `OLLAMA_BASE_URL`: `http://host.docker.internal:11434` (requires Ollama on host)

Local development:
- `MLFLOW_TRACKING_URI`: `http://localhost:5000` (default)

### Docker Services

The Docker container runs both services via `docker/start.sh`:
- **Marimo**: Port 2718 - Interactive notebook environment
- **MLflow**: Port 5000 - Experiment tracking UI

**GPU Support**: Use `docker-compose.cuda.yaml` for NVIDIA GPUs or `docker-compose.xpu.yaml` for Intel GPUs. The default `docker-compose.yaml` is CPU-only.

## PINA Multi-Agent Team (`marimo_flow.agents`) 🧑‍🚀🧑‍🚀🧑‍🚀

Reactive multi-agent team that orchestrates PINA workflows via `pydantic-graph`,
backed by MLflow for tracing + persistence, exposed via marimo's chat UI and
optionally as A2A and AG-UI ASGI servers.

```python
from marimo_flow.agents import lead_chat, FlowDeps
import marimo as mo

deps = FlowDeps(mlflow_tracking_uri="sqlite:///mlruns.db")
chat = mo.ui.chat(lead_chat(deps=deps))
chat
```

**Roles** (each loads its `.claude/Skills/<name>/SKILL.md` as `instructions=`):
- `notebook` — marimo MCP cell ops (skills: `marimo`, `marimo-pair`)
- `problem` — defines a PINA Problem from an open spec (skill: `pina`)
- `model` — designs a neural architecture for the problem (skill: `pina`)
- `solver` — wires Solver + Trainer config (skill: `pina`)
- `mlflow` — MLflow MCP tracking + registry (skill: `mlflow`)

A `RouteNode` classifier dispatches between sub-nodes; the lead agent wraps the
graph as a single tool so the same backend powers marimo chat, A2A, and AG-UI.

**Models:** provider-prefixed specs (`"<provider>:<model>"`) resolved through
pydantic-ai's `infer_model`. Defaults in
`marimo_flow.agents.deps.DEFAULT_MODELS` all point at Ollama Cloud
(`http://localhost:11434/v1`, `:cloud`-suffixed tags).

Override per role either via `config.yaml` at the repo root
(see `config.yaml.example`) or with `MARIMO_FLOW_MODEL_<ROLE>=<spec>`
env vars. Any provider in the pydantic-ai catalogue works — openai,
anthropic, groq, mistral, google-gla, bedrock, together, fireworks,
openrouter, deepseek, cerebras, xai, ollama, huggingface, ...

**Standalone servers:**

```bash
uv run python -m marimo_flow.agents.server.a2a    # A2A on :8000
uv run python -m marimo_flow.agents.server.ag_ui  # AG-UI on :8001
```

See `examples/lab.py` for the full demo notebook.

## Contributing 🤝

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Development setup and workflow
- Code standards and style guide
- Testing requirements
- Pull request process

**Quick Start for Contributors:**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the [coding standards](CONTRIBUTING.md#code-standards)
4. Test your changes: `uv run pytest`
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for comprehensive guidelines.

## Changelog 📋

See [CHANGELOG.md](CHANGELOG.md) for a detailed version history and release notes.

**Current Version:** 0.3.0

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Marimo and MLflow**
