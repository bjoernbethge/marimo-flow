# Marimo Flow 🌊


[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Marimo](https://img.shields.io/badge/Marimo-Latest-orange?logo=python&logoColor=white)](https://marimo.io)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue?logo=mlflow&logoColor=white)](https://mlflow.org)
[![MCP](https://img.shields.io/badge/MCP-Enabled-green?logo=anthropic&logoColor=white)](https://docs.marimo.io/guides/editor_features/mcp/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![Version](https://img.shields.io/badge/Version-0.2.0-blue.svg)](CHANGELOG.md)
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
  --backend-store-uri sqlite:///data/experiments/db/mlflow.db \
  --default-artifact-root ./data/experiments/artifacts \
  --serve-artifacts

# Start Marimo (in another terminal)
uv run marimo edit examples/
```

## Example Notebooks 📚

All notebooks live in `examples/` and can be opened with `uv run marimo edit examples/<file>.py`.

- **`01_pina_poisson_solver.py`** – Solve the Poisson equation with baseline PINNs or the Walrus foundation model. Training is tracked in MLflow with integrated Optuna sweep analytics and experiment history.

## Project Structure 📁

```
marimo-flow/
├── examples/                    # Marimo notebooks
│   └── 01_pina_poisson_solver.py
├── src/marimo_flow/             # Installable package
│   └── core/                    # PINA solvers, training, visualization
├── docs/                        # Project documentation
├── docker/                      # Dockerfiles + compose (CPU, CUDA, XPU)
├── data/mlflow/                 # MLflow storage (artifacts, db)
└── pyproject.toml               # Dependencies
```

### The `marimo_flow` Package

```python
from marimo_flow.core import (
    ModelFactory,        # Create PINA neural network models
    ProblemManager,      # Define PDE problems and domains
    SolverManager,       # Configure PINN / SAPINN solvers
    WalrusAdapter,       # Walrus foundation model adapter
    build_optuna_history_figure,
    build_optuna_param_importance_figure,
    build_optuna_parallel_figure,
)
```

## MCP (Model Context Protocol) Integration 🔌

**Marimo Flow is AI-first** with built-in Model Context Protocol (MCP) support for intelligent, context-aware development assistance.

### Why MCP Matters

Traditional notebooks require constant context-switching to documentation sites. With MCP:
- 📚 **Live Documentation**: Access up-to-date library docs directly in marimo
- 🤖 **AI Code Completion**: Context-aware suggestions from local LLMs (Ollama)
- 💡 **Smart Assistance**: Ask questions about libraries and get instant, accurate answers
- 🔄 **Always Current**: Documentation updates automatically, no more outdated tutorials

### Pre-Configured MCP Servers

#### Context7 - Universal Library Documentation
Access real-time documentation for **any Python library**:
```python
# Ask: "How do I use polars window functions?"
# Get: Current polars docs, code examples, best practices

# Ask: "Show me plotly 3D scatter plot examples"
# Get: Latest plotly API with working code samples
```

**Supported Libraries:**
- Polars, Pandas, NumPy - Data manipulation
- Plotly, Altair, Matplotlib - Visualization
- Scikit-learn, PyTorch - Machine Learning
- And 1000+ more Python packages

#### Marimo - Specialized Notebook Assistance
Get expert help with marimo-specific patterns:
```python
# Ask: "How do I create a reactive form in marimo?"
# Get: marimo form patterns, state management examples

# Ask: "Show me marimo UI element examples"
# Get: Complete UI component reference with code
```

### Real-World Examples

**Example 1: Learning New Libraries**
```python
# You're exploring polars window functions
# Type: "polars rolling mean example"
# MCP returns: Latest polars docs + working code
df.with_columns(
    pl.col("sales").rolling_mean(window_size=7).alias("7d_avg")
)
```

**Example 2: Debugging**
```python
# Stuck on a plotly error?
# Ask: "Why is my plotly 3D scatter not showing?"
# Get: Common issues, solutions, and corrected code
```

**Example 3: Best Practices**
```python
# Want to optimize code?
# Ask: "Best way to aggregate in polars?"
# Get: Performance tips, lazy evaluation patterns
```

### AI Features Powered by MCP

- **Code Completion**: Context-aware suggestions as you type (Ollama local LLM)
- **Inline Documentation**: Hover over functions for instant docs
- **Smart Refactoring**: AI suggests improvements based on current libraries
- **Interactive Q&A**: Chat with AI about your code using latest docs

### Configuration

MCP servers are pre-configured in `.marimo.toml`:

```toml
[mcp]
presets = ["context7", "marimo"]

[ai.ollama]
model = "gpt-oss:20b-cloud"
base_url = "http://localhost:11434/v1"
```

If you're running inside Docker, the same `mcp` block lives in `docker/.marimo.toml`, so both local and containerized sessions pick up identical presets.

### Adding Custom MCP Servers

You can extend functionality by adding custom MCP servers in `.marimo.toml`:

```toml
[mcp.mcpServers.your-custom-server]
command = "npx"
args = ["-y", "@your-org/your-mcp-server"]
```

### MLflow Trace Server (Optional)

Expose MLflow trace operations to MCP-aware IDEs/assistants (e.g., Claude Desktop, Cursor) by running:

```bash
mlflow mcp run
```

Run the command from an environment where `MLFLOW_TRACKING_URI` (or `MLFLOW_BACKEND_STORE_URI`/`MLFLOW_DEFAULT_ARTIFACT_ROOT`) points at your experiments. The server stays up until interrupted and can be proxied alongside Marimo/MLflow so every tool shares the same MCP context.

**Learn More:**
- [Marimo MCP Guide](https://docs.marimo.io/guides/editor_features/mcp/) - Official MCP documentation
- [Model Context Protocol](https://modelcontextprotocol.io/) - MCP specification and resources

## Claude Code Integration 🤖

**Marimo Flow** includes full Claude Code support with domain-specific skills, MCP servers, and automated hooks.

### Pre-Configured MCP Servers

| Server | Purpose | Config |
|--------|---------|--------|
| **marimo** | Notebook inspection, debugging, linting | HTTP on port 2718 |
| **mlflow** | Trace search, feedback, evaluation | stdio via `mlflow mcp run` |
| **context7** | Live library documentation | stdio via npx |
| **serena** | Semantic code search | stdio via uvx |

**Start marimo MCP server:**
```bash
# Install once (recommended)
uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"

# Start server
marimo edit --mcp --no-token --port 2718 --headless
```

### Domain Skills

Three specialized skills in `.claude/Skills/` provide expert guidance:

| Skill | Triggers | MCP Tools |
|-------|----------|-----------|
| **marimo** | `marimo`, `reactive notebook`, `mo.ui` | Notebook inspection, linting, context7 docs |
| **mlflow** | `mlflow`, `experiment tracking`, `genai tracing` | Trace search, feedback, evaluation, context7 docs |
| **pina** | `pina`, `pinns`, `pde solver`, `neural operator` | MLflow tracking, context7 docs |

**Pre-resolved context7 library IDs** (no lookup needed):
- `/marimo-team/marimo` - marimo docs (2,413 snippets)
- `/mlflow/mlflow` - mlflow docs (9,559 snippets)
- `/mathlab/pina` - PINA docs (2,345 snippets)

### Automated Hooks

Cross-platform hooks in `.claude/settings.json`:

| Hook | Trigger | Action |
|------|---------|--------|
| **SessionStart** | Session begins | Start marimo MCP server |
| **PostToolUse** | Edit/Write `.py` files | Auto-format with ruff |
| **PreToolUse** | Edit `uv.lock` | Block (protection) |

### VS Code Copilot

MCP config for VS Code Copilot in `.vscode/mcp.json`:
```json
{
  "servers": {
    "marimo": { "type": "http", "url": "http://127.0.0.1:2718/mcp/server" },
    "mlflow": { "type": "stdio", "command": "mlflow", "args": ["mcp", "run"] }
  }
}
```

## Configuration ⚙️

### Environment Variables

Docker setup (configured in `docker/docker-compose.yaml`):
- `MLFLOW_BACKEND_STORE_URI`: `sqlite:////app/data/experiments/db/mlflow.db`
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: `/app/data/experiments/artifacts`
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

## API Endpoints 🔌

### MLflow REST API
- **Experiments**: `GET /api/2.0/mlflow/experiments/list`
- **Runs**: `GET /api/2.0/mlflow/runs/search`
- **Models**: `GET /api/2.0/mlflow/registered-models/list`

### Marimo Server
- **Notebooks**: `GET /` - File browser and editor
- **Apps**: `GET /run/<notebook>` - Run notebook as web app

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

**Models:** Ollama Cloud at `http://localhost:11434/v1` (`:cloud`-suffixed tags).
Defaults in `marimo_flow.agents.deps.DEFAULT_MODELS`.

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

**Current Version:** 0.2.0

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Marimo and MLflow**
