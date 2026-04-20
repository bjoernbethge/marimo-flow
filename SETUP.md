# Quick Setup Guide for marimo-flow

Get started with marimo-flow in 5 minutes! This guide covers the essential steps to get your MCP-enabled ML development environment running.

## Prerequisites

- **Python 3.11+**
- **uv** - Fast Python package installer ([Install](https://github.com/astral-sh/uv))
- **Git**

## 1️⃣ Clone & Install

```bash
# Clone repository
git clone https://github.com/bjoernbethge/marimo-flow.git
cd marimo-flow

# Install dependencies (includes marimo[mcp], mlflow[mcp])
uv sync
```

## 2️⃣ Start Services

```bash
# Start all services (MLflow + Marimo with MCP)
./scripts/start-dev.sh
```

This starts:
- **MLflow Server**: http://localhost:5000 (experiment tracking)
- **Marimo Server**: http://localhost:2718 (reactive notebooks)
- **Marimo MCP**: http://localhost:2718/mcp/server (AI assistance)

## 3️⃣ Verify Setup

```bash
# Check if everything is working
./scripts/verify-mcp-setup.sh
```

Should show all green checkmarks ✓

## 4️⃣ Choose Your IDE

### VSCode

Already configured! Just open the project:

```bash
code .
```

**Tasks available** (`Ctrl+Shift+P` → "Tasks: Run Task"):
- Start All Services
- Start MLflow Server
- Start Marimo with MCP
- Run Tests

### Cursor

Already configured with AI rules!

```bash
cursor .
```

**Features**:
- Custom project rules in `.cursorrules`
- AI chat with `Cmd+L` or `Cmd+K`
- MCP-aware suggestions

### Claude Desktop

```bash
# Configure Claude Desktop for MCP
./scripts/setup-claude-desktop.sh
```

Then restart Claude Desktop and try:
```
List active marimo notebooks
Search MLflow experiments
Get docs for polars DataFrame
```

## 5️⃣ Open Your First Notebook

```bash
# Open in browser (auto-opens at http://localhost:2718)
# Or click any .py file in examples/ if using VSCode

# Try these notebooks:
# - examples/01_mlflow_experiment_console.py
# - examples/02_pina_walrus_solver.py
# - examples/03_pina_live_monitoring.py
```

## 🚀 Quick Test

**1. Check MLflow UI**
```bash
open http://localhost:5000
```

**2. Check Marimo UI**
```bash
open http://localhost:2718
```

**3. Test MCP** (in Claude Desktop)
```
List all active marimo notebooks
Show me MLflow experiments
Get polars documentation for lazy evaluation
```

## 📚 What's Next?

### Learn Marimo
- **Reactive Notebooks**: Change one cell → entire notebook updates
- **Git-Friendly**: Notebooks are `.py` files, not `.ipynb`
- **UI Elements**: Interactive sliders, dropdowns, forms
- **Docs**: https://docs.marimo.io

### Learn MLflow
- **Track Experiments**: Log params, metrics, artifacts
- **Model Registry**: Version and deploy models
- **Compare Runs**: Visualize experiment results
- **Docs**: https://mlflow.org/docs

### Explore Examples
- `01_mlflow_experiment_console.py` - MLflow experiment viewer
- `02_pina_walrus_solver.py` - Physics-Informed Neural Networks
- `03_pina_live_monitoring.py` - Live training monitoring

### Use MCP Servers
- **Marimo MCP**: Introspect running notebooks, find errors
- **Context7 MCP**: Get live docs for Python libraries
- **MLflow MCP**: Query experiments, track metrics

## 🔧 Common Commands

```bash
# Start everything
./scripts/start-dev.sh

# Stop everything
./scripts/start-dev.sh --stop

# Verify setup
./scripts/verify-mcp-setup.sh

# Run tests
uv run pytest

# Format code
ruff format . && ruff check --fix .

# Start only MLflow
./scripts/start-dev.sh --mlflow-only

# Start only Marimo
./scripts/start-dev.sh --marimo-only
```

## 🐛 Troubleshooting

### Services won't start?

```bash
# Check if ports are in use
lsof -i :5000  # MLflow
lsof -i :2718  # Marimo

# Kill processes if needed
./scripts/start-dev.sh --stop

# Try again
./scripts/start-dev.sh
```

### MCP not working?

```bash
# Check if marimo started with --mcp flag
ps aux | grep marimo
# Should see: marimo edit examples/ --mcp

# Restart with MCP
pkill -f marimo
marimo edit examples/ --mcp --port 2718

# Test MCP endpoint
curl http://localhost:2718/mcp/server
```

### Dependencies issues?

```bash
# Reinstall dependencies
uv sync --reinstall

# Clear cache
rm -rf .venv
uv sync
```

## 📖 Full Documentation

- **[MCP Setup Guide](docs/mcp-setup.md)** - Complete MCP configuration for all IDEs
- **[README.md](README.md)** - Full project documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines

## 🎯 Environment-Specific Setup

### Local Development
- Configuration: `.marimo.toml`
- Services: `./scripts/start-dev.sh`

### VSCode
- Settings: `.vscode/settings.json`
- Tasks: `.vscode/tasks.json`
- Marimo extension auto-starts

### Cursor
- Settings: `.cursor/settings.json`
- Rules: `.cursorrules`
- AI chat with project context

### Claude Desktop
- Setup: `./scripts/setup-claude-desktop.sh`
- Config: `~/.config/Claude/claude_desktop_config.json` (Linux)
- Config: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

### GitHub Actions
- Workflow: `.github/workflows/claude-code.yml`
- Trigger: `@claude` in issues/PRs
- Secret: `ANTHROPIC_API_KEY` required

## 🌊 Project Philosophy

**marimo-flow** combines three powerful paradigms:

1. **Reactive Development** (Marimo)
   - No hidden state
   - Automatic dependency tracking
   - Instant feedback

2. **Experiment Tracking** (MLflow)
   - Version everything
   - Compare experiments
   - Deploy models

3. **AI-First Development** (MCP)
   - Live documentation
   - Notebook introspection
   - Context-aware assistance

---

**Ready to build?** Start with `./scripts/start-dev.sh` and explore `examples/`!

For detailed MCP setup, see **[docs/mcp-setup.md](docs/mcp-setup.md)**.
