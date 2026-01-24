# MCP Workflow Integration Guide

Cross-skill workflows combining marimo, mlflow, context7, and playwright MCP servers.

## Available MCP Servers

| MCP Server | Tools | Purpose |
|------------|-------|---------|
| **marimo** | `get_active_notebooks`, `get_cell_runtime_data`, `get_tables_and_variables`, `get_notebook_errors`, `lint_notebook` | Notebook inspection, debugging, variable analysis |
| **mlflow** | `search_traces`, `get_trace`, `log_feedback`, `evaluate_traces`, `list_scorers`, `register_llm_judge` | Experiment tracking, trace analysis, evaluation |
| **context7** | `resolve-library-id`, `query-docs` | Up-to-date library documentation |
| **playwright** | `browser_navigate`, `browser_snapshot`, `browser_click`, `browser_fill_form` | UI testing, screenshots, web automation |

## Combined Workflows

### 1. Notebook Development + Documentation

**Goal**: Build marimo notebooks with up-to-date library knowledge.

```
1. context7: query-docs → Get latest polars/altair/pandas API
2. Edit notebook cells based on current docs
3. marimo: lint_notebook → Verify no issues
4. marimo: get_notebook_errors → Debug if needed
```

**Example:**
```
User: "Add a polars DataFrame transformation"
1. context7.resolve-library-id("polars")
2. context7.query-docs("/pola-rs/polars", "DataFrame group_by aggregation")
3. Edit the notebook with current API
4. marimo.lint_notebook() to verify
```

### 2. Experiment Monitoring Dashboard

**Goal**: Create interactive dashboards that display MLflow experiments.

```
1. mlflow: search_traces → Get experiment data
2. marimo: get_active_notebooks → Find dashboard notebook
3. marimo: get_tables_and_variables → Check current state
4. playwright: browser_snapshot → Capture dashboard view
```

**Example:**
```python
# In marimo notebook:
import mlflow

# Get traces via MCP or API
traces = mlflow.search_traces(experiment_id="1")

# Display in interactive table
mo.ui.table(traces, label="Experiment Traces")
```

### 3. LLM Evaluation Pipeline

**Goal**: Evaluate LLM outputs and log feedback.

```
1. mlflow: search_traces → Find traces to evaluate
2. mlflow: evaluate_traces → Run built-in scorers
3. mlflow: log_feedback → Add human feedback
4. marimo: get_tables_and_variables → Visualize results
```

**Scorers available:**
- `Correctness` - Response accuracy
- `Safety` - Harmful content check
- `RelevanceToQuery` - Addresses user input
- `RetrievalGroundedness` - Aligned with context

### 4. Visual UI Testing

**Goal**: Test marimo app UI with screenshots.

```
1. marimo: get_active_notebooks → Get app URL
2. playwright: browser_navigate → Open app
3. playwright: browser_fill_form → Interact with controls
4. playwright: browser_snapshot → Capture state
5. playwright: browser_take_screenshot → Save evidence
```

### 5. PINA Training with Tracking

**Goal**: Train physics-informed networks with full observability.

```
1. context7: query-docs → Get latest PINA API
2. Start training with MLflow tracking
3. mlflow: search_traces → Monitor progress
4. marimo: get_tables_and_variables → Inspect results
5. mlflow: log_feedback → Annotate interesting runs
```

**Example:**
```python
import mlflow
from pina import Trainer
from pina.solver import PINN

mlflow.set_experiment("pina-poisson")

with mlflow.start_run():
    mlflow.log_params({
        "layers": [64, 64, 64],
        "n_points": 1000,
        "epochs": 1500
    })

    trainer = Trainer(solver, max_epochs=1500)
    trainer.train()

    mlflow.log_metric("final_loss", trainer.callback_metrics["train_loss"])
    mlflow.pytorch.log_model(solver.model, "pinn")
```

## MCP Tool Quick Reference

### marimo MCP Tools

```python
# Get active notebook sessions
mcp__marimo__get_active_notebooks()

# Get cell structure preview
mcp__marimo__get_lightweight_cell_map(session_id, preview_lines=3)

# Get cell code, errors, variables
mcp__marimo__get_cell_runtime_data(session_id, cell_id)

# Get cell visual/console output
mcp__marimo__get_cell_outputs(session_id, cell_id)

# Get DataFrame and variable info
mcp__marimo__get_tables_and_variables(session_id, variable_names=[])

# Get database connection tables
mcp__marimo__get_database_tables(session_id, query=None)

# Find all notebook errors
mcp__marimo__get_notebook_errors(session_id)

# Lint notebook for issues
mcp__marimo__lint_notebook(session_id)
```

### mlflow MCP Tools

```python
# Search traces with filters
mcp__mlflow__search_traces(experiment_id, filter_string=None, max_results=100)

# Get full trace details
mcp__mlflow__get_trace(trace_id)

# Log feedback score
mcp__mlflow__log_feedback(trace_id, name, value, rationale=None)

# Log expected value
mcp__mlflow__log_expectation(trace_id, name, value)

# Evaluate traces with scorers
mcp__mlflow__evaluate_traces(experiment_id, trace_ids, scorers)

# List available scorers
mcp__mlflow__list_scorers(experiment_id=None, builtin=False)

# Create LLM judge scorer
mcp__mlflow__register_llm_judge(name, instructions, experiment_id)
```

### context7 MCP Tools

```python
# Query documentation (use library IDs directly, no resolve needed)
mcp__plugin_context7_context7__query-docs(libraryId, query)

# Pre-resolved Library IDs:
# - /marimo-team/marimo (marimo official, 2413 snippets)
# - /mlflow/mlflow (mlflow official, 9559 snippets)
# - /mathlab/pina (pina official, 2345 snippets)
# - /pola-rs/polars (polars, for dataframes)
# - /vega/altair (altair, for charts)
```

### playwright MCP Tools

```python
# Navigate to URL
mcp__plugin_playwright_playwright__browser_navigate(url)

# Get page snapshot (accessibility tree)
mcp__plugin_playwright_playwright__browser_snapshot()

# Click element
mcp__plugin_playwright_playwright__browser_click(ref, element)

# Fill form fields
mcp__plugin_playwright_playwright__browser_fill_form(fields)

# Take screenshot
mcp__plugin_playwright_playwright__browser_take_screenshot(type="png")
```

## Setup Requirements

### Installation

```bash
# marimo with all features
uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"

# mlflow with MCP
uv tool install "mlflow[mcp]>=3.5.1"
```

### Configuration (.mcp.json)

```json
{
  "mcpServers": {
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp", "run"],
      "env": { "MLFLOW_TRACKING_URI": "sqlite:///mlruns.db" }
    },
    "marimo": {
      "type": "http",
      "url": "http://127.0.0.1:2718/mcp/server"
    }
  }
}
```

### Starting Servers

```bash
# marimo MCP (run in background)
marimo edit --mcp --no-token --port 2718 --headless

# mlflow MCP (starts automatically via Claude Code)
# Or manually: mlflow mcp run
```

## Best Practices

1. **Check MCP status first** - Use `/mcp` in Claude Code to verify connections
2. **Get session IDs** - Always call `get_active_notebooks` before other marimo tools
3. **Use context7 for unfamiliar APIs** - Don't guess, look up current docs
4. **Combine tools intelligently** - Use marimo for data, mlflow for tracking, playwright for visual verification
5. **Log feedback iteratively** - Use mlflow feedback to improve over time
