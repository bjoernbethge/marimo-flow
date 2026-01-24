---
name: marimo
description: Interactive reactive Python notebook development with marimo - best practices, UI components, MCP integration, and deployment workflows
triggers:
  - marimo
  - reactive notebook
  - mo.ui
  - marimo app
  - notebook cells
  - interactive dashboard
  - polars dataframe
  - altair chart
allowed_tools:
  - Read
  - Write
  - Edit
  - Bash
  - mcp__marimo__get_marimo_rules
  - mcp__marimo__get_active_notebooks
  - mcp__marimo__get_lightweight_cell_map
  - mcp__marimo__get_cell_runtime_data
  - mcp__marimo__get_cell_outputs
  - mcp__marimo__get_tables_and_variables
  - mcp__marimo__get_database_tables
  - mcp__marimo__get_notebook_errors
  - mcp__marimo__lint_notebook
  - mcp__plugin_context7_context7__resolve-library-id
  - mcp__plugin_context7_context7__query-docs
---

# Marimo Development Skill

Expert guidance for building reactive Python notebooks and applications with marimo.

## MCP Integration Overview

marimo exposes MCP tools for notebook inspection, debugging, and linting.

**Installation:**
```bash
# Install as UV tool (recommended)
uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"
```

**Start MCP Server:**
```bash
marimo edit --mcp --no-token --port 2718 --headless
```

**Available MCP Tools:**
- `get_active_notebooks` - List open notebook sessions
- `get_lightweight_cell_map` - Cell structure preview
- `get_cell_runtime_data` - Cell code, errors, variables
- `get_cell_outputs` - Visual/console output
- `get_tables_and_variables` - DataFrame and variable info
- `get_notebook_errors` - Find all errors
- `lint_notebook` - Check for issues (breaking, runtime, formatting)

**When to use MCP tools:**
- Debugging notebook errors programmatically
- Inspecting cell dependencies and variables
- Validating notebooks before deployment
- Accessing runtime data from running notebooks

**See**: references/mcp_integration.md for complete guide

## Core Principles

### Reactive Programming Model
- Cells automatically re-execute when dependencies change
- Execution order determined by dependency graph, not cell position
- UI element changes trigger automatic updates in dependent cells

### Best Practices
1. **Minimize global variables** - Keep namespace clean
2. **Write idempotent cells** - Same inputs → same outputs
3. **Avoid mutations across cells** - Mutate only in defining cell
4. **Cache expensive computations** - Use `@mo.cache`
5. **Encapsulate in functions** - Avoid polluting global namespace

## Essential UI Elements

### 1. Slider (Numeric Input)
```python
slider = mo.ui.slider(start=0, stop=100, value=50, label="Threshold")
```

### 2. Dropdown (Selection)
```python
dropdown = mo.ui.dropdown(
    options=["Option 1", "Option 2", "Option 3"],
    value="Option 1",
    label="Select"
)
```

### 3. Button (Actions)
```python
button = mo.ui.button(
    label="Click me",
    on_change=lambda v: print(v)
)
```

### 4. Table (Data Display)
```python
table = mo.ui.table(data, sortable=True, filterable=True)
```

### 5. Text Input
```python
text = mo.ui.text(placeholder="Enter name", label="Name")
```

**Critical Rule**: Access `.value` in a DIFFERENT cell than where element is defined.

```python
# Cell 1: Define
slider = mo.ui.slider(0, 100, value=50)
slider

# Cell 2: Use
result = slider.value * 2
```

**See**: references/ui_components.md for complete UI catalog

## Layout Components

```python
# Vertical stack
mo.vstack([element1, element2], gap=1.5)

# Horizontal stack
mo.hstack([element1, element2], gap=2, widths="equal")

# Tabs
mo.tabs({"Tab 1": content1, "Tab 2": content2})
```

## State Management

```python
# Initialize state
get_count, set_count = mo.state(0)

# Update state
set_count(42)  # Set value
set_count(lambda count: count + 1)  # Update function

# Use in buttons
increment = mo.ui.button(
    label="Increment",
    on_change=lambda _: set_count(lambda v: v + 1)
)

# Display
mo.md(f"Count: {get_count()}")
```

**Key Rule**: Always assign state getters to global variables.

**See**: references/state_management.md for advanced patterns

## Simple Reactive Pattern

```python
# Cell 1: Control
n_points = mo.ui.slider(10, 100, value=50, label="Points")
n_points

# Cell 2: Generate data
import numpy as np
x = np.random.rand(n_points.value)
y = np.random.rand(n_points.value)

# Cell 3: Visualize (auto-updates)
import altair as alt
chart = alt.Chart({"x": x, "y": y}).mark_circle().encode(x="x", y="y")
chart
```

**See**: references/reactive_patterns.md for advanced patterns

## Basic App Template

```python
import marimo

app = marimo.App(width="medium", app_title="My App")

@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    return mo, pd

@app.cell
def ui_elements(mo):
    slider = mo.ui.slider(0, 100, value=50, label="Value")
    slider
    return slider,

@app.cell
def use_value(slider, mo):
    result = slider.value * 2
    mo.md(f"Result: **{result}**")
    return result,

if __name__ == "__main__":
    app.run()
```

## Deployment Commands

### Run Locally
```bash
# Development mode
marimo edit notebook.py

# Production mode (read-only)
marimo run notebook.py

# As Python script
python notebook.py
```

### Export Options
```bash
# Static HTML (snapshot)
marimo export html notebook.py -o output.html

# Interactive WASM (runs in browser)
uv run marimo export html-wasm notebook.py -o output.html

# Editable WASM
uv run marimo export html-wasm notebook.py -o output.html --mode edit

# Script
marimo export script notebook.py -o script.py
```

**WASM exports:**
- ✅ Fully interactive, no server needed
- ✅ Deploy to GitHub Pages, Netlify, Vercel
- ⚠️ Limited to Pyodide-compatible packages
- ⚠️ Must be served over HTTP

**See**: references/deployment.md for full deployment guide

## Optimization

```python
# Cache expensive operations
@mo.cache
def expensive_computation(params):
    return results

# Lazy rendering
mo.lazy(mo.ui.table(large_data))

# Memory cleanup
large_object = load_data()
process(large_object)
del large_object
```

## Common Pitfalls

### 1. Accessing UI Value in Same Cell
```python
# ❌ WRONG
slider = mo.ui.slider(0, 100)
result = slider.value  # ERROR

# ✅ CORRECT: Use different cells
```

### 2. Mutating Across Cells
```python
# ❌ WRONG
# Cell 1: numbers = [1, 2, 3]
# Cell 2: numbers.append(4)

# ✅ CORRECT
# Cell 1: numbers = [1, 2, 3]
# Cell 2: extended = numbers + [4]
```

### 3. State Without Assignment
```python
# ❌ WRONG: Local scope
def create(): return mo.state(0)

# ✅ CORRECT: Global scope
get_count, set_count = mo.state(0)
```

## Reference Documentation

**UI Components**: references/ui_components.md
- Complete catalog of UI elements
- Layout components
- Dynamic collections
- Embedding UI in markdown

**Reactive Patterns**: references/reactive_patterns.md
- Task management
- Periodic updates
- Dependent dropdowns
- Data filtering pipelines
- Real-time validation

**State Management**: references/state_management.md
- Basic state usage
- Complex state objects
- Batch updates
- State patterns (undo/redo, loading)
- Debugging state

**Deployment**: references/deployment.md
- Local development
- Export formats (HTML, WASM, script)
- Docker, Kubernetes, cloud deployment
- Performance optimization
- Security considerations

**MCP Integration**: references/mcp_integration.md
- marimo as MCP server
- marimo as MCP client
- marimo-docs tools
- Integration examples
- Security best practices

## Example Templates

**Basic Dashboard**: examples/basic_dashboard.py
- Simple dashboard with UI controls
- Data processing and filtering
- Reactive visualization
- Summary metrics

**Task Manager**: examples/task_manager.py
- State management with tasks
- Button callbacks
- Dynamic UI updates
- Form-like interactions

**Reactive Visualization**: examples/reactive_viz.py
- Interactive scatter plot with selection
- Linked visualizations
- Statistics panel
- Multiple chart types

## Cross-Skill Integration

### MLflow Experiment Console

Build interactive MLflow dashboards:

```python
import marimo as mo
import mlflow

# Experiment selector
experiments = mlflow.search_experiments()
exp_dropdown = mo.ui.dropdown(
    options={e.name: e.experiment_id for e in experiments},
    label="Experiment"
)

# In another cell: display runs
runs = mlflow.search_runs(experiment_ids=[exp_dropdown.value])
mo.ui.table(runs[["run_id", "metrics.loss", "params.learning_rate"]])
```

### PINA Training Dashboard

Interactive physics-informed neural network training:

```python
import marimo as mo
from pina import Trainer
from pina.solver import PINN

# Hyperparameter controls
epochs = mo.ui.slider(100, 5000, value=1000, label="Epochs")
lr = mo.ui.number(value=0.001, label="Learning Rate")
train_btn = mo.ui.run_button(label="Train")

# Training cell
if train_btn.value:
    trainer = Trainer(solver, max_epochs=epochs.value)
    trainer.train()
    mo.output.replace(plot_solution(solver))
```

### Using context7 for Documentation

Query up-to-date marimo documentation directly:

```
# context7 Library IDs (no resolve needed):
# - /marimo-team/marimo (official docs, 2413 snippets)
# - /llmstxt/marimo_io_llms_txt (llms.txt, 7417 snippets)

# Example: query-docs("/marimo-team/marimo", "mo.ui.table parameters")
```

## When to Use This Skill

✅ **Use marimo when:**
- Building interactive data dashboards
- Creating reproducible notebooks
- Deploying notebooks as web apps
- Need automatic reactivity
- Want git-friendly format

❌ **Don't use marimo when:**
- Need traditional Jupyter format
- Require specific Jupyter extensions
- Working with large Jupyter codebases

## Resources

- Documentation: https://docs.marimo.io
- GitHub: https://github.com/marimo-team/marimo
- Examples: https://marimo.io/gallery
