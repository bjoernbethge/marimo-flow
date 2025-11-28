# Marimo Reference - Quickstart & Core Concepts

**Last Updated**: 2025-11-21
**Source Version**: Latest (marimo-team/marimo)
**Status**: Current

## What is Marimo?

Marimo is an open-source reactive Python notebook that keeps code and outputs consistent. It's designed as a modern replacement for Jupyter notebooks with a focus on:
- **Reproducibility**: Code and outputs stay in sync
- **Git-friendly**: Notebooks are plain Python files
- **Deployable**: Export as web apps or static HTML
- **Reactive**: Cells automatically re-execute when dependencies change

Key differences from Jupyter:
- Uses Python syntax with marimo-specific UI elements
- Automatic dependency tracking and reactive updates
- Built-in state management without callbacks
- Seamless deployment and sharing

## Quick Reference

### Installation & Setup

```bash
# Basic installation
pip install marimo

# Install with recommended dependencies (DuckDB, Altair, Polars, OpenAI, etc.)
pip install "marimo[recommended]"

# Using uv
uv add marimo

# Using conda
conda install -c conda-forge marimo
```

### Basic Commands

```bash
# Create/edit a notebook
marimo edit

# Create/edit a specific notebook file
marimo edit my_notebook.py

# Run notebook as web app (hides code)
marimo run my_notebook.py

# Run as Python script
python my_notebook.py

# Run tutorial
marimo tutorial intro

# Export to other formats
marimo export my_notebook.py --html -o output.html
marimo export my_notebook.py --ipynb -o output.ipynb

# Show config location
marimo config show
```

## Core Concepts

### 1. Reactive Execution Model

Marimo automatically re-executes cells when their dependencies change. Unlike Jupyter, this is guaranteed and prevents stale outputs.

```python
import marimo as mo

# Cell 1: Define a value
x = 10

# Cell 2: Uses x
y = x * 2
mo.md(f"y equals {y}")

# Cell 3: Changes x - cells 2 and 3 automatically re-run
x = 20
```

**Key Points:**
- Each variable has one definition location
- All cells using that variable re-execute
- No manual "run all" needed
- Dependency graph is implicit in variable names

### 2. State Management with `mo.state()`

For mutable state that persists across cell executions, use `mo.state()`:

```python
import marimo as mo

# Initialize state - returns (getter, setter)
get_counter, set_counter = mo.state(0)
```

Reading state:
```python
# Read the current value
current_value = get_counter()
mo.md(f"Counter: {current_value}")
```

Updating state:
```python
# Update with direct value
set_counter(5)

# Update with function (receives current value)
set_counter(lambda count: count + 1)
```

State is reactive - any cell that calls `get_counter()` will re-run when state changes.

**Allow Self Loops** - By default, the cell setting state won't re-run when state changes. Enable re-runs:
```python
get_count, set_count = mo.state(0, allow_self_loops=True)
```

### 3. UI Elements

Marimo provides interactive UI components:

```python
import marimo as mo

# Slider
slider = mo.ui.slider(1, 10, label="Select a number")
mo.md(f"Slider value: {slider.value}")

# Button with callback
button = mo.ui.button(
    label="Click me",
    on_change=lambda _: print("Button clicked")
)

# Text input
text_input = mo.ui.text(placeholder="Enter text")
mo.md(f"Input: {text_input.value}")

# Dropdown/Select
dropdown = mo.ui.dropdown(["Option A", "Option B", "Option C"], label="Choose")
mo.md(f"Selected: {dropdown.value}")

# Radio buttons
radio = mo.ui.radio(["Yes", "No"], label="Answer?")

# Checkbox
checkbox = mo.ui.checkbox(label="Agree")

# Form
form = mo.ui.form(
    {"name": mo.ui.text(label="Name"), "age": mo.ui.slider(1, 100, label="Age")},
    submit_button_label="Submit"
)
```

### 4. Layout & Composition

```python
import marimo as mo

# Stack elements horizontally
mo.hstack([widget1, widget2, widget3], justify="center")

# Stack elements vertically
mo.vstack([widget1, widget2], gap="2")

# Tabs
mo.tabs({
    "Tab 1": content1,
    "Tab 2": content2,
})

# Expander/Accordion
mo.accordion({"Section 1": content1, "Section 2": content2})
```

### 5. Markdown & Display

```python
import marimo as mo

# Markdown with interpolation
mo.md(f"""
# Title
This is **bold** and this is *italic*.
Value is {variable}
""")

# Code blocks
mo.md("""
```python
print("code block")
```
""")

# LaTeX
mo.md(r"$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$")
```

## Common Patterns

### Pattern: Counter with Buttons

```python
import marimo as mo

# State
get_count, set_count = mo.state(0)

# UI
increment = mo.ui.button(
    label="Increment",
    on_change=lambda _: set_count(lambda v: v + 1)
)
decrement = mo.ui.button(
    label="Decrement",
    on_change=lambda _: set_count(lambda v: v - 1)
)

# Display
mo.hstack([decrement, increment], justify="center")
mo.md(f"## Counter: {get_count()}")
```

### Pattern: Form with Reactive Updates

```python
import marimo as mo

# Form inputs
name_input = mo.ui.text(placeholder="Name", label="Your name:")
age_slider = mo.ui.slider(1, 100, label="Age:")

# Reactive display using form values
if name_input.value:
    mo.md(f"Hello {name_input.value}! You are {age_slider.value} years old.")
else:
    mo.md("Please enter your name")
```

### Pattern: Todo List with State

```python
import marimo as mo
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    done: bool = False

# State for tasks
get_tasks, set_tasks = mo.state([])
get_input, set_input = mo.state("")

# Input field
task_input = mo.ui.text(placeholder="New task", label="Add task:")

# Add task button
def add_task(_):
    if task_input.value:
        new_task = Task(name=task_input.value)
        set_tasks([*get_tasks(), new_task])
        set_input("")

add_button = mo.ui.button(label="Add", on_change=add_task)

# Display
mo.md("## Todo List")
mo.vstack([
    mo.hstack([task_input, add_button]),
    mo.md("\n".join([f"- {task.name}" for task in get_tasks()]))
])
```

### Pattern: Notebook with Data Integration

```python
import marimo as mo
import polars as pl

# Load data
data = pl.read_csv("data.csv")

# Show data info
mo.md(f"Data shape: {data.shape}")
mo.md(f"Columns: {data.columns}")

# Interactive filtering
column_selector = mo.ui.dropdown(data.columns, label="Filter by:")
value_input = mo.ui.text(placeholder="Value", label="Value:")

# Reactive filtered view
if column_selector.value and value_input.value:
    filtered = data.filter(pl.col(column_selector.value) == value_input.value)
    mo.md(f"Filtered rows: {filtered.shape[0]}")
```

## Best Practices

### ✅ DO: One Variable Definition Per Notebook

Define each variable in exactly one cell. Marimo enforces this for reproducibility.

```python
# GOOD: Single definition
x = 10

# Later cell
y = x * 2
```

### ❌ DON'T: Redefine Variables

```python
# BAD: Multiple definitions break reactivity
x = 10  # Cell 1

# ...

x = 20  # Cell 2 - causes issues
```

### ✅ DO: Use UI Element .value Attribute

```python
# GOOD
slider = mo.ui.slider(1, 10)
result = slider.value * 2
```

### ❌ DON'T: Reference UI Elements Directly

```python
# BAD - UI elements are interactive objects
slider = mo.ui.slider(1, 10)
result = slider * 2  # Wrong!
```

### ✅ DO: Structure UI with mo.hstack/mo.vstack

```python
# GOOD - organized layout
controls = mo.hstack([slider, button], justify="center")
output = mo.md(f"Result: {slider.value}")
mo.vstack([controls, output])
```

### ✅ DO: Use State for Mutable Data

```python
# GOOD - for reactive mutable state
get_items, set_items = mo.state([])

# Add new item
set_items([*get_items(), new_item])
```

### ✅ DO: Create Reusable Functions

```python
# GOOD - helper functions for complex logic
def calculate_sum(items):
    return sum(items)

result = calculate_sum(get_items())
```

### ❌ DON'T: Use Global State Outside mo.state()

```python
# BAD - won't be reactive
items = []  # Don't do this for mutable data

# GOOD
get_items, set_items = mo.state([])
```

## Common Issues & Solutions

### Issue: Cell Won't Re-execute

**Problem**: Cell doesn't re-run even though a dependency changed.

**Solution**: Check that the variable is actually used. Marimo only re-executes cells that reference changed variables.

```python
# Make sure dependency is referenced
x = 10
y = x * 2  # This references x, so it will re-run when x changes
```

### Issue: State Not Updating in Display

**Problem**: `get_state()` returns old value.

**Solution**: Call `get_state()` in the cell that displays it, not elsewhere.

```python
# Good - display cell calls getter
mo.md(f"Value: {get_count()}")

# Bad - won't update
value = get_count()
mo.md(f"Value: {value}")
```

### Issue: UI Elements Not Responsive

**Problem**: Changing slider/input doesn't trigger updates.

**Solution**: Reference the UI element's value directly in dependent cells, not stored variables.

```python
# Good - directly reference .value
slider = mo.ui.slider(1, 10)
result = slider.value * 2

# Bad - storing value breaks reactivity
slider = mo.ui.slider(1, 10)
stored_value = slider.value  # Stored at cell run time
result = stored_value * 2     # Won't update when slider changes
```

## Deployment & Export

### Export to HTML

```bash
marimo export my_notebook.py --html -o index.html
```

### Deploy as Web App

```bash
# Run as interactive web app (code visible but not editable)
marimo run my_notebook.py

# Run with custom host/port
marimo run my_notebook.py --host 0.0.0.0 --port 8080
```

### Run as Python Script

```bash
# Execute notebook as regular Python script
python my_notebook.py

# Pass command-line arguments
python my_notebook.py --arg value
```

### Convert from Jupyter

```bash
# Convert .ipynb to marimo
marimo convert notebook.ipynb -o notebook.py

# Convert jupytext py:percent to marimo
marimo convert script.py -o notebook.py
```

## Integration Patterns

### With Polars DataFrames

```python
import marimo as mo
import polars as pl

# Load data
df = pl.read_csv("data.csv")

# Interactive exploration
mo.md(f"Shape: {df.shape}")
mo.md(f"Columns: {list(df.columns)}")

# Filter with UI
column = mo.ui.dropdown(df.columns, label="Column:")
value = mo.ui.text(label="Value:")

if column.value and value.value:
    filtered = df.filter(pl.col(column.value) == value.value)
    mo.md(f"Filtered: {filtered.shape[0]} rows")
```

### With Plotly Visualizations

```python
import marimo as mo
import plotly.express as px
import polars as pl

df = pl.read_csv("data.csv")

# Interactive chart
x_col = mo.ui.dropdown(df.columns, label="X axis:")
y_col = mo.ui.dropdown(df.columns, label="Y axis:")

if x_col.value and y_col.value:
    fig = px.scatter(
        df.to_pandas(),
        x=x_col.value,
        y=y_col.value
    )
    fig.show()
```

### With DuckDB SQL

```python
import marimo as mo
import duckdb

# SQL cells are first-class citizens
sql_query = mo.sql("""
    SELECT * FROM 'data.csv'
    LIMIT 10
""")

# Result is available as a table
mo.md(f"Query returned {len(sql_query)} rows")
```

## API Reference - Quick Lookup

### State Management

```python
mo.state(initial_value, allow_self_loops=False)
# Returns: (getter_func, setter_func)
# getter_func() -> value
# setter_func(value | lambda: new_value)
```

### UI Components

```python
mo.ui.slider(start, stop, step=1, label="")
mo.ui.button(label="", on_change=callback)
mo.ui.text(placeholder="", label="", value="")
mo.ui.dropdown(options, label="", value=None)
mo.ui.radio(options, label="", value=None)
mo.ui.checkbox(label="", value=False)
mo.ui.form(fields_dict, submit_button_label="Submit")
```

### Layout

```python
mo.hstack(elements, justify="start", gap="2")
mo.vstack(elements, justify="start", gap="2")
mo.tabs(tab_dict)  # {"Tab 1": content1, ...}
mo.accordion(section_dict)  # {"Section": content, ...}
```

### Display

```python
mo.md(markdown_string)  # Markdown with f-string interpolation
mo.html(html_string)    # Raw HTML
mo.image(url_or_path)   # Image display
mo.matshow(array)       # Matrix/array visualization
mo.spinner()            # Loading spinner
```

## Additional Resources

- **Official Docs**: https://docs.marimo.io
- **GitHub**: https://github.com/marimo-team/marimo
- **Examples**: https://github.com/marimo-team/marimo/tree/main/examples
- **Discord Community**: https://discord.gg/marimo

## Migration Notes

### From Jupyter Notebooks

- Use `marimo convert notebook.ipynb` for automatic conversion
- Refactor cells to have single variable definitions
- Replace interactive widgets with marimo UI elements
- Remove cell ordering dependencies (marimo handles this)

### Python Version

- Requires Python 3.9+
- Uses modern type hints and f-strings
