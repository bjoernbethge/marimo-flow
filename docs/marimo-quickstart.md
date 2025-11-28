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

## AI-Human Collaboration Patterns

### Pattern: Interactive AI Agent with User Validation

Marimo is ideal for human-AI collaboration workflows where humans guide AI reasoning and validate outputs:

```python
import marimo as mo
from pydantic_ai import Agent

# Initialize AI agent
analysis_agent = Agent(
    model="openai:gpt-4",
    system_prompt="You are a data analyst. Provide clear, step-by-step analysis."
)

# User input for analysis request
analysis_request = mo.ui.text_area(
    label="What would you like to analyze?",
    placeholder="Describe your analysis goal..."
)

# Run analysis with approval
run_button = mo.ui.button(label="Analyze", on_change=lambda _: None)

# AI generates analysis
if run_button.value and analysis_request.value:
    result = analysis_agent.run_sync(analysis_request.value)
    mo.md(f"## AI Analysis\n\n{result.data}")

    # Human validation
    approve = mo.ui.checkbox(label="Approve this analysis?")

    if approve.value:
        mo.md("✅ Analysis approved and saved")
    else:
        mo.md("⚠️ Awaiting revisions...")
```

### Pattern: Collaborative Notebook with Comments

```python
import marimo as mo
from datetime import datetime

# Initialize collaboration state
get_comments, set_comments = mo.state([])
get_approvals, set_approvals = mo.state({})

# User identity
user_name = mo.ui.text(label="Your name:", value="Analyst 1")

# Analysis cell
analysis_result = mo.md("""
# Data Quality Report
- Records: 10,000
- Missing values: 2.3%
- Duplicates: 0
""")

# Comment section
comment_input = mo.ui.text_area(
    label="Add a comment:",
    placeholder="Share thoughts or concerns..."
)

def add_comment(_):
    if comment_input.value:
        comment = {
            "author": user_name.value,
            "text": comment_input.value,
            "timestamp": datetime.now().isoformat()
        }
        set_comments([*get_comments(), comment])

comment_button = mo.ui.button(label="Add Comment", on_change=add_comment)

# Display comments
comments_display = mo.vstack([
    mo.md(f"**{c['author']}** ({c['timestamp']})"),
    mo.md(c['text'])
] for c in get_comments())

# Approval tracking
def toggle_approval(_):
    approvals = dict(get_approvals())
    approvals[user_name.value] = not approvals.get(user_name.value, False)
    set_approvals(approvals)

approve_button = mo.ui.button(label="Approve", on_change=toggle_approval)

# Workflow display
mo.vstack([
    analysis_result,
    mo.md("---"),
    mo.md("## Discussion"),
    comment_input,
    comment_button,
    comments_display,
    mo.md("---"),
    mo.md(f"## Approvals: {sum(get_approvals().values())}/{len(get_approvals())}"),
    approve_button
])
```

### Pattern: AI-Assisted Data Exploration

```python
import marimo as mo
import polars as pl
from pydantic_ai import Agent

# Load data
df = pl.read_csv("data.csv")

# Exploration agent
explore_agent = Agent(
    model="openai:gpt-4",
    system_prompt="Analyze data profiles and suggest interesting patterns to investigate."
)

# User selects column
column = mo.ui.dropdown(df.columns, label="Select column:")

# AI generates insights
if column.value:
    stats = df[column.value].describe()
    prompt = f"Column: {column.value}\nStatistics:\n{stats}"

    insights = explore_agent.run_sync(prompt)
    mo.md(f"## AI Insights\n\n{insights.data}")

    # Human exploration control
    drill_down = mo.ui.dropdown(
        ["Distribution", "Outliers", "Relationships", "Anomalies"],
        label="Explore:",
    )

    if drill_down.value:
        follow_up = explore_agent.run_sync(f"Focus on: {drill_down.value}")
        mo.md(f"## Deep Dive\n\n{follow_up.data}")
```

### Pattern: Collaborative Model Evaluation

```python
import marimo as mo
from datetime import datetime

# Model comparison state
get_model_scores, set_model_scores = mo.state({})
get_reviewer_notes, set_reviewer_notes = mo.state([])

# Reviewer information
reviewer = mo.ui.text(label="Reviewer name:", value="")

# Model selection
models = ["Model A (v1.2)", "Model B (v2.0)", "Model C (experimental)"]
selected_model = mo.ui.dropdown(models, label="Evaluate model:")

# Scoring interface
accuracy = mo.ui.slider(0, 1, step=0.01, label="Accuracy score:")
latency = mo.ui.slider(0, 5000, step=100, label="Latency (ms):")
robustness = mo.ui.dropdown(
    ["Poor", "Fair", "Good", "Excellent"],
    label="Robustness:"
)

# Save evaluation
def save_evaluation(_):
    if selected_model.value and reviewer.value:
        score_key = f"{selected_model.value}_{reviewer.value}"
        scores = dict(get_model_scores())
        scores[score_key] = {
            "accuracy": accuracy.value,
            "latency": latency.value,
            "robustness": robustness.value,
            "timestamp": datetime.now().isoformat()
        }
        set_model_scores(scores)

save_button = mo.ui.button(label="Save Evaluation", on_change=save_evaluation)

# Notes
notes = mo.ui.text_area(label="Notes:", placeholder="Add observations...")

def add_notes(_):
    if notes.value and reviewer.value:
        note = {
            "author": reviewer.value,
            "model": selected_model.value,
            "text": notes.value,
            "time": datetime.now().isoformat()
        }
        set_reviewer_notes([*get_reviewer_notes(), note])

notes_button = mo.ui.button(label="Add Notes", on_change=add_notes)

# Display aggregated results
scores_summary = mo.md(f"""
## Evaluation Summary

{len(get_model_scores())} evaluations recorded

Models evaluated: {len(set(k.split('_')[0] for k in get_model_scores().keys()))}
Reviewers: {len(set(k.split('_')[1] for k in get_model_scores().keys()))}
""")

# Layout
mo.vstack([
    mo.md("## Model Evaluation Form"),
    selected_model,
    reviewer,
    accuracy,
    latency,
    robustness,
    save_button,
    mo.md("---"),
    notes,
    notes_button,
    mo.md("---"),
    scores_summary
])
```

### Pattern: Asynchronous Review Workflow

```python
import marimo as mo
from enum import Enum
from datetime import datetime

class ReviewStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"

# Workflow state
get_document, set_document = mo.state({
    "title": "Q4 Analysis Report",
    "status": ReviewStatus.DRAFT,
    "author": "Data Team",
    "reviewers": [],
    "created_at": datetime.now().isoformat()
})

# Author view
mo.md("## Report Submission")
submit_button = mo.ui.button(label="Submit for Review", on_change=lambda _: None)

if submit_button.value:
    doc = dict(get_document())
    doc["status"] = ReviewStatus.SUBMITTED
    set_document(doc)
    mo.md("✅ Report submitted for review")

# Display status
doc = get_document()
mo.md(f"""
### Document Status
- Title: {doc['title']}
- Author: {doc['author']}
- Status: **{doc['status'].upper()}**
- Created: {doc['created_at']}
""")

# Reviewer view (would be in separate app or user context)
if doc["status"] in [ReviewStatus.SUBMITTED, ReviewStatus.IN_REVIEW]:
    reviewer_decision = mo.ui.radio(
        [ReviewStatus.APPROVED, ReviewStatus.REJECTED],
        label="Review decision:"
    )
    feedback = mo.ui.text_area(label="Feedback:")

    review_button = mo.ui.button(label="Submit Review", on_change=lambda _: None)
```

## Collaborative Notebook Best Practices

### ✅ DO: Structure for Multiple Contributors

```python
# GOOD - clear sections for different contributors
# ===== DATA PREPARATION (by: Data Engineer) =====
data = load_data()

# ===== EXPLORATORY ANALYSIS (by: Analyst) =====
mo.md("## Data Overview")
summary_stats = data.describe()

# ===== MODEL DEVELOPMENT (by: Data Scientist) =====
model = train_model(data)

# ===== VALIDATION (by: QA Reviewer) =====
metrics = evaluate_model(model)
```

### ✅ DO: Use State for Shared Results

```python
# GOOD - state allows multiple cells to read/modify shared data
get_results, set_results = mo.state({})

# Different cells can update results
def update_result(key, value):
    results = dict(get_results())
    results[key] = value
    set_results(results)

# All changes are tracked and reactive
mo.md(f"Results: {get_results()}")
```

### ❌ DON'T: Use Notebook for Real-time Editing

```python
# BAD - marimo is not a real-time collaborative editor
# Use version control + code review workflow instead
# Each change = commit to git
```

### ✅ DO: Maintain Clear Approvals

```python
# GOOD - explicit approval tracking
get_approvals, set_approvals = mo.state({
    "data_quality": False,
    "methodology": False,
    "results": False
})

def approve_section(section):
    approvals = dict(get_approvals())
    approvals[section] = True
    set_approvals(approvals)

all_approved = all(get_approvals().values())
mo.md(f"All sections approved: {all_approved}")
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
mo.ui.text_area(placeholder="", label="", value="")
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

## Debugging & Development

### Common Patterns for AI Collaboration

**1. Error Handling in AI Workflows**
```python
import marimo as mo
from pydantic_ai import Agent

try:
    result = agent.run_sync(prompt)
    mo.md(f"✅ {result.data}")
except Exception as e:
    mo.md(f"❌ Error: {str(e)}")
    mo.md("Please try again with a different prompt")
```

**2. Progress Tracking**
```python
# Use state to track progress
get_progress, set_progress = mo.state(0)

# Update progress in long-running operations
for i in range(10):
    process_item(i)
    set_progress(i + 1)

mo.md(f"Progress: {get_progress()}/10")
```

**3. Conditional Rendering Based on Approval**
```python
# Show/hide sections based on approval state
get_approved, set_approved = mo.state(False)

approve_btn = mo.ui.button("Approve", on_change=lambda _: set_approved(True))

if get_approved():
    mo.md("## Approved Results")
    # Show sensitive data only after approval
else:
    approve_btn
```

## Additional Resources

- **Official Docs**: https://docs.marimo.io
- **GitHub**: https://github.com/marimo-team/marimo
- **Examples**: https://github.com/marimo-team/marimo/tree/main/examples
- **Discord Community**: https://discord.gg/marimo
- **Pydantic AI Docs**: https://ai.pydantic.dev

## Migration Notes

### From Jupyter Notebooks

- Use `marimo convert notebook.ipynb` for automatic conversion
- Refactor cells to have single variable definitions
- Replace interactive widgets with marimo UI elements
- Remove cell ordering dependencies (marimo handles this)

### Python Version

- Requires Python 3.9+
- Uses modern type hints and f-strings

## Integration with marimo-flow

Marimo works seamlessly with the marimo-flow ecosystem:

- **MLflow Integration**: Track experiments with `mlflow` module (see `mlflow-quickstart.md`)
- **Polars Data Processing**: Use with Polars for efficient data manipulation
- **PINA Models**: Integrate physics-informed neural networks
- **OpenVINO Inference**: Deploy optimized models using OpenVINO

For more details, see the integration patterns documentation.
