# Reactive Patterns Reference

Advanced patterns for building reactive marimo applications.

## Pattern: Reactive Visualization

Build charts that automatically update when UI controls change.

```python
# Cell 1: UI controls
n_points = mo.ui.slider(10, 100, value=50, label="Points")
n_points

# Cell 2: Generate data based on control
import numpy as np
import polars as pl

x = np.random.rand(n_points.value)
y = np.random.rand(n_points.value)
df = pl.DataFrame({"x": x, "y": y})

# Cell 3: Create chart (auto-updates when slider changes)
import altair as alt

chart = alt.Chart(df).mark_circle(opacity=0.7).encode(
    x=alt.X('x', title='X axis'),
    y=alt.Y('y', title='Y axis')
).properties(
    title=f"Scatter plot with {n_points.value} points",
    width=400,
    height=300
)
chart
```

## Pattern: Task Management

Build interactive task lists with state management.

```python
# State for tasks
from dataclasses import dataclass

@dataclass
class Task:
    name: str
    done: bool = False

get_tasks, set_tasks = mo.state([])

# UI for adding tasks
task_entry = mo.ui.text(placeholder="New task...")

add_button = mo.ui.button(
    label="Add task",
    on_change=lambda _: set_tasks(
        lambda tasks: tasks + [Task(task_entry.value)]
    ) if task_entry.value else None
)

clear_button = mo.ui.button(
    label="Clear completed",
    on_change=lambda _: set_tasks(
        lambda tasks: [t for t in tasks if not t.done]
    )
)

# Display tasks with checkboxes
task_list = mo.ui.array([
    mo.ui.checkbox(value=task.done, label=task.name)
    for task in get_tasks()
], on_change=lambda values: set_tasks(
    lambda tasks: [Task(t.name, done=values[i])
                   for i, t in enumerate(tasks)]
))

# Layout
mo.vstack([
    mo.hstack([task_entry, add_button, clear_button]),
    task_list
])
```

## Pattern: Periodic Updates

Implement auto-refreshing dashboards.

```python
# Run cell on schedule
refresh = mo.ui.refresh(default_interval="5s")
refresh

# This cell runs every 5 seconds
refresh  # Include refresh to trigger
current_time = datetime.now()
mo.md(f"Last update: {current_time}")
```

## Pattern: Gated Execution

Control when expensive operations run.

```python
# Cell 1: Control button
run_button = mo.ui.run_button("Start processing")
run_button

# Cell 2: Conditional execution
mo.stop(not run_button.value, mo.md("Click button to run"))
# Code here only runs after button click
process_data()
```

## Pattern: Reactive Charts with Selection

Create charts that allow data selection.

```python
# Cell 1: Create reactive Altair chart
chart = mo.ui.altair_chart(
    alt.Chart(df).mark_point().encode(
        x='x',
        y='y'
    ).add_params(
        alt.selection_interval()
    )
)
chart

# Cell 2: Use selected data
selected_data = chart.value
mo.md(f"Selected {len(selected_data)} points")
```

## Pattern: Multi-Step Form

Build forms with validation and multi-step workflows.

```python
# Cell 1: Form definition
form_data = mo.ui.dictionary({
    "name": mo.ui.text(placeholder="Full name"),
    "email": mo.ui.text(placeholder="Email", kind="email"),
    "age": mo.ui.number(start=0, stop=150, value=25),
    "terms": mo.ui.checkbox(label="I agree to terms")
}).form()
form_data

# Cell 2: Validation
mo.stop(form_data.value is None, mo.md("Please submit the form"))

data = form_data.value
errors = []

if not data["name"]:
    errors.append("Name is required")
if not data["email"]:
    errors.append("Email is required")
if not data["terms"]:
    errors.append("You must agree to terms")

mo.stop(errors, mo.md("**Errors:**\n" + "\n".join(f"- {e}" for e in errors)))

# Cell 3: Process valid data
mo.md(f"Welcome {data['name']}!")
```

## Pattern: Dependent Dropdowns

Create cascading selection controls.

```python
# Cell 1: Category selection
categories = {
    "Fruits": ["Apple", "Banana", "Orange"],
    "Vegetables": ["Carrot", "Broccoli", "Spinach"],
    "Grains": ["Rice", "Wheat", "Oats"]
}

category = mo.ui.dropdown(
    options=list(categories.keys()),
    label="Category"
)
category

# Cell 2: Item selection based on category
items = mo.ui.dropdown(
    options=categories.get(category.value, []),
    label="Item"
)
items

# Cell 3: Display selection
mo.md(f"Selected: **{category.value}** > **{items.value}**")
```

## Pattern: Progressive Disclosure

Show/hide content based on user interaction.

```python
# Cell 1: Toggle control
show_advanced = mo.ui.checkbox(label="Show advanced options")
show_advanced

# Cell 2: Conditional content
mo.stop(not show_advanced.value)

advanced_options = mo.ui.dictionary({
    "timeout": mo.ui.slider(1, 60, value=30, label="Timeout (s)"),
    "retries": mo.ui.number(0, 10, value=3, label="Retries"),
    "verbose": mo.ui.checkbox(label="Verbose logging")
})
advanced_options
```

## Pattern: Data Filtering Pipeline

Build interactive data exploration tools.

```python
# Cell 1: Load data
import pandas as pd
df = pd.read_csv("data.csv")

# Cell 2: Filter controls
filters = mo.ui.dictionary({
    "category": mo.ui.multiselect(
        options=df["category"].unique().tolist(),
        label="Categories"
    ),
    "min_value": mo.ui.slider(
        start=df["value"].min(),
        stop=df["value"].max(),
        label="Min Value"
    ),
    "search": mo.ui.text(placeholder="Search name...")
})
filters

# Cell 3: Apply filters
filtered_df = df.copy()

if filters.value["category"]:
    filtered_df = filtered_df[
        filtered_df["category"].isin(filters.value["category"])
    ]

filtered_df = filtered_df[
    filtered_df["value"] >= filters.value["min_value"]
]

if filters.value["search"]:
    filtered_df = filtered_df[
        filtered_df["name"].str.contains(
            filters.value["search"],
            case=False
        )
    ]

mo.md(f"Showing {len(filtered_df)} of {len(df)} rows")

# Cell 4: Display results
mo.ui.table(filtered_df)
```

## Pattern: Real-time Validation

Provide immediate feedback on user input.

```python
# Cell 1: Input with validation
email_input = mo.ui.text(placeholder="Enter email", kind="email")
email_input

# Cell 2: Validate and provide feedback
import re

email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
is_valid = bool(re.match(email_pattern, email_input.value))

if email_input.value:
    if is_valid:
        mo.callout(
            mo.md(f"✓ Valid email: {email_input.value}"),
            kind="success"
        )
    else:
        mo.callout(
            mo.md("✗ Invalid email format"),
            kind="danger"
        )
```

## Pattern: State Synchronization

Keep UI elements in sync with state.

```python
# Cell 1: Initialize state
get_value, set_value = mo.state(50)

# Cell 2: UI that updates state
slider = mo.ui.slider(
    0, 100,
    value=get_value(),
    on_change=set_value,
    label="Synchronized value"
)
slider

# Cell 3: Another UI synced to same state
number = mo.ui.number(
    0, 100,
    value=get_value(),
    on_change=set_value,
    label="Same value"
)
number

# Cell 4: Display current state
mo.md(f"Current value: **{get_value()}**")
```

## Pattern: Debounced Updates

Reduce computation for expensive operations.

```python
# Cell 1: Debounced input
search = mo.ui.text(
    placeholder="Search...",
    debounce=True  # Wait for user to stop typing
)
search

# Cell 2: Expensive operation only runs after debounce
import time

mo.stop(not search.value)

# Simulate expensive search
time.sleep(0.5)
results = perform_search(search.value)

mo.md(f"Found {len(results)} results for '{search.value}'")
```
