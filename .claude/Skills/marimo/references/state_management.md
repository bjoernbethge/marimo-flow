# State Management Reference

Complete guide to managing state in marimo applications.

## Understanding State

marimo's reactive model automatically tracks variable dependencies between cells. However, for interactive applications that need to maintain state across user interactions, use `mo.state()`.

## Basic State Usage

### Creating State

```python
# Initialize state with value
get_count, set_count = mo.state(0)

# Access current value
current = get_count()

# Update with new value
set_count(42)

# Update with function (receives current value)
set_count(lambda count: count + 1)
```

### State in Button Handlers

```python
# Initialize counter
get_count, set_count = mo.state(0)

# Increment button
increment = mo.ui.button(
    label="Increment",
    on_change=lambda _: set_count(lambda v: v + 1)
)

# Decrement button
decrement = mo.ui.button(
    label="Decrement",
    on_change=lambda _: set_count(lambda v: v - 1)
)

# Reset button
reset = mo.ui.button(
    label="Reset",
    on_change=lambda _: set_count(0)
)

# Display
mo.vstack([
    mo.hstack([increment, decrement, reset]),
    mo.md(f"Count: **{get_count()}**")
])
```

## State Persistence for UI Elements

Maintain UI element values across cell re-executions.

```python
# Cell 1: Initialize state
get_slider_value, set_slider_value = mo.state(50)

# Cell 2: Create slider with persisted value
slider = mo.ui.slider(
    0, 100,
    value=get_slider_value(),
    on_change=set_slider_value,
    label="Persistent slider"
)
slider

# Cell 3: The value persists even if cell 2 re-runs
mo.md(f"Slider value: {slider.value}")
```

## Self-Looping State

Allow a cell to re-run itself - useful for animations or auto-updating displays.

```python
# Cell 1: Self-looping state
import time

get_timer, set_timer = mo.state(0, allow_self_loops=True)

# Update every second
time.sleep(1)
set_timer(lambda t: t + 1)

mo.md(f"Timer: {get_timer()} seconds")
```

**Warning**: Be careful with self-loops to avoid infinite execution. Always include a stopping condition or rate limiting.

## Complex State Objects

### Using Dataclasses

```python
from dataclasses import dataclass, replace

@dataclass
class AppState:
    username: str = ""
    score: int = 0
    level: int = 1

get_state, set_state = mo.state(AppState())

# Update individual fields
set_state(lambda s: replace(s, score=s.score + 10))

# Display
mo.md(f"""
**User**: {get_state().username}
**Score**: {get_state().score}
**Level**: {get_state().level}
""")
```

### Using Dictionaries

```python
get_config, set_config = mo.state({
    "theme": "dark",
    "language": "en",
    "notifications": True
})

# Update specific key
set_config(lambda cfg: {**cfg, "theme": "light"})

# Access values
current_theme = get_config()["theme"]
```

## State Scope and Lifetime

### Global State (Recommended)

```python
# ✅ CORRECT: State getter assigned to global variable
get_count, set_count = mo.state(0)

# Can be accessed in any cell
mo.md(f"Count: {get_count()}")
```

### Local State (Avoid)

```python
# ❌ WRONG: State created in function
def create_counter():
    get_count, set_count = mo.state(0)
    return get_count, set_count

# State won't work correctly across cells
```

## Batch State Updates

Update multiple states atomically to avoid intermediate renders.

```python
from mo import batch

get_x, set_x = mo.state(0)
get_y, set_y = mo.state(0)

reset_button = mo.ui.button(
    label="Reset coordinates",
    on_change=lambda _: (
        batch(
            lambda: (
                set_x(0),
                set_y(0)
            )
        )
    )
)

mo.md(f"Position: ({get_x()}, {get_y()})")
```

## State with Collections

### List State

```python
get_items, set_items = mo.state([])

# Add item
set_items(lambda items: items + ["new item"])

# Remove item by index
set_items(lambda items: items[:idx] + items[idx+1:])

# Update item
set_items(lambda items: [
    items[i] if i != idx else updated_value
    for i in range(len(items))
])

# Clear all
set_items([])
```

### Set State

```python
get_tags, set_tags = mo.state(set())

# Add tag
set_tags(lambda tags: tags | {"new_tag"})

# Remove tag
set_tags(lambda tags: tags - {"old_tag"})

# Toggle tag
set_tags(lambda tags: (
    tags - {tag} if tag in tags else tags | {tag}
))
```

## State Patterns

### Undo/Redo Pattern

```python
get_history, set_history = mo.state([0])
get_position, set_position = mo.state(0)

def get_current():
    history = get_history()
    pos = get_position()
    return history[pos]

def update_value(new_value):
    history = get_history()
    pos = get_position()
    # Truncate future history
    new_history = history[:pos + 1] + [new_value]
    set_history(new_history)
    set_position(pos + 1)

def undo():
    pos = get_position()
    if pos > 0:
        set_position(pos - 1)

def redo():
    history = get_history()
    pos = get_position()
    if pos < len(history) - 1:
        set_position(pos + 1)

# UI
undo_btn = mo.ui.button(label="Undo", on_change=lambda _: undo())
redo_btn = mo.ui.button(label="Redo", on_change=lambda _: redo())
```

### Loading State Pattern

```python
get_loading, set_loading = mo.state(False)
get_data, set_data = mo.state(None)

async def fetch_data():
    set_loading(True)
    try:
        data = await api.fetch()
        set_data(data)
    finally:
        set_loading(False)

# Display
if get_loading():
    mo.md("Loading...")
elif get_data():
    mo.ui.table(get_data())
else:
    mo.md("No data loaded")
```

### Form State Pattern

```python
get_form_data, set_form_data = mo.state({
    "name": "",
    "email": "",
    "age": 0
})

def update_field(field, value):
    set_form_data(lambda data: {**data, field: value})

name_input = mo.ui.text(
    value=get_form_data()["name"],
    on_change=lambda v: update_field("name", v)
)

email_input = mo.ui.text(
    value=get_form_data()["email"],
    on_change=lambda v: update_field("email", v)
)
```

## State Debugging

### Logging State Changes

```python
def logged_setter(name, setter):
    def wrapper(value_or_fn):
        if callable(value_or_fn):
            def logged_fn(old_value):
                new_value = value_or_fn(old_value)
                print(f"{name}: {old_value} -> {new_value}")
                return new_value
            setter(logged_fn)
        else:
            print(f"{name}: -> {value_or_fn}")
            setter(value_or_fn)
    return wrapper

get_count, _set_count = mo.state(0)
set_count = logged_setter("count", _set_count)
```

### State Inspector

```python
def inspect_state():
    return {
        "count": get_count(),
        "items": get_items(),
        "config": get_config()
    }

# Display current state
mo.json(inspect_state())
```

## Common State Pitfalls

### 1. Mutating State Directly

```python
# ❌ WRONG: Mutating state object
state = get_state()
state.field = "new value"  # Won't trigger updates

# ✅ CORRECT: Use setter
set_state(lambda s: replace(s, field="new value"))
```

### 2. State in Function Scope

```python
# ❌ WRONG: Local scope
def my_component():
    get_local, set_local = mo.state(0)  # Lost on re-run
    return get_local()

# ✅ CORRECT: Global scope
get_global, set_global = mo.state(0)

def my_component():
    return get_global()
```

### 3. Forgetting Lambda for Updates

```python
# ❌ WRONG: Reading then writing (race condition)
current = get_count()
set_count(current + 1)

# ✅ CORRECT: Atomic update with lambda
set_count(lambda count: count + 1)
```

## State Best Practices

1. **Always assign state to global variables**
2. **Use lambdas for updates that depend on current value**
3. **Use immutable updates (don't mutate objects)**
4. **Keep state minimal - derive what you can**
5. **Use descriptive names for state variables**
6. **Consider using dataclasses for complex state**
7. **Batch related updates to avoid flicker**
8. **Be cautious with self-looping state**
