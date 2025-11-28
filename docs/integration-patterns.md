# Integration Patterns - Advanced Reference

**Last Updated**: 2025-11-21
**Status**: Current

This document covers advanced patterns for integrating marimo, polars, plotly, and pina together in real-world applications.

---

## Pattern 1: Interactive Data Explorer

### Goal
Build an interactive data exploration tool with dynamic filtering and visualization.

### Architecture
```
marimo (UI controls) → polars (data operations) → plotly (visualization)
```

### Implementation

```python
import marimo as mo
import polars as pl
import plotly.express as px

# Load data once
_df = pl.read_csv("data.csv")

# UI Controls
column_x = mo.ui.dropdown(_df.columns, label="X Axis:", value=_df.columns[0])
column_y = mo.ui.dropdown(_df.columns, label="Y Axis:", value=_df.columns[1])

filter_column = mo.ui.dropdown(_df.columns, label="Filter Column:")
filter_value = mo.ui.text(placeholder="Filter Value", label="Filter Value:")

# Data Processing with Polars
if filter_column.value and filter_value.value:
    filtered_df = _df.filter(
        pl.col(filter_column.value) == filter_value.value
    )
else:
    filtered_df = _df

# Visualization
if column_x.value and column_y.value:
    fig = px.scatter(
        filtered_df.to_pandas(),
        x=column_x.value,
        y=column_y.value,
        title="Data Explorer"
    )
    mo.Html(fig.to_html())
else:
    mo.md("Please select columns for X and Y axes")
```

### Key Patterns
- UI values accessed with `.value` attribute
- Polars filters data efficiently
- Plotly figures embedded with `mo.Html()`
- Reactive updates on any UI change

### Performance Tips
- Keep data in Polars DataFrame (lazy evaluation optional)
- Cache data loading outside reactive cells
- Use `.sample()` for large datasets

---

## Pattern 2: Multi-View Dashboard

### Goal
Create dashboard with multiple synchronized views of same data.

### Architecture
```
marimo (layout + state) → polars (data pipeline) → plotly (multiple charts)
```

### Implementation

```python
import marimo as mo
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# State for synchronized views
get_selected_category, set_selected_category = mo.state("All")

# Load data
df = pl.read_csv("sales_data.csv")

# UI: Category selector
categories = ["All"] + df["category"].unique().to_list()
category_selector = mo.ui.dropdown(categories, label="Category:")

# Update state when selection changes
if category_selector.value:
    set_selected_category(category_selector.value)

# Filter data based on selection
if get_selected_category() != "All":
    display_df = df.filter(pl.col("category") == get_selected_category())
else:
    display_df = df

# Create subplot dashboard
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Revenue Over Time", "Top Products", "Inventory Levels", "Profit Margin")
)

# View 1: Time series
time_data = display_df.group_by("date").agg(pl.col("revenue").sum())
fig.add_trace(
    go.Scatter(x=time_data["date"], y=time_data["revenue"], name="Revenue"),
    row=1, col=1
)

# View 2: Bar chart - Top products
top_products = display_df.group_by("product").agg(
    pl.col("revenue").sum().alias("total")
).sort("total", descending=True).head(5)
fig.add_trace(
    go.Bar(x=top_products["product"], y=top_products["total"], name="Sales"),
    row=1, col=2
)

# View 3: Inventory
inventory = display_df.group_by("product").agg(pl.col("quantity").mean())
fig.add_trace(
    go.Scatter(x=inventory["product"], y=inventory["quantity"], name="Avg Qty"),
    row=2, col=1
)

# View 4: Profit margin
margins = display_df.with_columns(
    (pl.col("profit") / pl.col("revenue")).alias("margin")
).group_by("category").agg(pl.col("margin").mean())
fig.add_trace(
    go.Bar(x=margins["category"], y=margins["margin"], name="Margin %"),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False)
mo.Html(fig.to_html())
```

### Key Patterns
- Use `mo.state()` for synchronized selection across views
- Combine multiple Polars operations in pipeline
- Use subplots for compact multi-view layout
- Filter data once, display multiple ways

### Performance Considerations
- Pre-compute aggregations in Polars
- Cache group_by results for large datasets
- Use lazy evaluation: `df.lazy().filter(...).collect()`

---

## Pattern 3: Real-Time Data Pipeline

### Goal
Build a data pipeline that processes incoming data and updates visualizations.

### Architecture
```
Input Data → polars (cleaning/transformation) → cache → marimo (display)
                                              ↓
                                          plotly (visualization)
```

### Implementation

```python
import marimo as mo
import polars as pl
import plotly.express as px
from datetime import datetime
import pandas as pd

# State for accumulating data
get_data, set_data = mo.state(pl.DataFrame())

# Simulate receiving new data
def receive_new_data():
    """Simulate incoming data from source"""
    new_rows = pl.DataFrame({
        "timestamp": [datetime.now()],
        "value": [42 + hash(datetime.now()) % 20],
        "status": ["active"]
    })
    return new_rows

# Button to simulate new data arrival
refresh_button = mo.ui.button(
    label="Receive New Data",
    on_change=lambda _: update_data()
)

def update_data():
    """Append new data and keep recent 100 records"""
    new = receive_new_data()
    current = get_data()

    if current.height == 0:
        updated = new
    else:
        updated = pl.concat([current, new])

    # Keep only recent records for performance
    updated = updated.tail(100)
    set_data(updated)

# Process data with Polars
data = get_data()

if data.height > 0:
    # Compute rolling statistics
    data_with_stats = data.with_columns([
        pl.col("value").rolling_mean(window_size=5).alias("moving_avg"),
        pl.col("value").rolling_std(window_size=5).alias("volatility")
    ])

    # Create visualization
    fig = px.line(
        data_with_stats.to_pandas(),
        x="timestamp",
        y=["value", "moving_avg"],
        title="Real-Time Data Stream"
    )

    mo.vstack([
        refresh_button,
        mo.Html(fig.to_html()),
        mo.md(f"**Records**: {data.height} | **Latest**: {data_with_stats[-1]['value'][0]}")
    ])
else:
    mo.md("No data yet. Click button to receive data.")
```

### Key Patterns
- Use `mo.state()` to accumulate data
- Apply Polars transformations after each update
- Keep only recent data for performance
- Display stats alongside visualization

### Optimization Tips
- Use `.tail(n)` to limit data size
- Batch updates instead of single records
- Use rolling windows for efficient computations
- Consider lazy evaluation for complex pipelines

---

## Pattern 4: ML Model Training & Evaluation

### Goal
Train PINA models on data processed by Polars, visualize results with Plotly in marimo.

### Architecture
```
polars (data prep) → pina (training) → plotly (visualization) → marimo (display)
```

### Implementation

```python
import marimo as mo
import polars as pl
import torch
from pina import Problem, FeedForward, SupervisedSolver
from pina.domain import Domain
from pina.operators import Condition
from pina.trainer import Trainer
import plotly.graph_objects as go

# Load and prepare data
df = pl.read_csv("training_data.csv")

# Normalize with Polars
mean_x = df["x"].mean()
std_x = df["x"].std()
mean_y = df["y"].mean()
std_y = df["y"].std()

x_norm = (df["x"] - mean_x) / std_x
y_norm = (df["y"] - mean_y) / std_y

# Convert to tensors
x_data = torch.from_numpy(x_norm.to_numpy()).float().unsqueeze(-1)
y_data = torch.from_numpy(y_norm.to_numpy()).float().unsqueeze(-1)

# Create problem
domain = Domain()
domain.add("x", -3, 3)

problem = Problem(domain)
problem.add_condition(Condition(domain, x_data, y_data))

# Create and train model
mo.md("## Training PINA Model")
mo.md("Training in progress...")

model = FeedForward(
    input_dimensions=1,
    output_dimensions=1,
    layers=[64, 64],
    activations=torch.nn.Tanh()
)

solver = SupervisedSolver(problem=problem, model=model)

trainer = Trainer(solver, max_epochs=100)
trainer.fit()

mo.md("✓ Training complete!")

# Evaluate on test set
x_test = torch.linspace(-3, 3, 100).unsqueeze(-1)
y_pred = solver(x_test).detach().numpy()

# Denormalize predictions
y_pred_denorm = y_pred * std_y + mean_y

# Create visualization
fig = go.Figure()

# Original data
fig.add_trace(go.Scatter(
    x=df["x"].to_numpy(),
    y=df["y"].to_numpy(),
    mode='markers',
    name='Training Data',
    marker=dict(size=8, color='blue')
))

# Model predictions
fig.add_trace(go.Scatter(
    x=x_test.squeeze().numpy() * std_x + mean_x,
    y=y_pred_denorm.squeeze(),
    mode='lines',
    name='PINA Model',
    line=dict(color='red', width=3)
))

fig.update_layout(
    title="PINA Model vs Training Data",
    xaxis_title="X",
    yaxis_title="Y",
    hovermode='x unified'
)

mo.Html(fig.to_html())

# Evaluation metrics
train_loss = trainer.trainer.callback_metrics.get('train_loss', 'N/A')
val_loss = trainer.trainer.callback_metrics.get('val_loss', 'N/A')

mo.md(f"""
## Model Evaluation
- **Training Loss**: {train_loss}
- **Validation Loss**: {val_loss}
""")
```

### Key Patterns
- Use Polars for normalization/preprocessing
- Convert Polars DataFrames to tensors for PINA
- Train model during marimo execution
- Visualize results with Plotly

### Best Practices
- Always normalize data (especially for neural networks)
- Use appropriate loss weighting
- Monitor training metrics
- Visualize predictions vs actual data

---

## Pattern 5: Scientific Problem Solving

### Goal
Solve a PDE using PINA, display solution with marimo/plotly interface.

### Architecture
```
marimo (UI) → pina (solve PDE) → plotly (visualize solution)
```

### Implementation

```python
import marimo as mo
import torch
from pina import Problem, FeedForward, PINNSolver
from pina.domain import Domain
from pina.operators import Condition, Identity
from pina.trainer import Trainer
import plotly.graph_objects as go

mo.md("# PDE Solver Interface")

# UI: Problem parameters
x_domain = mo.ui.slider(0.5, 5.0, step=0.5, label="Domain size (x):")
t_domain = mo.ui.slider(0.5, 5.0, step=0.5, label="Domain size (t):")
n_epochs = mo.ui.slider(50, 500, step=50, label="Training epochs:")

# Create domain based on UI inputs
domain = Domain()
domain.add("x", 0, x_domain.value)
domain.add("t", 0, t_domain.value)

# Define PDE problem (simplified heat equation)
problem = Problem(domain)

# Initial condition: u(x, 0) = sin(pi*x)
x_init = torch.linspace(0, x_domain.value, 50).unsqueeze(-1)
t_init = torch.zeros_like(x_init)
u_init = torch.sin(torch.pi * x_init / x_domain.value)

initial_points = torch.cat([x_init, t_init], dim=1)
problem.add_condition(Condition(domain, initial_points, u_init))

# Boundary conditions: u(0,t) = u(L,t) = 0
x_boundary = torch.zeros(50, 1)
t_boundary = torch.linspace(0, t_domain.value, 50).unsqueeze(-1)
u_boundary = torch.zeros(50, 1)

boundary_points_x0 = torch.cat([x_boundary, t_boundary], dim=1)
problem.add_condition(Condition(domain, boundary_points_x0, u_boundary))

# Create model
model = FeedForward(
    input_dimensions=2,
    output_dimensions=1,
    layers=[64, 64, 64],
    activations=torch.nn.Tanh()
)

# Create and train solver
mo.md(f"Training PINN solver for {int(n_epochs.value)} epochs...")

solver = PINNSolver(problem=problem, model=model)
trainer = Trainer(solver, max_epochs=int(n_epochs.value))
trainer.fit()

mo.md("✓ Solution computed!")

# Visualize solution
x_range = torch.linspace(0, x_domain.value, 50)
t_range = torch.linspace(0, t_domain.value, 50)
X, T = torch.meshgrid(x_range, t_range, indexing='ij')

# Flatten for network input
inputs = torch.cat([X.reshape(-1, 1), T.reshape(-1, 1)], dim=1)
u_solution = solver(inputs).detach().reshape(50, 50)

# Create 3D surface plot
fig = go.Figure(data=[go.Surface(
    x=x_range.numpy(),
    y=t_range.numpy(),
    z=u_solution.numpy(),
    colorscale='Viridis'
)])

fig.update_layout(
    title="PDE Solution: Heat Equation",
    xaxis_title="Position (x)",
    yaxis_title="Time (t)",
    zaxis_title="Temperature (u)",
    height=700
)

mo.Html(fig.to_html())

# Display solution statistics
mo.md(f"""
## Solution Statistics
- **Min**: {u_solution.min():.4f}
- **Max**: {u_solution.max():.4f}
- **Mean**: {u_solution.mean():.4f}
- **Std Dev**: {u_solution.std():.4f}
""")
```

### Key Patterns
- Use marimo UI for problem parameters
- Define domain and conditions based on inputs
- Train PINN solver dynamically
- Visualize solutions in 3D or 2D

### Scientific Considerations
- Properly define initial/boundary conditions
- Choose appropriate network architecture
- Monitor physics residuals
- Validate solutions against analytical solutions when possible

---

## Pattern 6: Batch Processing Pipeline

### Goal
Process multiple datasets efficiently with Polars, train models with PINA, aggregate results.

### Architecture
```
Input Files → polars (parallel processing) → pina (model training) → plotly (comparison)
```

### Implementation

```python
import marimo as mo
import polars as pl
import torch
from pina import Problem, FeedForward, SupervisedSolver
from pina.trainer import Trainer
from pathlib import Path
import plotly.graph_objects as go

# Find all data files
data_files = list(Path("data").glob("*.csv"))

mo.md(f"## Batch Processing {len(data_files)} Files")

results = []

for file_path in data_files:
    mo.md(f"Processing {file_path.name}...")

    # Load with Polars
    df = pl.read_csv(file_path)

    # Quick preprocessing
    df = df.filter(pl.col("value") > 0)  # Remove invalid
    df = df.with_columns(
        ((pl.col("value") - pl.col("value").mean()) / pl.col("value").std()).alias("normalized")
    )

    # Train model on this dataset
    x_data = torch.from_numpy(df["x"].to_numpy()).float().unsqueeze(-1)
    y_data = torch.from_numpy(df["normalized"].to_numpy()).float().unsqueeze(-1)

    # Simple model
    model = FeedForward(1, 1, [32, 32], torch.nn.Tanh())

    # Train quickly
    trainer = Trainer(model, max_epochs=20)
    # (Simplified - actual PINA solver would be used)

    # Store results
    results.append({
        "file": file_path.name,
        "rows": df.height,
        "mean": df["value"].mean(),
        "std": df["value"].std()
    })

# Aggregate results with Polars
results_df = pl.DataFrame(results)

mo.md("## Results Summary")

# Create summary visualization
summary_stats = results_df.select([
    pl.col("file"),
    pl.col("rows"),
    pl.col("mean"),
    pl.col("std")
])

fig = go.Figure()

fig.add_trace(go.Bar(
    name="Mean",
    x=summary_stats["file"],
    y=summary_stats["mean"]
))

fig.add_trace(go.Bar(
    name="Std Dev",
    x=summary_stats["file"],
    y=summary_stats["std"]
))

fig.update_layout(
    title="Summary Statistics Across Files",
    barmode="group",
    xaxis_title="File",
    yaxis_title="Value"
)

mo.Html(fig.to_html())

# Display table
mo.md(summary_stats.to_pandas().to_html())
```

### Key Patterns
- Iterate over multiple files efficiently
- Apply Polars transformations consistently
- Train individual models per dataset
- Aggregate results for comparison

### Performance Optimization
- Use lazy evaluation for large datasets
- Parallelize training with GPU
- Cache intermediate results
- Filter early to reduce data size

---

## Pattern 7: Error Handling & Validation

### Goal
Robust data pipeline with validation and error recovery.

### Implementation

```python
import marimo as mo
import polars as pl
from pina import Problem, FeedForward, SupervisedSolver
import torch

get_errors, set_errors = mo.state([])

def safe_load_data(filepath):
    """Load data with error handling"""
    try:
        df = pl.read_csv(filepath)

        # Validation checks
        if df.height == 0:
            raise ValueError("DataFrame is empty")

        # Check required columns
        required = ["x", "y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Check data types
        numeric_cols = ["x", "y"]
        for col in numeric_cols:
            try:
                df.select(pl.col(col).cast(pl.Float64))
            except:
                raise ValueError(f"Column {col} cannot be converted to numeric")

        return df, None
    except Exception as e:
        error_msg = f"Error loading {filepath}: {str(e)}"
        set_errors([*get_errors(), error_msg])
        return None, error_msg

def safe_train_model(df):
    """Train model with error handling"""
    try:
        x = torch.from_numpy(df["x"].to_numpy()).float().unsqueeze(-1)
        y = torch.from_numpy(df["y"].to_numpy()).float().unsqueeze(-1)

        # Check for NaN/Inf
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("X contains NaN or Inf")
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise ValueError("Y contains NaN or Inf")

        # Model training
        model = FeedForward(1, 1, [32, 32], torch.nn.Tanh())
        # ... training code ...

        return model, None
    except Exception as e:
        error_msg = f"Error training model: {str(e)}"
        set_errors([*get_errors(), error_msg])
        return None, error_msg

# Main pipeline with error handling
filepath = mo.ui.text(placeholder="data.csv", label="File path:")
load_button = mo.ui.button(label="Load & Train")

if load_button.value:
    df, load_error = safe_load_data(filepath.value)

    if df is not None:
        model, train_error = safe_train_model(df)

        if model is not None:
            mo.md("✓ Model trained successfully!")
        else:
            mo.md(f"❌ Training failed: {train_error}")
    else:
        mo.md(f"❌ Loading failed: {load_error}")

# Display all errors
if get_errors():
    mo.md("## Errors Encountered")
    for error in get_errors():
        mo.md(f"- {error}")
else:
    mo.md("No errors")
```

### Key Patterns
- Validate data at each stage
- Catch and store errors in state
- Provide user feedback
- Continue processing on errors

---

## Best Practices Across All Patterns

### Performance
1. **Polars First**: Process data with Polars before visualization
2. **Lazy Evaluation**: Use `.lazy()` for large datasets
3. **Caching**: Store intermediate results in marimo state
4. **Batch Operations**: Group operations instead of iteration

### Code Quality
1. **Separation of Concerns**: Keep UI, data, and models separate
2. **Error Handling**: Wrap operations in try-catch blocks
3. **Validation**: Check data at pipeline boundaries
4. **Documentation**: Comment complex operations

### User Experience
1. **Feedback**: Show progress and status messages
2. **Responsiveness**: Update UI quickly
3. **Clear Layouts**: Use mo.vstack/mo.hstack for organization
4. **Helpful Errors**: Display actionable error messages

### Integration Points
1. **Marimo → Polars**: Pass UI values to filter operations
2. **Polars → PINA**: Convert normalized tensors carefully
3. **PINA → Plotly**: Denormalize predictions for visualization
4. **Plotly → Marimo**: Embed with `mo.Html()`

---

## Common Pitfalls & Solutions

### Pitfall 1: Responsive Updates Too Slow

**Problem**: Dashboard updates slowly due to large computations.

**Solution**:
- Precompute results in Polars
- Use lazy evaluation
- Cache results in state
- Sample data for interactive views, use full data for exports

### Pitfall 2: Data Type Mismatches

**Problem**: Errors when converting between libraries.

**Solution**:
- Use `.to_pandas()` for Polars → Plotly
- Convert to tensors explicitly for PINA
- Check dtypes at each stage
- Validate data types early

### Pitfall 3: Memory Overflow

**Problem**: Running out of memory with large datasets.

**Solution**:
- Use lazy evaluation in Polars
- Sample data for visualization
- Process in batches
- Stream data instead of loading all at once

### Pitfall 4: Model Not Converging

**Problem**: PINA model training stalls.

**Solution**:
- Normalize input/output data
- Use appropriate activation functions (Tanh for PINNs)
- Adjust learning rate
- Check for NaN/Inf in data

---

## Testing Your Integrations

### Unit Tests for Pipelines

```python
import polars as pl
import torch

def test_data_pipeline():
    # Create test data
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # Process
    result = df.with_columns((pl.col("x") * 2).alias("x2"))

    # Validate
    assert result.height == 3
    assert "x2" in result.columns
    assert result["x2"].sum() == 12

def test_tensor_conversion():
    # Create Polars data
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})

    # Convert to tensor
    tensor = torch.from_numpy(df["x"].to_numpy()).float()

    # Validate
    assert tensor.shape == (3,)
    assert tensor.dtype == torch.float32
```

---

## Performance Benchmarks

### Data Processing Speed
- **Polars eager**: ~10-100x pandas
- **Polars lazy**: Optimized query execution
- **Plotly rendering**: < 1s for typical charts
- **PINA training**: Depends on model size and data

### Memory Usage
- **Polars**: ~30-50% of pandas memory
- **Plotly figures**: ~5-10MB for typical charts
- **PINA models**: ~1-100MB depending on architecture

---

## Debugging Tips

### Marimo
```python
# Check reactive dependencies
mo.md("This cell depends on: x, y")

# Monitor state changes
mo.md(f"State value: {get_state()}")
```

### Polars
```python
# Check query plan
query = pl.scan_csv("file.csv").filter(...)
print(query.explain(optimized=True))

# Inspect schema
print(df.schema)
```

### Plotly
```python
# Debug layout
fig.update_layout(showlegend=True)

# Check data ranges
print(fig.data[0])
```

### PINA
```python
# Monitor training loss
print(trainer.trainer.callback_metrics)

# Check model output
output = model(sample_input)
print(output.shape, output.dtype)
```

---

## References

- **Marimo State**: marimo-quickstart.md → State Management
- **Polars Operations**: polars-quickstart.md → Core Concepts
- **Plotly Customization**: plotly-quickstart.md → Customization & Styling
- **PINA Workflow**: pina-quickstart.md → Four-Step Workflow
