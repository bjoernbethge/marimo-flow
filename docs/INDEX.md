# Technology Reference Index

**Last Updated**: 2025-11-25
**Status**: Complete - All 5 technologies documented

This directory contains comprehensive LLM-friendly reference documentation for key technologies used in the marimo-flow project.

## Overview

### Core Technologies Documented

1. **marimo** - Reactive Python notebook framework (with AI-Human Collaboration)
2. **mlflow** - Machine Learning lifecycle management & experiment tracking
3. **polars** - High-performance DataFrame library
4. **plotly** - Interactive visualization library
5. **pina** - Physics-Informed Neural Networks framework

---

## Quick Technology Matrix

| Technology | Type | Primary Use | Performance | Learning Curve |
|---|---|---|---|---|
| marimo | Notebook | Interactive development, AI collaboration, deployment | Fast startup | Low-Medium |
| mlflow | ML Ops | Experiment tracking, model registry, deployment | High | Low-Medium |
| polars | Data | DataFrame operations, transformation | Very High | Low-Medium |
| plotly | Visualization | Interactive charts, dashboards | High | Low |
| pina | ML/SciML | Neural networks, differential equations | High | Medium-High |

---

## Document Guide

### marimo-quickstart.md

**Coverage**: Complete overview of marimo reactive notebooks with AI-Human Collaboration

Key sections:
- Installation and basic commands
- Core concepts (reactivity, state management, UI elements)
- Layout and composition patterns
- **AI-Human Collaboration Patterns** (new!)
  - Interactive AI Agent with User Validation
  - Collaborative Notebooks with Comments
  - AI-Assisted Data Exploration
  - Collaborative Model Evaluation
  - Asynchronous Review Workflows
- Common usage patterns with code examples
- Best practices and anti-patterns
- Debugging & Development patterns
- Integration patterns (Polars, Plotly, DuckDB, MLflow)
- Deployment and export options
- API reference for quick lookup

**When to use this**:
- Setting up marimo notebooks
- Understanding reactivity model
- Building interactive UIs
- Building AI-powered applications
- Implementing human-in-the-loop workflows
- Deploying notebooks as apps
- Integrating with other libraries

**Key APIs**:
- `mo.state()` - Reactive state
- `mo.ui.*` - UI components (slider, button, dropdown, text_area, etc.)
- `mo.hstack()/mo.vstack()` - Layout elements
- `mo.md()` - Markdown display
- Export commands: `marimo export`, `marimo run`

**AI-Focused Features**:
- User approval gates for AI outputs
- Interactive validation workflows
- Collaboration tracking
- Progress monitoring

---

### mlflow-quickstart.md

**Coverage**: Complete guide to MLflow experiment tracking and model management

Key sections:
- What is MLflow and key features
- Installation and MLflow server setup
- Core concepts (Experiments, Runs, Parameters, Metrics, Artifacts)
- Model Registry and versioning
- Common ML workflow patterns
  - Complete ML workflow with tracking
  - Deep learning with PyTorch
  - Hyperparameter tuning
  - Custom metrics tracking
  - Model serving & loading
- Integration with marimo notebooks
  - Interactive dashboard creation
  - Real-time metric tracking
  - Experiment comparison
- Best practices and anti-patterns
- Troubleshooting guide
- API reference for quick lookup

**When to use this**:
- Tracking ML experiments
- Managing model versions
- Comparing model performance
- Deploying models to production
- Creating experiment dashboards
- Integrating with training scripts
- Building reproducible ML pipelines

**Key APIs**:
- `mlflow.set_experiment()` - Organize experiments
- `mlflow.start_run()` - Begin tracking
- `mlflow.log_param()/log_params()` - Log hyperparameters
- `mlflow.log_metric()/log_metrics()` - Log performance metrics
- `mlflow.log_artifact()/log_artifacts()` - Log files
- Framework-specific: `mlflow.sklearn.log_model()`, `mlflow.pytorch.log_model()`, etc.
- `mlflow.pyfunc.load_model()` - Load for inference

**Marimo Integration**:
- Build interactive experiment dashboards
- Real-time tracking during training
- Model comparison interfaces
- Results visualization

---

### polars-quickstart.md

**Coverage**: Complete guide to Polars DataFrame operations

Key sections:
- Installation and basic setup
- Core concepts (eager vs lazy evaluation, expressions)
- DataFrame operations (select, filter, with_columns, group_by)
- Advanced features (joins, window functions, type casting)
- String and date/time operations
- Common patterns and anti-patterns
- Performance optimization tips
- Troubleshooting guide
- API reference for expressions and operations

**When to use this**:
- Data loading and transformation
- Filtering and aggregation
- Performance-critical data operations
- Complex data pipelines
- Integration with marimo for data exploration

**Key Concepts**:
- **Eager API**: Immediate execution
- **Lazy API**: Deferred execution with optimization
- **Expressions**: `pl.col()`, `pl.all()`, `pl.exclude()`
- **Main operations**: `.select()`, `.filter()`, `.with_columns()`, `.group_by()`

**Performance Notes**:
- 10-100x faster than pandas
- Use lazy API for large datasets
- Built on Rust + Apache Arrow
- Multi-core execution by default

---

### plotly-quickstart.md

**Coverage**: Interactive visualization with Plotly

Key sections:
- Installation (basic + chart-studio)
- Plotly Express (high-level API) for common charts
- Graph Objects (low-level API) for customization
- Figure properties and updates
- Customization and styling
- Multi-series and grouping
- 3D charts and subplots
- Hover text and annotations
- Common patterns (dashboards, dropdowns, time series)
- Best practices and anti-patterns
- Export options (HTML, PNG, PDF)
- Integration with Pandas, Polars, Marimo, Dash

**When to use this**:
- Creating interactive charts
- Building dashboards
- Exporting visualizations
- Exploring data visually
- Sharing results with interactivity

**Two Main Approaches**:
1. **Plotly Express** (`px`) - Simple, declarative (recommended for most cases)
2. **Graph Objects** (`go`) - Low-level, full control (for complex layouts)

**Chart Types Available**:
- Basic: scatter, line, bar, histogram, box, violin
- Advanced: heatmap, 3D scatter, sunburst, treemap, funnel
- Statistical: distribution, regression, ANOVA

**Export Options**:
- HTML (interactive, standalone)
- PNG/PDF/SVG (static images, requires kaleido)
- Embed in web apps via Dash

---

### pina-quickstart.md

**Coverage**: Scientific Machine Learning with Physics-Informed Neural Networks

Key sections:
- What is PINA (introduction to SciML)
- Installation and verification
- Four-step workflow (Problem → Model → Solver → Train)
- Problem definition (data-driven vs physics-informed)
- Model architectures (FeedForward, ResNet, GraphNet)
- Solvers (Supervised, PINN, DeepONet)
- Training and evaluation
- Loss functions and regularization
- Common patterns with complete examples
- Best practices for PINNs
- Troubleshooting guide
- API reference

**When to use this**:
- Solving differential equations (PDEs, ODEs)
- Data-driven learning with neural networks
- Physics-informed constraint enforcement
- Learning neural operators
- Scientific computing with deep learning

**Core Workflow**:
```
1. Define Problem (domain + conditions)
2. Design Model (FeedForward, ResNet, etc.)
3. Select Solver (Supervised, PINN, DeepONet)
4. Train (using Trainer with PyTorch Lightning)
```

**Key Concepts**:
- **PINNs**: Combine neural networks with physical constraints
- **Physics Constraints**: Differential equation residuals as loss terms
- **Data Conditions**: Training data + boundary conditions
- **Multi-device Training**: Scale to multiple GPUs

**Built On**:
- PyTorch (core framework)
- PyTorch Lightning (training abstractions)
- PyTorch Geometric (graph operations)

---

## Integration Patterns

### Common Workflows in This Project

#### 1. Data Exploration & Visualization
```
marimo (notebook) → polars (data loading) → plotly (visualization)
```

Example:
```python
import marimo as mo
import polars as pl
import plotly.express as px

df = pl.read_csv("data.csv")
fig = px.scatter(df.to_pandas(), x="x", y="y")
mo.Html(fig.to_html())
```

#### 2. Interactive Dashboard
```
marimo (UI) → polars (data pipeline) → plotly (charts)
```

Example:
```python
import marimo as mo
import polars as pl
import plotly.express as px

# UI
column_select = mo.ui.dropdown(df.columns, label="Column:")

# Data processing
if column_select.value:
    filtered_df = df.select(column_select.value)

    # Visualization
    fig = px.bar(filtered_df.to_pandas())
    mo.Html(fig.to_html())
```

#### 3. Machine Learning Pipeline
```
polars (data prep) → pina (model training) → plotly (results visualization)
```

Example:
```python
import polars as pl
import torch
from pina import Problem, FeedForward, SupervisedSolver
import plotly.express as px

# Load and preprocess data
df = pl.read_csv("data.csv")
x = torch.from_numpy(df["x"].to_numpy()).float()
y = torch.from_numpy(df["y"].to_numpy()).float()

# Train model
model = FeedForward(1, 1, [64, 64])
solver = SupervisedSolver(problem, model)

# Visualize results
predictions = solver(x).detach().numpy()
fig = px.scatter(x=x, y=y, title="Predictions vs Data")
```

#### 4. Scientific Problem Solving with Physics
```
pina (model) + marimo (interface) + plotly (visualization)
```

Example:
```python
import marimo as mo
import torch
from pina import Problem, PINNSolver
import plotly.graph_objects as go

# Define problem (PDE)
problem = Problem(domain)
# ... add conditions ...

# Train
solver = PINNSolver(problem, model)
trainer.fit()

# Visualize solution
x_range = torch.linspace(0, 1, 100)
u = solver(x_range)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_range, y=u.detach()))
mo.Html(fig.to_html())
```

---

## File Organization

```
docs/
├── INDEX.md                    # This file - navigation guide
├── marimo-quickstart.md        # Marimo reference (with AI collaboration)
├── mlflow-quickstart.md        # MLflow reference (NEW!)
├── polars-quickstart.md        # Polars reference
├── plotly-quickstart.md        # Plotly reference
├── pina-quickstart.md          # PINA reference
└── [future docs]               # Additional references as needed
```

---

## How to Use These References

### For Quick Lookup
Each document has an **API Reference - Quick Lookup** section with the most commonly used functions and classes.

### For Learning
Start with "Core Concepts" sections which explain fundamental ideas, then progress to "Common Patterns" for real-world examples.

### For Problem-Solving
Each document includes "Common Issues & Solutions" for debugging and "Best Practices" for avoiding pitfalls.

### For Integration
Refer to the **Integration Patterns** section in this INDEX for combining multiple technologies.

---

## Quick Command Reference

### Marimo
```bash
marimo edit                     # Create/edit notebook
marimo run notebook.py          # Deploy as app
marimo export notebook.py       # Export to HTML/ipynb
marimo convert old.ipynb        # Convert from Jupyter
```

### Polars
```python
df = pl.read_csv("file.csv")
result = df.filter(...).group_by(...).agg(...)
result_lazy = pl.scan_csv(...).collect()  # Lazy evaluation
```

### Plotly
```python
fig = px.scatter(df, x="x", y="y")        # Express API
fig = go.Figure(data=[go.Scatter(...)])    # Graph Objects
fig.show()                                  # Display
fig.write_html("chart.html")               # Save
```

### PINA
```python
# 1. Define problem
problem = Problem(domain)
problem.add_condition(Condition(...))

# 2. Create model
model = FeedForward(input_dims, output_dims, layers)

# 3. Create solver
solver = SupervisedSolver(problem, model)

# 4. Train
trainer = Trainer(solver, max_epochs=100)
trainer.fit()
```

---

## Technology Versions

All documentation is current as of **November 25, 2025** with latest stable versions:

- **marimo**: Latest (marimo-team/marimo)
- **mlflow**: 2.0+ (mlflow/mlflow)
- **polars**: Latest (pola-rs/polars)
- **plotly**: Latest (plotly/plotly.py)
- **pina**: Latest (mathLab/PINA)

---

## External Resources

### Official Documentation
- Marimo: https://docs.marimo.io
- MLflow: https://mlflow.org/docs/latest/index.html
- Polars: https://docs.pola.rs/
- Plotly: https://plotly.com/python/
- PINA: https://mathlab.github.io/PINA/

### GitHub Repositories
- Marimo: https://github.com/marimo-team/marimo
- MLflow: https://github.com/mlflow/mlflow
- Polars: https://github.com/pola-rs/polars
- Plotly: https://github.com/plotly/plotly.py
- PINA: https://github.com/mathLab/PINA

### Community & Support
- Marimo Discord: https://discord.gg/marimo
- Plotly Community: https://community.plotly.com/
- PINA Contact: pina.mathlab@gmail.com
- Stack Overflow: Tag with library name

---

## Contributing to This Documentation

When adding new references:
1. Follow the template structure in existing files
2. Include version information and update dates
3. Add complete, runnable code examples
4. Include both good and bad patterns (DO/DON'T sections)
5. Link to official documentation
6. Keep API references concise and scannable
7. Include common issues and solutions

---

## Status & Completeness

### Marimo - Complete (Enhanced)
- Covers all core concepts
- Includes state management, UI, deployment
- **NEW**: AI-Human collaboration patterns
- **NEW**: Pydantic AI integration examples
- Integration patterns documented
- Status: Ready for production use

### MLflow - Complete (NEW)
- Installation and server setup
- Experiment tracking & metrics
- Model Registry & versioning
- Framework-specific examples (sklearn, PyTorch, TensorFlow, XGBoost)
- Marimo integration patterns
- Troubleshooting guide
- Status: Ready for production use

### Polars - Complete
- Eager and lazy evaluation explained
- All major operations documented
- Performance tips included
- Troubleshooting guide complete
- Status: Ready for production use

### Plotly - Complete
- Both Express and Graph Objects covered
- 20+ chart types explained
- Export options documented
- Integration patterns shown
- Status: Ready for production use

### PINA - Complete
- Workflow and concepts explained
- Multiple examples provided
- Troubleshooting section included
- Integration with PyTorch documented
- Status: Ready for learning and basic use

---

## Next Steps

1. **Start here**: Pick a technology you want to learn
2. **Read core concepts**: Understand fundamental ideas
3. **Try examples**: Run provided code snippets
4. **Explore patterns**: See real-world usage
5. **Reference API**: Use quick lookup sections
6. **Integrate**: Combine technologies as needed
7. **Solve problems**: Use troubleshooting guide when stuck

---

## Document Metadata

- **Created**: 2025-11-21
- **Last Updated**: 2025-11-25
- **Format**: Markdown (LLM-optimized)
- **Scope**: Reference documentation, not tutorials
- **Target Audience**: Developers, data scientists, ML engineers, AI practitioners
- **Prerequisites**: Python 3.9+, basic programming knowledge
- **New in Update**: MLflow quickstart + AI-Human collaboration patterns in marimo

---

## License & Attribution

These reference documents synthesize information from official documentation:
- Marimo: https://github.com/marimo-team/marimo (Apache 2.0)
- Polars: https://github.com/pola-rs/polars (MIT)
- Plotly: https://github.com/plotly/plotly.py (MIT)
- PINA: https://github.com/mathLab/PINA (BSD)

All original content in these references is freely available for use and modification.
