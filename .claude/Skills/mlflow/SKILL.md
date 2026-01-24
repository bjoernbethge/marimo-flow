---
name: mlflow
description: MLflow for ML lifecycle management - experiment tracking, LLM/GenAI tracing, model registry, and deployment with GenAI and MCP support
triggers:
  - mlflow
  - experiment tracking
  - model registry
  - llm tracking
  - genai tracing
  - mlflow autolog
  - mlflow tracking
  - log metrics
  - track experiment
allowed_tools:
  - Read
  - Write
  - Edit
  - Bash
  - mcp__mlflow__search_traces
  - mcp__mlflow__get_trace
  - mcp__mlflow__delete_traces
  - mcp__mlflow__set_trace_tag
  - mcp__mlflow__delete_trace_tag
  - mcp__mlflow__log_feedback
  - mcp__mlflow__log_expectation
  - mcp__mlflow__get_assessment
  - mcp__mlflow__update_assessment
  - mcp__mlflow__delete_assessment
  - mcp__mlflow__evaluate_traces
  - mcp__mlflow__list_scorers
  - mcp__mlflow__register_llm_judge
  - mcp__plugin_context7_context7__resolve-library-id
  - mcp__plugin_context7_context7__query-docs
---

# MLflow Development Skill

Expert guidance for ML lifecycle management with MLflow, including GenAI/LLM tracking and MCP integration.

## Core Concepts

MLflow is an open-source platform for managing the ML lifecycle with four main components:

1. **Tracking**: Log parameters, metrics, and artifacts
2. **Models**: Package and deploy models
3. **Registry**: Centralize model storage and versioning
4. **Projects**: Package reproducible runs

## Quick Start

### Configuration

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my-experiment")
```

## Autologging

MLflow provides automatic logging for major frameworks:

### ML Frameworks

```python
import mlflow

# Scikit-learn
mlflow.sklearn.autolog()

# PyTorch
mlflow.pytorch.autolog()

# TensorFlow/Keras
mlflow.tensorflow.autolog()
```

### GenAI/LLM Providers

```python
import mlflow

# OpenAI
mlflow.openai.autolog()

# Anthropic
mlflow.anthropic.autolog()

# LangChain
mlflow.langchain.autolog()
```

**What gets logged automatically:**
- Model parameters and hyperparameters
- Training metrics
- Model artifacts and dependencies
- Tokens, latency, and cost (for LLMs)
- Tool calls and function invocations

## Manual Tracking

```python
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_params({"batch_size": 32, "epochs": 100})

    # Log metrics
    mlflow.log_metric("train_loss", 0.5)

    # Log metrics with steps
    for epoch in range(num_epochs):
        train_loss = train_model()
        mlflow.log_metric("train_loss", train_loss, step=epoch)

    # Log model
    mlflow.sklearn.log_model(model, name="model")
```

## GenAI Tracing

### Basic Tracing

```python
import mlflow

@mlflow.trace
def my_llm_app(question: str) -> str:
    """Traced LLM application"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

# Trace is automatically logged
result = my_llm_app("What is MLflow?")

# Access trace
trace_id = mlflow.get_last_active_trace_id()
trace = mlflow.get_trace(trace_id=trace_id)
```

### Production Context

```python
# Add production context to traces
mlflow.update_current_trace(
    tags={
        "mlflow.trace.session": session_id,
        "mlflow.trace.user": user_id,
        "environment": "production"
    }
)
```

## Model Registry

### Register and Deploy Models

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# Register during training
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="MyModel"
    )

# Set alias for deployment
client.set_registered_model_alias(
    name="MyModel",
    alias="champion",
    version=1
)

# Load model by alias
model = mlflow.pyfunc.load_model("models:/MyModel@champion")
```

### Model Versioning

```python
# Load specific version
model = mlflow.pyfunc.load_model("models:/MyModel/2")

# Load by stage
model = mlflow.pyfunc.load_model("models:/MyModel/Production")

# Transition model stage
client.transition_model_version_stage(
    name="MyModel",
    version=2,
    stage="Production"
)
```

## MCP Integration

MLflow has MCP support for trace operations:

### Installation

```bash
# Install as UV tool (recommended)
uv tool install "mlflow[genai,mcp]>=2.19.0"

# Or add to project
uv add "mlflow[genai,mcp]>=2.19.0"
```

### Starting MLflow MCP Server

```bash
# Run directly (after uv tool install)
mlflow mcp run

# With custom tracking URI
MLFLOW_TRACKING_URI=sqlite:///mlruns.db mlflow mcp run
```

### Claude Code Configuration

**.mcp.json (project configuration):**
```json
{
  "mcpServers": {
    "mlflow": {
      "command": "mlflow",
      "args": ["mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "sqlite:///mlruns.db"
      }
    }
  }
}
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | MLflow tracking server URL or sqlite path |
| `MLFLOW_EXPERIMENT_ID` | No | Default experiment ID |
| `DATABRICKS_HOST` | For Databricks | Workspace URL |
| `DATABRICKS_TOKEN` | For Databricks | Personal access token |

### MCP Tools Available

The MLflow MCP server exposes these tools:

| Tool | Purpose |
|------|---------|
| `search_traces` | Search traces with filters (experiment_id, tags, timestamps) |
| `get_trace` | Get detailed trace info including spans, inputs, outputs |
| `log_feedback` | Log feedback scores (accuracy, quality, custom) |
| `log_expectation` | Log expected values for trace evaluation |
| `evaluate_traces` | Run automated evaluation with scorers |
| `list_scorers` | List available evaluation scorers |
| `register_llm_judge` | Create custom LLM-based scorer |
| `set_trace_tag` / `delete_trace_tag` | Manage trace metadata |
| `delete_traces` | Clean up traces by criteria |

**Workflow Example:**
```
1. search_traces → Find traces to evaluate
2. evaluate_traces → Run built-in scorers (Correctness, Safety, etc.)
3. log_feedback → Add human feedback
4. get_trace → Inspect detailed results
```

## Best Practices

1. **Organize Experiments**: Use hierarchical naming and tags
2. **Version Everything**: Use Git versioning for GenAI apps
3. **Add Production Context**: Always include session, user, and environment info
4. **Monitor Costs**: Track token usage and estimated costs
5. **Regular Evaluation**: Run evaluations with multiple scorers

## Cross-Skill Integration

### Marimo Dashboard for Experiments

Build interactive experiment dashboards:

```python
import marimo as mo
import mlflow

# Experiment selector
experiments = mlflow.search_experiments()
exp_select = mo.ui.dropdown(
    options={e.name: e.experiment_id for e in experiments},
    label="Select Experiment"
)

# Display runs with filtering
runs_df = mlflow.search_runs(experiment_ids=[exp_select.value])
mo.ui.table(runs_df, selection="single", label="Experiment Runs")
```

### PINA Training Tracking

Track physics-informed neural network training:

```python
import mlflow
from pina import Trainer
from pina.callbacks import MetricTracker

mlflow.set_experiment("pina-experiments")

with mlflow.start_run():
    mlflow.log_params({"layers": [64, 64], "activation": "Tanh"})

    trainer = Trainer(solver, max_epochs=1000, callbacks=[MetricTracker()])
    trainer.train()

    # Log PINA metrics
    for key, value in trainer.callback_metrics.items():
        mlflow.log_metric(key, value)

    mlflow.pytorch.log_model(solver.model, "pinn")
```

### Using context7 for Documentation

Query up-to-date MLflow documentation directly:

```
# context7 Library IDs (no resolve needed):
# - /mlflow/mlflow (official docs, 9559 snippets)
# - /websites/mlflow (website docs, 36205 snippets)

# Example: query-docs("/mlflow/mlflow", "mlflow.trace decorator usage")
```

## When to Use This Skill

✅ **Use MLflow when:**
- Tracking ML experiments and hyperparameters
- Building and deploying LLM/GenAI applications
- Managing model lifecycle and versioning
- Comparing model performance across iterations
- Need production observability for GenAI apps
- Collaborating on ML projects

❌ **Don't use MLflow when:**
- Simple one-off scripts without iteration
- No need for experiment tracking or model versioning
- Using platform-specific tools (e.g., SageMaker Experiments)

## Reference Documentation

For detailed guides, see the references folder:

- **[GenAI Tracking](./references/genai_tracking.md)**: Complete LLM tracking guide with all providers, Git versioning, token tracking, and trace querying
- **[Framework Integrations](./references/framework_integrations.md)**: LangChain, LlamaIndex, DSPy, CrewAI, and ML framework integration examples
- **[Evaluation](./references/evaluation.md)**: GenAI evaluation, built-in scorers, custom scorers, and model comparison
- **[Production Deployment](./references/production_deployment.md)**: FastAPI integration, async logging, Kubernetes deployment, and monitoring
- **[Model Registry](./references/model_registry.md)**: Complete model registry guide with versioning, aliases, stages, and deployment patterns

## Example Templates

Ready-to-use templates in the examples folder:

- **[langchain_tracking.py](./examples/langchain_tracking.py)**: LangChain with MLflow autologging - chains, agents, and production patterns
- **[fastapi_tracing.py](./examples/fastapi_tracing.py)**: FastAPI production tracing with streaming, batch processing, and health checks
- **[hyperparameter_tuning.py](./examples/hyperparameter_tuning.py)**: Grid search, random search, Optuna integration, and nested runs

## Resources

- Documentation: https://mlflow.org/docs/latest/
- GitHub: https://github.com/mlflow/mlflow
- GenAI Guide: https://mlflow.org/docs/latest/genai/
- Tracking API: https://mlflow.org/docs/latest/tracking.html
- Model Registry: https://mlflow.org/docs/latest/model-registry.html
