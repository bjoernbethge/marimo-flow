# GenAI & LLM Tracking with MLflow

Complete guide for tracking LLM applications with MLflow's GenAI features.

## Autologging for LLM Providers

MLflow provides automatic tracing for major LLM frameworks:

```python
import mlflow

# OpenAI
mlflow.openai.autolog()

# Anthropic
mlflow.anthropic.autolog()

# LangChain
mlflow.langchain.autolog()

# LlamaIndex
mlflow.llama_index.autolog()

# DSPy
mlflow.dspy.autolog()

# LiteLLM (multi-provider)
mlflow.litellm.autolog()

# CrewAI
mlflow.crewai.autolog()

# Agno
mlflow.agno.autolog()
```

**What gets logged automatically:**
- ✅ Tokens (input, output, total)
- ✅ Latency and cost
- ✅ Full request/response
- ✅ Tool calls and function invocations
- ✅ Exceptions and errors

## Manual Tracing

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

## Git-Based Version Tracking

Track GenAI application versions with Git integration:

```python
import mlflow
import mlflow.genai

# Enable Git version tracking
context = mlflow.genai.enable_git_model_versioning()

# Enable autologging
mlflow.openai.autolog()

# All traces are now linked to Git commit
@mlflow.trace
def customer_support_agent(question: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful support agent."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

# Traces automatically versioned with Git commit hash
```

## Token Usage Tracking

```python
# After running an LLM call with autologging enabled
trace = mlflow.get_trace(trace_id=last_trace_id)

# Total usage
total_usage = trace.info.token_usage
print(f"Input tokens: {total_usage['input_tokens']}")
print(f"Output tokens: {total_usage['output_tokens']}")
print(f"Total tokens: {total_usage['total_tokens']}")

# Per-span usage
for span in trace.data.spans:
    if usage := span.get_attribute("mlflow.chat.tokenUsage"):
        print(f"{span.name}: {usage['total_tokens']} tokens")
```

## Search and Query Traces

```python
import mlflow

# Get active model for linking traces
active_model_id = mlflow.get_active_model_id()

# Search traces for specific model
traces = mlflow.search_traces(model_id=active_model_id)

# Filter by tags
user_traces = mlflow.search_traces(
    experiment_ids=[experiment_id],
    filter_string="tags.`mlflow.trace.user` = 'user-123'",
    max_results=100
)

# Filter by session
session_traces = mlflow.search_traces(
    filter_string="tags.`mlflow.trace.session` = 'session-abc'",
    max_results=100
)
```

## LLM Agent with Tools

```python
import mlflow
import json

@mlflow.trace(span_type="AGENT")
def run_agent_with_tools(question: str):
    messages = [{"role": "user", "content": question}]

    # First LLM call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=available_tools
    )

    # Handle tool calls
    if tool_calls := response.choices[0].tool_calls:
        for tool_call in tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Second LLM call with tool results
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

    return response.choices[0].content
```

## Version Everything

```python
# Use Git versioning for GenAI apps
with mlflow.genai.enable_git_model_versioning():
    # Your app code
    pass

# Tag with version info
mlflow.set_tag("app_version", "1.0.0")
mlflow.set_tag("git_commit", git_hash)
```

## Monitor Costs

```python
# Track token usage and costs
trace = mlflow.get_trace(trace_id)
tokens = trace.info.token_usage

cost_per_token = 0.00002  # Example rate
estimated_cost = tokens['total_tokens'] * cost_per_token

mlflow.log_metric("estimated_cost", estimated_cost)
```
