# Pydantic AI Toolsets Reference
**Version**: 1.84.1 | **Updated**: 2026-04-22

## Quick Reference

| Question | Answer |
|----------|--------|
| **FunctionToolset generic?** | Yes: `FunctionToolset[DepsT]` where `DepsT` is your dependency type |
| **Register tools?** | Decorator: `@ts.tool` (with ctx) / `@ts.tool_plain` (no ctx); method: `ts.add_function()` |
| **Mix MCPServer + FunctionToolset?** | Yes: `Agent(model, toolsets=[mcp_server, function_ts])` — both inherit `AgentToolset[DepsT]` |
| **RunContext in tools?** | `def my_tool(ctx: RunContext[DepsT], arg: str) -> str` — access via `ctx.deps` |
| **Pass deps at runtime?** | `agent.run_sync('prompt', deps=my_deps)` or async `agent.run_stream('prompt', deps=my_deps)` |
| **Streaming with MLflow?** | ⚠️ Use `run_stream_events()` for raw events or `stream_text()` to yield deltas only (see issue #640) |

---

## 1. FunctionToolset Construction

### Signature
```python
from pydantic_ai import FunctionToolset
from pydantic_ai.models import RunContext

ts = FunctionToolset[DepsT](
    tools: Sequence[Tool | Callable] = [],
    max_retries: int = 1,
    timeout: float | None = None,
    docstring_format: str = 'auto',
    id: str | None = None,
    instructions: str | None = None,
)
```

**Generic parameter `DepsT`**: Type of dependencies available to tools. Example: `FunctionToolset[User]` or `FunctionToolset[MyDeps]`.

### Registering Tools

**Decorator (with RunContext):**
```python
@ts.tool
async def my_tool(ctx: RunContext[DepsT], query: str) -> str:
    """Search for something.
    
    Args:
        query: The search term
    """
    return f"Results for {query} in {ctx.deps}"
```

**Decorator (plain, no context):**
```python
@ts.tool_plain
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

**Via constructor:**
```python
def plain_func(x: str) -> str:
    return x.upper()

ts = FunctionToolset[None](tools=[plain_func])
```

**Via `add_function()`:**
```python
ts = FunctionToolset[None]()
ts.add_function(plain_func)
```

**Sources:**
- [FunctionToolset API](https://pydantic.dev/docs/ai/api/pydantic-ai/toolsets/)
- [Toolsets Guide](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)

---

## 2. Passing Toolsets to Agent

### Signature
```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.mcp import MCPServerStreamableHTTP, MCPServerStdio

agent = Agent(
    model,
    deps_type=DepsT,
    toolsets: Sequence[AgentToolset[DepsT] | ToolsetFunc[DepsT]] | None = None,
)
```

### Mixing MCP + FunctionToolset

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.mcp import MCPServerStdio

# Function toolset
fn_ts = FunctionToolset[MyDeps]()

@fn_ts.tool
def search(ctx: RunContext[MyDeps], q: str) -> str:
    return f"Found: {q}"

# MCP server toolset
mcp_ts = MCPServerStdio(command=['node', 'server.js'])

# Combined agent
agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
    toolsets=[fn_ts, mcp_ts],  # Both AgentToolset subclasses
)
```

**Key**: Both `FunctionToolset` and `MCPServer*` inherit from `AgentToolset[DepsT]`, so they are type-compatible in the `toolsets` list.

**Sources:**
- [Agent Constructor](https://pydantic.dev/docs/ai/api/pydantic-ai/agent/)
- [Toolsets Guide](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)

---

## 3. RunContext in Toolset Tools

### Accessing Dependencies

When you declare `@ts.tool` on a function with `ctx: RunContext[DepsT]` as the first parameter, the agent automatically injects the runtime dependencies:

```python
from dataclasses import dataclass
from pydantic_ai import FunctionToolset, Agent
from pydantic_ai.models import RunContext

@dataclass
class FlowDeps:
    api_key: str
    db_client: object

flow_ts = FunctionToolset[FlowDeps]()

@flow_ts.tool
async def fetch_data(ctx: RunContext[FlowDeps], query: str) -> str:
    """Fetch data from database.
    
    Args:
        query: SQL query
    """
    # ctx.deps is typed as FlowDeps
    client = ctx.deps.db_client
    key = ctx.deps.api_key
    return f"Data using {key}: {query}"
```

### Type Safety Requirement

For `ctx.deps` to be properly typed, the **Agent must have `deps_type` set**:

```python
agent = Agent(
    model,
    deps_type=FlowDeps,  # ✓ Required for type safety
    toolsets=[flow_ts],
)
```

Without `deps_type`, the agent accepts tools with any `RunContext[T]`, but static type checkers will warn.

**Sources:**
- [RunContext & Dependencies](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Function Tools](https://pydantic.dev/docs/ai/tools-toolsets/tools/)

---

## 4. Passing Dependencies at Runtime

### Sync Execution
```python
from dataclasses import dataclass

@dataclass
class MyDeps:
    user_id: str
    session: object

deps = MyDeps(user_id="123", session=some_session)

result = agent.run_sync(
    "Fetch user data",
    deps=deps,
)
```

### Async Execution
```python
async with agent.run_stream(
    "Fetch user data",
    deps=deps,
) as run:
    async for text in run.stream_text():
        print(text)
```

### Sharing Dependencies Across Lead + Sub-Agents

Use the same `deps` object when invoking sub-agents from a lead agent's tool:

```python
@lead_ts.tool
async def invoke_subagent(
    ctx: RunContext[FlowDeps],
    prompt: str,
) -> str:
    """Call sub-agent with same deps context."""
    result = await subagent.run(
        prompt,
        deps=ctx.deps,  # Share lead agent's deps
    )
    return result.data
```

**Sources:**
- [Agent Methods](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [Dependencies](https://ai.pydantic.dev/dependencies/)

---

## 5. Streaming with MLflow Autolog

### Known Issue: Circular Reference in `run_stream()`

When using `mlflow.pydantic_ai.autolog()` with `agent.run_stream()`, the stream may yield non-JSON-serializable MLflow event objects, causing "Circular reference detected" errors in `mo.ui.chat` callbacks.

### Recommended Pattern: Use `stream_text()` Only

```python
import mlflow
from pydantic_ai import Agent

mlflow.pydantic_ai.autolog()

async with agent.run_stream("Your prompt", deps=deps) as run:
    async for text_delta in run.stream_text():  # ✓ Yields str only
        # Safe to pass to mo.ui.chat callback
        yield text_delta
```

### Alternative: Use `run_stream_events()` for Control

For fine-grained event handling without autolog serialization issues:

```python
from pydantic_ai.models import StreamEvent

async for event in agent.run_stream_events(
    "Your prompt",
    deps=deps,
):
    if isinstance(event, StreamEvent):
        # Handle raw event
        if hasattr(event, 'text'):
            yield event.text
```

### Avoid: Full `run_stream()` Result Serialization

```python
# ❌ Do NOT yield the full run result:
async with agent.run_stream("prompt", deps=deps) as run:
    yield run  # ← Circular reference with MLflow autolog
```

**Sources:**
- [Streaming Tool Calls Issue #640](https://github.com/pydantic/pydantic-ai/issues/640)
- [Marimo Chat UI](https://docs.marimo.io/api/inputs/chat/)
- [Pydantic AI Streaming](https://datastud.dev/posts/pydantic-ai-streaming/)

---

## 6. Minimal Code Snippets

### (a) FunctionToolset[FlowDeps] with one tool

```python
from dataclasses import dataclass
from pydantic_ai import FunctionToolset
from pydantic_ai.models import RunContext

@dataclass
class FlowDeps:
    user_id: str

flow_ts = FunctionToolset[FlowDeps]()

@flow_ts.tool
async def get_user_info(
    ctx: RunContext[FlowDeps],
    field: str,
) -> str:
    """Get user information.
    
    Args:
        field: User field name
    """
    return f"User {ctx.deps.user_id} {field}"
```

### (b) Attach FunctionToolset + MCP to Agent

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

agent = Agent(
    'openai:gpt-4o',
    deps_type=FlowDeps,
    toolsets=[
        flow_ts,  # FunctionToolset[FlowDeps]
        MCPServerStdio(command=['python', 'mcp_server.py']),
    ],
)
```

### (c) Invoke agent with deps

```python
# Sync
result = agent.run_sync(
    "Get user info for field: name",
    deps=FlowDeps(user_id="alice"),
)
print(result.data)

# Async with streaming
async with agent.run_stream(
    "Get user info for field: email",
    deps=FlowDeps(user_id="bob"),
) as run:
    async for text in run.stream_text():
        print(text, end='')
```

**Sources:**
- [Official Docs](https://pydantic.dev/docs/ai/)
- [GitHub Examples](https://github.com/pydantic/pydantic-ai)

---

## Summary

| Task | API |
|------|-----|
| Create typed toolset | `FunctionToolset[DepsT]()` |
| Add tool w/ context | `@ts.tool` → `ctx: RunContext[DepsT]` |
| Mix MCP + functions | `Agent(toolsets=[mcp_ts, fn_ts])` |
| Pass deps at runtime | `agent.run_sync/run_stream(..., deps=my_deps)` |
| Stream safely with MLflow | `run.stream_text()` (not full result) |
| Type-safe RunContext | Require `Agent(deps_type=DepsT)` |

---

## References

- [Pydantic AI Docs](https://pydantic.dev/docs/ai/)
- [Pydantic AI API: Agent](https://pydantic.dev/docs/ai/api/pydantic-ai/agent/)
- [Pydantic AI API: Toolsets](https://pydantic.dev/docs/ai/api/pydantic-ai/toolsets/)
- [GitHub: pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai)
