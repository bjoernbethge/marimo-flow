# MCP Integration Reference

Complete guide to marimo's Model Context Protocol (MCP) integration.

## Overview

marimo has dual MCP support:
1. **marimo as MCP Server**: Expose notebook data and operations as AI tools
2. **marimo as MCP Client**: Connect to MCP servers for enhanced documentation
3. **marimo-docs MCP Server**: Dedicated documentation access server

## 1. marimo as MCP Server

Run marimo notebooks as MCP servers to expose AI tools for programmatic access.

### Installation

```bash
# Install marimo as UV tool with all features (recommended)
uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"

# Or install in project
uv add "marimo[mcp]"
```

### Starting MCP Server

```bash
# Run as MCP server (the --mcp flag is hidden but functional)
# After installing with uv tool install:
marimo edit --mcp --no-token --port 2718 --headless

# Or with uvx (one-off)
uvx "marimo[mcp]" edit --mcp --no-token --port 2718 --headless

# The MCP endpoint will be available at:
# http://127.0.0.1:2718/mcp/server
```

### Integration with Claude Code

**Option 1: CLI command (one-time)**
```bash
claude mcp add --transport http marimo http://127.0.0.1:2718/mcp/server
```

**Option 2: Project .mcp.json configuration**
```json
{
  "mcpServers": {
    "marimo": {
      "type": "http",
      "url": "http://127.0.0.1:2718/mcp/server"
    }
  }
}
```

**Option 3: SessionStart hook (auto-start)**
```json
// .claude/settings.json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "powershell -ExecutionPolicy Bypass -File scripts/start-marimo-mcp.ps1"
      }]
    }]
  }
}
```

### Important Notes

- The `--mcp` flag is **hidden** in `marimo edit --help` but fully functional
- Use `--headless` for background server mode
- Use `--no-token` for local development only (adds authentication in production)
- Default port is dynamic; specify `--port 2718` for consistent configuration

### Available Tools

When running as MCP server, marimo exposes these tools:

#### get_active_notebooks
Lists all currently open notebook sessions.

```json
{
  "notebooks": [
    {
      "path": "/path/to/notebook.py",
      "session_id": "abc123",
      "status": "running"
    }
  ]
}
```

#### get_lightweight_cell_map
Gets cell structure and dependency graph without executing code.

```json
{
  "cells": [
    {
      "cell_id": "cell_0",
      "code": "import pandas as pd",
      "defines": ["pd"],
      "uses": []
    },
    {
      "cell_id": "cell_1",
      "code": "df = pd.read_csv('data.csv')",
      "defines": ["df"],
      "uses": ["pd"]
    }
  ]
}
```

#### get_cell_runtime_data
Extracts tracebacks and variable states from running cells.

```json
{
  "cell_id": "cell_1",
  "status": "error",
  "traceback": "FileNotFoundError: data.csv not found",
  "variables": {}
}
```

#### get_tables_and_variables
Access data frames and variables from notebook scope.

```json
{
  "variables": {
    "df": {
      "type": "DataFrame",
      "shape": [100, 5],
      "columns": ["A", "B", "C", "D", "E"]
    },
    "count": {
      "type": "int",
      "value": 42
    }
  }
}
```

#### get_database_tables
Inspect database connections and available tables.

```json
{
  "connections": [
    {
      "name": "main_db",
      "type": "postgresql",
      "tables": ["users", "orders", "products"]
    }
  ]
}
```

#### get_notebook_errors
Find all failing cells and their error messages.

```json
{
  "errors": [
    {
      "cell_id": "cell_3",
      "error_type": "ValueError",
      "message": "Invalid value for parameter x"
    }
  ]
}
```

#### lint_notebook
Lint a notebook to check for issues using marimo's internal linting engine.

```json
{
  "diagnostics": [
    {
      "cell_id": "cell_2",
      "severity": "error",
      "message": "Variable 'x' is used before definition",
      "category": "breaking"
    }
  ],
  "counts": {
    "breaking": 1,
    "runtime": 0,
    "formatting": 2
  }
}
```

Categories:
- **breaking**: Problems preventing notebook execution
- **runtime**: Problems causing unexpected behavior
- **formatting**: Code style and formatting issues

#### get_cell_outputs
Get cell execution output including visual display and console streams.

```json
{
  "cell_id": "cell_1",
  "output": {
    "mimetype": "text/html",
    "data": "<div>Chart content</div>"
  },
  "console": {
    "stdout": "Processing complete",
    "stderr": ""
  }
}
```

### Use Cases

**Debugging Notebooks Programmatically**
```python
# Claude can help debug by inspecting errors
errors = get_notebook_errors()
for error in errors:
    print(f"Cell {error['cell_id']}: {error['message']}")
```

**Inspecting Cell Dependencies**
```python
# Understand notebook structure
cell_map = get_lightweight_cell_map()
for cell in cell_map['cells']:
    print(f"Cell defines: {cell['defines']}")
    print(f"Cell uses: {cell['uses']}")
```

**Accessing Runtime Data**
```python
# Get current variable values
data = get_tables_and_variables()
df_info = data['variables']['df']
print(f"DataFrame shape: {df_info['shape']}")
```

**Auditing Notebook State**
```python
# Check all notebooks and their status
notebooks = get_active_notebooks()
for nb in notebooks:
    print(f"{nb['path']}: {nb['status']}")
```

## 2. marimo as MCP Client

marimo can connect to MCP servers for enhanced functionality, particularly documentation access.

### Configuration

Add MCP servers in `.marimo.toml`:

```toml
[mcp]
# Use preset MCP servers
presets = ["marimo", "context7"]

# Or configure custom servers
[[mcp.servers.custom_server]]
command = "path/to/mcp-server"
args = ["--port", "8080"]
```

### Recommended MCP Servers

#### marimo Preset
Official marimo documentation and examples.

```toml
[mcp]
presets = ["marimo"]
```

Provides:
- Complete API documentation
- Usage examples
- Best practices
- Migration guides

#### context7 Preset
Version-specific library documentation.

```toml
[mcp]
presets = ["context7"]
```

Provides:
- Up-to-date package docs
- Version-specific APIs
- Code examples
- Compatibility info

### Custom MCP Servers

```toml
[[mcp.servers.my_docs]]
command = "python"
args = ["-m", "my_mcp_server"]
env = { API_KEY = "${MY_API_KEY}" }
```

## 3. marimo-docs MCP Server

Dedicated MCP server specifically for marimo documentation access.

### Available Tools

#### get_element_api
Get detailed API documentation for specific UI elements.

**Usage:**
```python
# Request docs for a specific element
docs = get_element_api("slider")
```

**Response:**
```json
{
  "element": "slider",
  "signature": "mo.ui.slider(start, stop, value=None, step=1, label='', debounce=False)",
  "description": "Create an interactive slider for numeric input",
  "parameters": [
    {
      "name": "start",
      "type": "float",
      "description": "Minimum value"
    },
    {
      "name": "stop",
      "type": "float",
      "description": "Maximum value"
    }
  ],
  "examples": [
    "slider = mo.ui.slider(0, 100, value=50, label='Threshold')"
  ]
}
```

#### search_api
Search across all marimo API documentation.

**Usage:**
```python
# Search for specific functionality
results = search_api("form validation")
```

**Response:**
```json
{
  "results": [
    {
      "title": "Form validation with mo.stop",
      "content": "Use mo.stop() to validate form inputs...",
      "url": "https://docs.marimo.io/guides/forms"
    },
    {
      "title": "Form submission",
      "content": "Wrap UI elements with .form() to gate execution...",
      "url": "https://docs.marimo.io/api/ui/form"
    }
  ]
}
```

### When to Use MCP Tools

Use MCP documentation tools when:
- ✅ Looking up unfamiliar marimo APIs
- ✅ Finding code examples for specific features
- ✅ Exploring parameter options for UI elements
- ✅ Researching best practices from official docs
- ✅ Understanding advanced reactive patterns
- ✅ Checking compatibility with specific packages

Don't use MCP tools when:
- ❌ Basic syntax is already known
- ❌ Working on non-marimo code
- ❌ Information available in local skill docs

## Integration Examples

### Claude Code + marimo MCP

```python
# In your marimo notebook
import marimo as mo
import pandas as pd

# Load data
df = pd.read_csv("data.csv")

# Claude Code can access this via MCP
# and help debug or analyze
```

**Claude Code can:**
- Inspect the notebook structure
- Access variable values (like `df`)
- Identify errors
- Suggest fixes
- Generate new cells

### Documentation Lookup Flow

```mermaid
User asks about UI element
    ↓
Claude checks local skill docs
    ↓
If needed, calls get_element_api("element_name")
    ↓
Returns detailed API docs + examples
    ↓
Claude provides comprehensive answer
```

### Debugging Flow

```mermaid
User reports notebook error
    ↓
Claude calls get_notebook_errors()
    ↓
Identifies failing cells
    ↓
Claude calls get_cell_runtime_data(cell_id)
    ↓
Analyzes traceback and variables
    ↓
Claude suggests fix
```

## Security Considerations

### MCP Server Mode

When running marimo as MCP server:

- **Only expose to trusted AI assistants**
- **Don't expose on public networks** (use localhost)
- **Be aware tools can access notebook data**
- **Use `--no-token` only in secure environments**
- **Consider network isolation** (firewall rules)

### Safe Practices

```bash
# ✅ SAFE: Localhost only
marimo edit notebook.py --mcp --host 127.0.0.1

# ⚠️ DANGEROUS: Exposed to network
marimo edit notebook.py --mcp --host 0.0.0.0

# ✅ SAFE: With authentication
marimo edit notebook.py --mcp

# ⚠️ LESS SECURE: Without authentication
marimo edit notebook.py --mcp --no-token
```

## Troubleshooting

### MCP Server Not Starting

```bash
# Check if MCP dependencies installed
pip show marimo | grep mcp

# Reinstall with MCP support
pip install "marimo[mcp]" --force-reinstall
```

### Claude Code Can't Connect

```bash
# Verify server is running
curl http://localhost:PORT/mcp/server

# Check firewall rules
# Ensure port is not blocked

# Try different port
marimo edit notebook.py --mcp --port 8888
```

### MCP Tools Not Available

```toml
# Verify .marimo.toml configuration
[mcp]
presets = ["marimo", "context7"]

# Restart marimo after config change
```

### Documentation Not Loading

```bash
# Clear MCP cache
rm -rf ~/.marimo/mcp_cache

# Restart marimo
marimo edit notebook.py
```

## Best Practices

1. **Start MCP server only when needed** (adds overhead)
2. **Use localhost for security** (don't expose to network)
3. **Enable MCP presets for better docs** (marimo, context7)
4. **Clean up MCP connections** (close unused servers)
5. **Document MCP usage** (for team collaboration)
6. **Test MCP tools** (verify before production use)
7. **Monitor MCP performance** (watch for slowdowns)

## Advanced Usage

### Custom MCP Tools

You can expose custom tools from your notebook:

```python
import marimo as mo

# Define custom tool
@mo.mcp.tool
def analyze_data(table_name: str):
    """Custom analysis tool exposed via MCP"""
    # Your analysis logic
    return results
```

### MCP with Authentication

```python
# Add authentication to MCP endpoint
@mo.mcp.require_auth
def sensitive_operation():
    """Requires authentication to access"""
    return protected_data
```

### MCP Monitoring

```python
# Track MCP tool usage
@mo.mcp.monitored
def expensive_operation():
    """Logged MCP tool calls"""
    return results
```

## Resources

- MCP Specification: https://modelcontextprotocol.io
- marimo MCP Docs: https://docs.marimo.io/guides/mcp
- Claude Code MCP: https://docs.anthropic.com/claude/mcp
