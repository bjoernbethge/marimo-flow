# MCP Setup für AI-Human Notebooks

## Overview

marimo-flow nutzt MCP (Model Context Protocol) für echte AI-Human Kooperation in Notebooks. Das erlaubt Claude Code, direkt mit marimo Notebooks zu arbeiten - erstellen, editieren, ausführen.

## Development Setup

### 1. marimo mit MCP starten

```bash
marimo edit --mcp
```

Output wird ungefähr so aussehen:
```
🔗 Experimental MCP server configuration
➜  MCP server URL: http://127.0.0.1:2718/mcp/server?access_token=4cbBpaRcxA3MlheAt0vpCg
➜  Add to Claude Code: claude mcp add --transport http marimo http://127.0.0.1:2718/mcp/server?access_token=4cbBpaRcxA3MlheAt0vpCg
```

### 2. MCP in Claude Code registrieren

Kopiere die `marimo` Command aus der marimo Output oben:

```bash
claude mcp add --transport http marimo http://127.0.0.1:2718/mcp/server?access_token=YOUR_TOKEN_HERE
```

⚠️ **Wichtig**: Der Token wechselt bei jedem marimo Neustart!

### 3. Mit Claude Code arbeiten

Sobald MCP registriert ist, kannst du:

```bash
claude code
```

Dann in Claude Code:
- Neue Notebooks erstellen (marimo MCP wird vorgeschlagen)
- Bestehende Notebooks editieren
- Notebooks mit Claude zusammen weiterentwickeln

## Automatisiertes Setup (empfohlen für Dev)

Nutze das Start-Script um Token automatisch zu extrahieren:

```bash
./scripts/start-marimo-mcp.sh
```

Das Script:
- Startet marimo mit MCP
- Extracted den aktuellen Token
- Gibt dir die Claude Code Command
- Behält den Server am laufen

## Was macht das MCP?

Das marimo MCP exponiert diese Tools an Claude Code:

| Tool | Beschreibung |
|------|---|
| `list_notebooks` | Liste alle marimo Notebooks auf |
| `read_notebook` | Lese Notebook Content |
| `create_notebook` | Erstelle neues Notebook |
| `edit_notebook` | Editiere Cells in Notebook |
| `run_notebook` | Führe Notebook aus |
| `get_notebook_state` | Hole aktuelle Cell States |

## Beispiel Workflow

```
You: "Create a data analysis notebook for CSV files"

Claude Code:
→ Uses marimo MCP to create new notebook
→ Adds cells for: file upload, data loading, profiling, visualization
→ Runs notebook to show preview
→ Asks for feedback

You: "Add a summary statistics cell"

Claude Code:
→ Uses marimo MCP to edit notebook
→ Adds new cell with polars .describe()
→ Runs it to show the output
```

## Production Deployment

Für Production (nicht localhost):

1. Setze `MARIMO_MCP_SERVER` Environment Variable
2. Nutze HTTPS statt HTTP
3. Nutze starke Access Tokens
4. Limiting auf spezifische Claude Instances

```bash
export MARIMO_MCP_SERVER="https://your-domain.com:8000"
marimo edit --mcp --mcp-server-auth-token "your-secure-token"
```

## Troubleshooting

### MCP Server nicht erreichbar
```bash
# Check if marimo is still running
ps aux | grep marimo

# Restart if needed
./scripts/start-marimo-mcp.sh
```

### Token ungültig
Der Token ist an die marimo Session gebunden. Bei jedem Neustart:
1. Stoppe alte marimo Instance
2. Starte neue mit `./scripts/start-marimo-mcp.sh`
3. Kopiere neuen Token in Claude Code

### Connection refused
Stelle sicher dass:
- marimo läuft: `marimo edit --mcp`
- Port 2718 nicht blockiert ist
- Firewall erlaubt localhost connections

## MCP Presets

In `.marimo.toml`:
```toml
[mcp]
presets = ["context7"]  # Hugging Face docs

# marimo wird dynamisch hinzugefügt wenn --mcp flag verwendet
```

Mehr MCP Servers können hinzugefügt werden:
- `mlflow`: MLflow Experiment Tracking
- `anthropic`: Claude API Integration
- `filesystem`: File Operations
- Custom MCPs: Externe APIs, Databases, etc.
