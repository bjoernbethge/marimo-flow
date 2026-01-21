#!/usr/bin/env bash
#
# Setup MCP servers for Claude Desktop
#
# This script helps configure Claude Desktop to use marimo-flow MCP servers
#

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARIMO_PORT=2718
MLFLOW_PORT=5000

# Detect OS and config location
if [[ "$OSTYPE" == "darwin"* ]]; then
    CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    CLAUDE_CONFIG="$HOME/.config/Claude/claude_desktop_config.json"
else
    echo -e "${YELLOW}⚠${NC} Unsupported OS: $OSTYPE"
    echo "Please manually configure Claude Desktop using the JSON below."
    exit 1
fi

cat << EOF
${BLUE}
╔═══════════════════════════════════════════════════════════════╗
║           Claude Desktop MCP Configuration Setup             ║
╚═══════════════════════════════════════════════════════════════╝
${NC}

${GREEN}Configuration file:${NC} ${CLAUDE_CONFIG}

${GREEN}MCP Servers for marimo-flow:${NC}

1. ${BLUE}Marimo MCP${NC} - Notebook introspection
   Tools: get_active_notebooks, get_notebook_errors, get_cell_runtime_data

2. ${BLUE}Context7 MCP${NC} - Live documentation for Python libraries
   Tools: search_docs, get_library_docs

3. ${BLUE}MLflow MCP${NC} - Experiment tracking and model management
   Tools: search_experiments, search_runs, log_metric, list_models

${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}Configuration JSON:${NC}

{
  "mcpServers": {
    "marimo": {
      "transport": "http",
      "url": "http://localhost:${MARIMO_PORT}/mcp/server"
    },
    "context7": {
      "transport": "sse",
      "url": "https://context7.com/api/v1/mcp/sse"
    },
    "mlflow": {
      "transport": "stdio",
      "command": "$(which mlflow || echo 'mlflow')",
      "args": ["mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:${MLFLOW_PORT}",
        "PATH": "${PATH}"
      }
    }
  }
}

${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}

${GREEN}Setup Steps:${NC}

1. ${BLUE}Start Services${NC}
   ${YELLOW}\$ cd ${PROJECT_ROOT}${NC}
   ${YELLOW}\$ ./scripts/start-dev.sh${NC}

2. ${BLUE}Copy Configuration${NC}
   Open: ${CLAUDE_CONFIG}
   Add the JSON above to your config file

3. ${BLUE}Restart Claude Desktop${NC}
   Quit and reopen Claude Desktop to load MCP servers

4. ${BLUE}Test MCP Connection${NC}
   In Claude Desktop, try:
   - "List all active marimo notebooks"
   - "Search MLflow experiments"
   - "Get documentation for polars DataFrame"

${GREEN}Automatic Setup:${NC}

Would you like to automatically update your Claude Desktop config?
This will backup your existing config to:
  ${CLAUDE_CONFIG}.backup

EOF

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}⚠${NC} Setup cancelled. Please configure manually using the JSON above."
    exit 0
fi

# Create config directory if it doesn't exist
mkdir -p "$(dirname "${CLAUDE_CONFIG}")"

# Backup existing config
if [ -f "${CLAUDE_CONFIG}" ]; then
    cp "${CLAUDE_CONFIG}" "${CLAUDE_CONFIG}.backup"
    echo -e "${GREEN}✓${NC} Backed up existing config to ${CLAUDE_CONFIG}.backup"
fi

# Generate new config
cat > "${CLAUDE_CONFIG}" << EOF
{
  "mcpServers": {
    "marimo": {
      "transport": "http",
      "url": "http://localhost:${MARIMO_PORT}/mcp/server"
    },
    "context7": {
      "transport": "sse",
      "url": "https://context7.com/api/v1/mcp/sse"
    },
    "mlflow": {
      "transport": "stdio",
      "command": "$(which mlflow || echo 'mlflow')",
      "args": ["mcp", "run"],
      "env": {
        "MLFLOW_TRACKING_URI": "http://localhost:${MLFLOW_PORT}",
        "PATH": "${PATH}"
      }
    }
  }
}
EOF

echo -e "${GREEN}✓${NC} Claude Desktop config updated!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Start services: ./scripts/start-dev.sh"
echo "2. Restart Claude Desktop"
echo "3. Test MCP connection with: 'List active marimo notebooks'"
echo ""
echo -e "${GREEN}Done!${NC}"
