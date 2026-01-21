#!/usr/bin/env bash
#
# Verify MCP setup for marimo-flow
#
# This script checks if all MCP servers are properly configured
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

MARIMO_PORT=2718
MLFLOW_PORT=5000
ERRORS=0

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           MCP Setup Verification for marimo-flow             ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Helper functions
check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

check_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# 1. Check dependencies
echo -e "${BLUE}[1/8] Checking Dependencies${NC}"

if command -v uv &> /dev/null; then
    check_pass "uv is installed ($(uv --version))"
else
    check_fail "uv is not installed"
fi

if command -v python3 &> /dev/null; then
    check_pass "Python is installed ($(python3 --version))"
else
    check_fail "Python is not installed"
fi

if command -v curl &> /dev/null; then
    check_pass "curl is installed"
else
    check_fail "curl is not installed"
fi

echo ""

# 2. Check project structure
echo -e "${BLUE}[2/8] Checking Project Structure${NC}"

if [ -f "pyproject.toml" ]; then
    check_pass "pyproject.toml exists"
else
    check_fail "pyproject.toml not found"
fi

if [ -f ".marimo.toml" ]; then
    check_pass ".marimo.toml exists"

    # Check for MCP presets
    if grep -q "presets.*marimo" ".marimo.toml"; then
        check_pass "Marimo MCP preset configured"
    else
        check_warn "Marimo MCP preset not found in .marimo.toml"
    fi

    if grep -q "presets.*context7" ".marimo.toml"; then
        check_pass "Context7 MCP preset configured"
    else
        check_warn "Context7 MCP preset not found in .marimo.toml"
    fi

    if grep -q "mcpServers.mlflow" ".marimo.toml"; then
        check_pass "MLflow MCP server configured"
    else
        check_warn "MLflow MCP server not configured in .marimo.toml"
    fi
else
    check_fail ".marimo.toml not found"
fi

if [ -d "examples" ]; then
    check_pass "examples/ directory exists"
    NOTEBOOK_COUNT=$(find examples -maxdepth 1 -name "*.py" -type f | wc -l)
    check_info "Found $NOTEBOOK_COUNT notebooks in examples/"
else
    check_fail "examples/ directory not found"
fi

echo ""

# 3. Check data directories
echo -e "${BLUE}[3/8] Checking Data Directories${NC}"

if [ -d "data/mlflow/db" ]; then
    check_pass "MLflow database directory exists"
else
    check_warn "MLflow database directory not found (will be created on first run)"
fi

if [ -d "data/mlflow/artifacts" ]; then
    check_pass "MLflow artifacts directory exists"
else
    check_warn "MLflow artifacts directory not found (will be created on first run)"
fi

echo ""

# 4. Check Python dependencies
echo -e "${BLUE}[4/8] Checking Python Dependencies${NC}"

if [ -d ".venv" ]; then
    check_pass "Virtual environment exists"
else
    check_warn "Virtual environment not found (run: uv sync)"
fi

# Check if marimo is installed
if uv run python -c "import marimo" 2>/dev/null; then
    MARIMO_VERSION=$(uv run python -c "import marimo; print(marimo.__version__)")
    check_pass "marimo is installed (v${MARIMO_VERSION})"

    # Check for MCP support
    if uv run python -c "import marimo; assert hasattr(marimo, 'mcp')" 2>/dev/null; then
        check_pass "marimo MCP support detected"
    else
        check_warn "marimo MCP support not detected (needs marimo[mcp])"
    fi
else
    check_fail "marimo is not installed (run: uv sync)"
fi

# Check if mlflow is installed
if uv run python -c "import mlflow" 2>/dev/null; then
    MLFLOW_VERSION=$(uv run python -c "import mlflow; print(mlflow.__version__)")
    check_pass "mlflow is installed (v${MLFLOW_VERSION})"
else
    check_fail "mlflow is not installed (run: uv sync)"
fi

echo ""

# 5. Check running services
echo -e "${BLUE}[5/8] Checking Running Services${NC}"

if lsof -i :${MLFLOW_PORT} > /dev/null 2>&1; then
    check_pass "MLflow server is running on port ${MLFLOW_PORT}"

    if curl -s -f "http://localhost:${MLFLOW_PORT}/health" > /dev/null 2>&1; then
        check_pass "MLflow server is responding"
    else
        check_warn "MLflow server is running but not responding"
    fi
else
    check_warn "MLflow server is not running (start with: ./scripts/start-dev.sh)"
fi

if lsof -i :${MARIMO_PORT} > /dev/null 2>&1; then
    check_pass "Marimo server is running on port ${MARIMO_PORT}"

    if curl -s -f "http://localhost:${MARIMO_PORT}" > /dev/null 2>&1; then
        check_pass "Marimo server is responding"
    else
        check_warn "Marimo server is running but not responding"
    fi

    if curl -s -f "http://localhost:${MARIMO_PORT}/mcp/server" > /dev/null 2>&1; then
        check_pass "Marimo MCP endpoint is accessible"
    else
        check_warn "Marimo MCP endpoint not accessible (restart with --mcp flag)"
    fi
else
    check_warn "Marimo server is not running (start with: ./scripts/start-dev.sh)"
fi

echo ""

# 6. Check IDE configurations
echo -e "${BLUE}[6/8] Checking IDE Configurations${NC}"

if [ -f ".vscode/settings.json" ]; then
    check_pass "VSCode settings.json exists"
else
    check_warn "VSCode settings.json not found"
fi

if [ -f ".vscode/tasks.json" ]; then
    check_pass "VSCode tasks.json exists"
else
    check_warn "VSCode tasks.json not found"
fi

if [ -f ".cursor/settings.json" ]; then
    check_pass "Cursor settings.json exists"
else
    check_warn "Cursor settings.json not found"
fi

if [ -f ".cursorrules" ]; then
    check_pass ".cursorrules exists"
else
    check_warn ".cursorrules not found"
fi

echo ""

# 7. Check GitHub Actions
echo -e "${BLUE}[7/8] Checking GitHub Actions${NC}"

if [ -f ".github/workflows/claude-code.yml" ]; then
    check_pass "Claude Code workflow exists"

    # Check for MCP configuration in workflow
    if grep -q "mcp_servers:" ".github/workflows/claude-code.yml"; then
        check_pass "MCP servers configured in workflow"
    else
        check_warn "MCP servers not configured in workflow"
    fi
else
    check_warn "Claude Code workflow not found"
fi

echo ""

# 8. Check scripts
echo -e "${BLUE}[8/8] Checking Scripts${NC}"

if [ -f "scripts/start-dev.sh" ]; then
    if [ -x "scripts/start-dev.sh" ]; then
        check_pass "start-dev.sh is executable"
    else
        check_warn "start-dev.sh is not executable (run: chmod +x scripts/start-dev.sh)"
    fi
else
    check_fail "start-dev.sh not found"
fi

if [ -f "scripts/setup-claude-desktop.sh" ]; then
    if [ -x "scripts/setup-claude-desktop.sh" ]; then
        check_pass "setup-claude-desktop.sh is executable"
    else
        check_warn "setup-claude-desktop.sh is not executable (run: chmod +x scripts/setup-claude-desktop.sh)"
    fi
else
    check_warn "setup-claude-desktop.sh not found"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Summary
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ MCP Setup Verification Passed!${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Start services: ./scripts/start-dev.sh"
    echo "2. Configure Claude Desktop: ./scripts/setup-claude-desktop.sh"
    echo "3. Open your IDE (VSCode/Cursor) and start coding!"
    echo ""
    echo -e "${BLUE}Test MCP:${NC}"
    echo "- Claude Desktop: 'List active marimo notebooks'"
    echo "- Marimo: http://localhost:${MARIMO_PORT}"
    echo "- MLflow: http://localhost:${MLFLOW_PORT}"
else
    echo -e "${RED}✗ MCP Setup Verification Failed with $ERRORS error(s)${NC}"
    echo ""
    echo -e "${YELLOW}Please fix the errors above and run this script again.${NC}"
    exit 1
fi
