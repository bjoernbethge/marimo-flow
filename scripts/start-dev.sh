#!/usr/bin/env bash
#
# Start marimo-flow development environment with all MCP servers
#
# This script starts:
# 1. MLflow tracking server (port 5000)
# 2. Marimo notebook server with MCP (port 2718)
# 3. MLflow MCP server (stdio - for Claude Desktop/Code)
#
# Usage:
#   ./scripts/start-dev.sh          # Start all services
#   ./scripts/start-dev.sh --help   # Show help
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MLFLOW_PORT=5000
MARIMO_PORT=2718
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/mlflow"
DB_PATH="${DATA_DIR}/db/mlflow.db"
ARTIFACTS_PATH="${DATA_DIR}/artifacts"

# PID file for tracking background processes
PIDS_FILE="${PROJECT_ROOT}/.dev-pids"

# Help message
show_help() {
    cat << EOF
${BLUE}marimo-flow Development Environment${NC}

Starts all services needed for local development with MCP support.

${GREEN}Usage:${NC}
  ./scripts/start-dev.sh [OPTIONS]

${GREEN}Options:${NC}
  -h, --help              Show this help message
  -m, --mlflow-only       Start only MLflow server
  -r, --marimo-only       Start only Marimo server
  -s, --stop              Stop all running services
  --no-browser            Don't open browser windows

${GREEN}Services:${NC}
  MLflow Server:          http://localhost:${MLFLOW_PORT}
  Marimo Notebooks:       http://localhost:${MARIMO_PORT}
  MLflow MCP Server:      stdio (for Claude Desktop/Code)

${GREEN}MCP Servers Available:${NC}
  1. Marimo MCP          - Notebook introspection (active with --mcp flag)
  2. Context7 MCP        - Live documentation (configured in .marimo.toml)
  3. MLflow MCP          - Experiment tracking (start with: mlflow mcp run)

${GREEN}Examples:${NC}
  # Start everything
  ./scripts/start-dev.sh

  # Start only MLflow
  ./scripts/start-dev.sh --mlflow-only

  # Stop all services
  ./scripts/start-dev.sh --stop

EOF
}

# Log functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if a port is in use
is_port_in_use() {
    local port=$1
    lsof -i :${port} > /dev/null 2>&1
}

# Wait for a service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    log_info "Waiting for ${service_name} to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "${url}" > /dev/null 2>&1; then
            log_success "${service_name} is ready!"
            return 0
        fi
        sleep 1
        ((attempt++))
    done

    log_error "${service_name} failed to start within ${max_attempts} seconds"
    return 1
}

# Stop all services
stop_services() {
    log_info "Stopping all marimo-flow services..."

    # Kill processes from PID file
    if [ -f "${PIDS_FILE}" ]; then
        while read pid; do
            if ps -p ${pid} > /dev/null 2>&1; then
                log_info "Stopping process ${pid}..."
                kill ${pid} 2>/dev/null || true
            fi
        done < "${PIDS_FILE}"
        rm -f "${PIDS_FILE}"
    fi

    # Kill any remaining processes on our ports
    for port in ${MLFLOW_PORT} ${MARIMO_PORT}; do
        if is_port_in_use ${port}; then
            log_info "Killing process on port ${port}..."
            lsof -ti :${port} | xargs kill -9 2>/dev/null || true
        fi
    done

    log_success "All services stopped"
}

# Setup directories
setup_directories() {
    log_info "Setting up data directories..."

    mkdir -p "${DATA_DIR}/db"
    mkdir -p "${DATA_DIR}/artifacts"

    if [ ! -f "${DB_PATH}" ]; then
        log_info "Creating MLflow database..."
        touch "${DB_PATH}"
    fi

    log_success "Directories ready"
}

# Start MLflow server
start_mlflow() {
    log_info "Starting MLflow server on port ${MLFLOW_PORT}..."

    if is_port_in_use ${MLFLOW_PORT}; then
        log_warning "MLflow already running on port ${MLFLOW_PORT}"
        return 0
    fi

    uv run mlflow server \
        --host 0.0.0.0 \
        --port ${MLFLOW_PORT} \
        --backend-store-uri "sqlite:///${DB_PATH}" \
        --default-artifact-root "${ARTIFACTS_PATH}" \
        --serve-artifacts \
        > "${PROJECT_ROOT}/mlflow.log" 2>&1 &

    local mlflow_pid=$!
    echo ${mlflow_pid} >> "${PIDS_FILE}"

    wait_for_service "http://localhost:${MLFLOW_PORT}/health" "MLflow"

    log_success "MLflow server started (PID: ${mlflow_pid})"
    log_info "  UI: http://localhost:${MLFLOW_PORT}"
    log_info "  Logs: ${PROJECT_ROOT}/mlflow.log"
}

# Start Marimo server
start_marimo() {
    log_info "Starting Marimo server with MCP on port ${MARIMO_PORT}..."

    if is_port_in_use ${MARIMO_PORT}; then
        log_warning "Marimo already running on port ${MARIMO_PORT}"
        return 0
    fi

    # Set environment variables
    export MLFLOW_TRACKING_URI="http://localhost:${MLFLOW_PORT}"
    export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PROJECT_ROOT}/snippets"

    uv run marimo edit examples/ \
        --mcp \
        --port ${MARIMO_PORT} \
        --host 0.0.0.0 \
        > "${PROJECT_ROOT}/marimo.log" 2>&1 &

    local marimo_pid=$!
    echo ${marimo_pid} >> "${PIDS_FILE}"

    wait_for_service "http://localhost:${MARIMO_PORT}" "Marimo"

    log_success "Marimo server started (PID: ${marimo_pid})"
    log_info "  UI: http://localhost:${MARIMO_PORT}"
    log_info "  MCP: http://localhost:${MARIMO_PORT}/mcp/server"
    log_info "  Logs: ${PROJECT_ROOT}/marimo.log"
}

# Show MCP connection instructions
show_mcp_instructions() {
    cat << EOF

${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}
${GREEN}║               MCP Servers Ready for Connection                 ║${NC}
${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}

${BLUE}1. Marimo MCP Server${NC}
   Running at: http://localhost:${MARIMO_PORT}/mcp/server

   ${YELLOW}Add to Claude Desktop:${NC}
   Open: ~/.config/Claude/claude_desktop_config.json
   Add:
   {
     "mcpServers": {
       "marimo": {
         "transport": "http",
         "url": "http://localhost:${MARIMO_PORT}/mcp/server"
       }
     }
   }

${BLUE}2. Context7 MCP Server${NC}
   Pre-configured in .marimo.toml (preset: "context7")
   No additional setup needed!

${BLUE}3. MLflow MCP Server${NC}
   ${YELLOW}Start separately:${NC}
   export MLFLOW_TRACKING_URI=http://localhost:${MLFLOW_PORT}
   uv run mlflow mcp run

   ${YELLOW}Add to Claude Desktop:${NC}
   {
     "mcpServers": {
       "mlflow": {
         "transport": "stdio",
         "command": "mlflow",
         "args": ["mcp", "run"],
         "env": {
           "MLFLOW_TRACKING_URI": "http://localhost:${MLFLOW_PORT}"
         }
       }
     }
   }

${GREEN}Available MCP Tools:${NC}
  Marimo:   get_active_notebooks, get_notebook_errors, get_cell_runtime_data
  Context7: search_docs, get_library_docs
  MLflow:   search_experiments, search_runs, log_metric, list_models

${GREEN}Quick Test:${NC}
  1. Open Claude Desktop
  2. Ask: "List active marimo notebooks"
  3. Ask: "Search MLflow experiments"

EOF
}

# Main function
main() {
    local mlflow_only=false
    local marimo_only=false
    local no_browser=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -m|--mlflow-only)
                mlflow_only=true
                shift
                ;;
            -r|--marimo-only)
                marimo_only=true
                shift
                ;;
            -s|--stop)
                stop_services
                exit 0
                ;;
            --no-browser)
                no_browser=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Banner
    cat << EOF
${BLUE}
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║               🌊 marimo-flow Development                     ║
║                                                               ║
║   Reactive ML Notebooks + MLflow Tracking + MCP Support      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
${NC}
EOF

    # Check if uv is installed
    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    # Initialize PID file
    > "${PIDS_FILE}"

    # Setup
    setup_directories

    # Start services
    if [ "$mlflow_only" = true ]; then
        start_mlflow
    elif [ "$marimo_only" = true ]; then
        start_marimo
    else
        start_mlflow
        start_marimo
    fi

    # Show MCP instructions
    if [ "$marimo_only" = false ]; then
        show_mcp_instructions
    fi

    # Open browsers if not disabled
    if [ "$no_browser" = false ] && command -v xdg-open &> /dev/null; then
        if [ "$marimo_only" = false ]; then
            log_info "Opening MLflow UI..."
            xdg-open "http://localhost:${MLFLOW_PORT}" 2>/dev/null || true
        fi
        if [ "$mlflow_only" = false ]; then
            log_info "Opening Marimo UI..."
            xdg-open "http://localhost:${MARIMO_PORT}" 2>/dev/null || true
        fi
    fi

    # Final message
    log_success "Development environment ready!"
    echo ""
    log_info "Press Ctrl+C to stop all services, or run: ./scripts/start-dev.sh --stop"
    echo ""

    # Wait for Ctrl+C
    trap stop_services EXIT INT TERM

    # Keep script running
    while true; do
        sleep 1
    done
}

# Run main
main "$@"
