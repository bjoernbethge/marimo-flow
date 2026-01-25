@echo off
REM Start marimo with MCP server enabled
REM Uses the hidden --mcp flag to expose /mcp/server endpoint
REM Requires: uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"

REM Check if marimo is already running on port 2718
netstat -ano | findstr ":2718" >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting marimo MCP server on port 2718...
    echo MCP endpoint will be at: http://127.0.0.1:2718/mcp/server
    start /B marimo edit --mcp --no-token --port 2718 --headless
    REM Wait for server to start
    timeout /t 3 /nobreak >nul
    echo Marimo MCP server started.
) else (
    echo Marimo MCP server already running on port 2718.
)
