# Start marimo with MCP server enabled
# Uses the hidden --mcp flag to expose /mcp/server endpoint
# Requires: uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0"

$port = 2718
$endpoint = "http://127.0.0.1:$port/mcp/server"

# Check if marimo is already running on the port
$existing = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue

if (-not $existing) {
    Write-Host "Starting marimo MCP server on port $port..." -ForegroundColor Cyan
    Write-Host "MCP endpoint: $endpoint" -ForegroundColor Yellow

    # Start marimo in background (installed via: uv tool install "marimo[lsp,recommended,sql,mcp]>=0.18.0")
    Start-Process -FilePath 'marimo' -ArgumentList 'edit', '--mcp', '--no-token', '--port', $port, '--headless' -NoNewWindow

    # Wait for server to start
    Start-Sleep -Seconds 3

    # Verify it started
    $running = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "Marimo MCP server started successfully!" -ForegroundColor Green
        Write-Host "Add to Claude Code: claude mcp add --transport http marimo $endpoint" -ForegroundColor Yellow
    } else {
        Write-Host "Failed to start marimo MCP server" -ForegroundColor Red
    }
} else {
    Write-Host "Marimo MCP server already running on port $port" -ForegroundColor Green
}
