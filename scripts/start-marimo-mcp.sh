#!/bin/bash
# Start marimo with MCP server for Claude Code integration
# Usage: ./scripts/start-marimo-mcp.sh [--no-auth]

set -e

NO_AUTH=false
if [[ "$1" == "--no-auth" ]]; then
    NO_AUTH=true
fi

echo "🚀 Starting marimo with MCP server..."
echo ""

# Start marimo with MCP in background and capture output
MARIMO_OUTPUT=$(marimo edit --mcp 2>&1 &)

# Wait a bit for server to start
sleep 3

# Extract the MCP server URL from output
MCP_URL=$(echo "$MARIMO_OUTPUT" | grep "MCP server URL:" | sed 's/.*MCP server URL: //' | tr -d ' ')
TOKEN=$(echo "$MCP_URL" | grep -oP 'access_token=\K[^&]+' || echo "")

if [ -z "$MCP_URL" ]; then
    echo "❌ Failed to extract MCP server URL"
    echo "Output was:"
    echo "$MARIMO_OUTPUT"
    exit 1
fi

echo "✅ marimo is running!"
echo ""
echo "📝 Notebook URL: http://127.0.0.1:2718?access_token=$TOKEN"
echo ""
echo "🔗 For Claude Code integration:"
if [ "$NO_AUTH" = true ]; then
    echo "   claude mcp add --transport http marimo $MCP_URL"
else
    echo "   claude mcp add --transport http marimo $MCP_URL"
fi
echo ""
echo "💡 Tip: Add --no-auth flag to disable token requirement:"
echo "   ./scripts/start-marimo-mcp.sh --no-auth"
echo ""
echo "Press Ctrl+C to stop the server"

# Keep script running
wait
