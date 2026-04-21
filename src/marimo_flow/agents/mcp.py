"""MCP toolset builders for the agents.

Two purpose-built helpers point at the project's existing MCP servers
(see .vscode/mcp.json):
  * marimo MCP at http://127.0.0.1:2718/mcp/server (HTTP)
  * mlflow MCP via `mlflow mcp run` (stdio)

`build_mcp_servers()` is a generic transport-selector kept for ad-hoc
use — adapted from `marimo-agent/rag_marimo_agent.py:322-340`.
"""

from __future__ import annotations

from pydantic_ai.mcp import (
    MCPServerSSE,
    MCPServerStdio,
    MCPServerStreamableHTTP,
)

DEFAULT_MARIMO_MCP_URL = "http://127.0.0.1:2718/mcp/server"


def build_marimo_mcp(url: str = DEFAULT_MARIMO_MCP_URL) -> MCPServerStreamableHTTP:
    return MCPServerStreamableHTTP(url=url)


def build_mlflow_mcp(tracking_uri: str = "sqlite:///mlruns.db") -> MCPServerStdio:
    return MCPServerStdio(
        command="mlflow",
        args=["mcp", "run"],
        env={"MLFLOW_TRACKING_URI": tracking_uri},
    )


def build_mcp_servers(
    transport: str,
    *,
    cmd: str = "deno",
    args: str = "",
    url: str = "",
) -> list:
    if transport == "disabled":
        return []
    if transport == "stdio":
        arg_list = [a for a in args.split(" ") if a]
        return [MCPServerStdio(command=cmd, args=arg_list)]
    if transport == "sse":
        return [MCPServerSSE(url=url)]
    if transport == "streamable-http":
        return [MCPServerStreamableHTTP(url=url)]
    raise ValueError(f"unknown transport: {transport!r}")
