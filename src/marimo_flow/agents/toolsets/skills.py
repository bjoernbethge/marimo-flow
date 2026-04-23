"""FunctionToolset for discovering and reading project skills.

Used by NotebookNode alongside the marimo MCP server. No deps required —
skill discovery is a pure filesystem operation.
"""

from __future__ import annotations

from pydantic_ai import FunctionToolset

from marimo_flow.agents.skills import list_skills, read_skill_reference

skills_toolset: FunctionToolset[None] = FunctionToolset(id="skills")


@skills_toolset.tool_plain
def discover_skills() -> list[str]:
    """List all installed skills (project + user)."""
    return list_skills()


@skills_toolset.tool_plain
def fetch_skill_reference(name: str, ref_path: str) -> str:
    """Read an additional file from a skill (e.g. references/api.md)."""
    return read_skill_reference(name, ref_path)
