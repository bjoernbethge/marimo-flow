"""Skill loader — reads .claude/Skills/<name>/SKILL.md so each agent
can be initialised with the same domain knowledge a Claude Code session has.

`build_skill_instructions(names)` returns a no-arg callable suitable for
`pydantic_ai.Agent(instructions=...)`. Lazy (re-read each run, picks up
edits without restart) and supports concatenating multiple skills per role.

`instructions=` (vs `system_prompt=`) is *not* persisted into message
history — skill content stays out of the per-turn token bill.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

PROJECT_SKILL_SUBPATH = Path(".claude") / "Skills"
USER_SKILL_SUBPATH = Path(".claude") / "skills"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
SKILL_SEPARATOR = "\n\n---\n\n"


def _candidate_skill_dirs(name: str, project_root: Path) -> list[Path]:
    return [
        project_root / PROJECT_SKILL_SUBPATH / name,
        Path.home() / USER_SKILL_SUBPATH / name,
    ]


def _resolve_skill_dir(name: str, project_root: Path) -> Path:
    for d in _candidate_skill_dirs(name, project_root):
        if (d / "SKILL.md").is_file():
            return d
    raise FileNotFoundError(
        f"Skill '{name}' not found. Looked in: "
        + ", ".join(str(d) for d in _candidate_skill_dirs(name, project_root))
    )


def load_skill(name: str, *, project_root: Path | None = None) -> str:
    root = project_root or Path.cwd()
    return (_resolve_skill_dir(name, root) / "SKILL.md").read_text(encoding="utf-8")


def list_skills(*, project_root: Path | None = None) -> list[str]:
    root = project_root or Path.cwd()
    names: set[str] = set()
    for base in (root / PROJECT_SKILL_SUBPATH, Path.home() / USER_SKILL_SUBPATH):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "SKILL.md").is_file():
                names.add(child.name)
    return sorted(names)


def read_skill_reference(
    name: str, ref_path: str, *, project_root: Path | None = None
) -> str:
    root = project_root or Path.cwd()
    skill_dir = _resolve_skill_dir(name, root).resolve()
    target = (skill_dir / ref_path).resolve()
    if skill_dir not in target.parents and target != skill_dir:
        raise ValueError(f"Path '{ref_path}' is outside skill '{name}' directory")
    if not target.is_file():
        raise FileNotFoundError(f"Reference not found: {target}")
    return target.read_text(encoding="utf-8")


def build_skill_instructions(
    names: list[str], *, project_root: Path | None = None
) -> Callable[[], str]:
    """Return a no-arg callable that loads + concatenates the named skills.

    Pass directly to `pydantic_ai.Agent(instructions=...)`. Missing skills
    are skipped silently; if all are missing, returns DEFAULT_INSTRUCTIONS.
    """

    def _loader() -> str:
        parts: list[str] = []
        for name in names:
            try:
                parts.append(load_skill(name, project_root=project_root))
            except FileNotFoundError:
                continue
        return SKILL_SEPARATOR.join(parts) if parts else DEFAULT_INSTRUCTIONS

    return _loader
