"""Tests for skill loader, discovery, and instructions-callable factory."""

from pathlib import Path

import pytest

from marimo_flow.agents.skills import (
    build_skill_instructions,
    list_skills,
    load_skill,
    read_skill_reference,
)


@pytest.fixture
def tmp_skill_dir(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    skills = project / ".claude" / "Skills"
    (skills / "demo").mkdir(parents=True)
    (skills / "demo" / "SKILL.md").write_text("# Demo skill\nUse this for demos.\n")
    (skills / "demo" / "references").mkdir()
    (skills / "demo" / "references" / "api.md").write_text("# API reference\n")
    (skills / "other").mkdir()
    (skills / "other" / "SKILL.md").write_text("# Other\nOther skill body.\n")
    return project


def test_load_skill_returns_skill_md_content(tmp_skill_dir: Path):
    text = load_skill("demo", project_root=tmp_skill_dir)
    assert "Demo skill" in text


def test_load_skill_raises_when_missing(tmp_skill_dir: Path):
    with pytest.raises(FileNotFoundError):
        load_skill("nonexistent", project_root=tmp_skill_dir)


def test_list_skills_returns_sorted_names(tmp_skill_dir: Path):
    names = list_skills(project_root=tmp_skill_dir)
    assert "demo" in names
    assert "other" in names
    assert names == sorted(names)


def test_read_skill_reference_returns_file_content(tmp_skill_dir: Path):
    content = read_skill_reference(
        "demo", "references/api.md", project_root=tmp_skill_dir
    )
    assert "API reference" in content


def test_read_skill_reference_blocks_path_traversal(tmp_skill_dir: Path):
    with pytest.raises(ValueError, match="outside skill"):
        read_skill_reference("demo", "../other/SKILL.md", project_root=tmp_skill_dir)


def test_build_skill_instructions_concats_multiple(tmp_skill_dir: Path):
    fn = build_skill_instructions(["demo", "other"], project_root=tmp_skill_dir)
    text = fn()
    assert "Demo skill" in text
    assert "Other skill body" in text


def test_build_skill_instructions_skips_missing_skills(tmp_skill_dir: Path):
    fn = build_skill_instructions(["demo", "nonexistent"], project_root=tmp_skill_dir)
    text = fn()
    assert "Demo skill" in text
    assert text != ""


def test_build_skill_instructions_returns_default_when_none_found(tmp_skill_dir: Path):
    fn = build_skill_instructions(["nonexistent"], project_root=tmp_skill_dir)
    assert fn() == "You are a helpful assistant."


def test_build_skill_instructions_callable_signature_is_no_arg(tmp_skill_dir: Path):
    """pydantic-ai Agent(instructions=callable) supports no-arg callables."""
    import inspect

    fn = build_skill_instructions(["demo"], project_root=tmp_skill_dir)
    assert len(inspect.signature(fn).parameters) == 0
