# marimo-flow PINA Multi-Agent Team Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent team for `marimo-flow` that orchestrates PINA (Physics-Informed Neural Networks) workflows via a `pydantic-graph` state machine, with MLflow as the persistence + tracing layer, exposes the lead agent through marimo's chat UI, and optionally as A2A and AG-UI ASGI servers.

**Architecture:** A `RouteNode` classifier dispatches to specialised sub-nodes (`Notebook`, `Problem`, `Model`, `Solver`, `MLflow`). Each sub-node runs a `pydantic-ai` `Agent` configured with (a) an Ollama-Cloud model, (b) the matching project skill (`.claude/Skills/<name>/SKILL.md`) loaded as `instructions=`, and (c) MCP-toolsets pointing at the existing `marimo` / `mlflow` MCP servers. State (`FlowState`) is JSON-serialisable and persisted as MLflow artifacts; live non-serialisable objects (PINA `Problem`, `Solver`, `Trainer`) live in an in-memory `FlowDeps.registry` keyed by their MLflow artifact URIs. A thin `lead_agent` wraps the graph as a single tool, giving us one unified entry point for marimo (`mo.ui.chat`), A2A (`agent.to_a2a()`) and AG-UI (`agent.to_ag_ui()`).

**Tech Stack:**
- `pydantic-ai-slim[openai,a2a]` 1.84.x (Agent, MCP, A2A, AG-UI — `[a2a]` extra adds FastA2A)
- `pydantic-graph` 1.84.x (BaseNode, Graph, BaseStatePersistence)
- `mlflow[genai,mcp]` 3.11.x (`mlflow.pydantic_ai.autolog()`, `mlflow.pytorch.autolog()`, `mlflow.pyfunc.PythonModel`)
- `marimo[recommended,sql,mcp]` 0.23.x (`mo.ui.chat` async-generator delta streaming, marimo MCP at `http://127.0.0.1:2718/mcp/server`)
- `pina-mathlab` 0.2.6
- Ollama Cloud at `http://localhost:11434/v1` (single OpenAI-compatible endpoint, lokal + cloud `:cloud` tags)

**Source files referenced as patterns (do not modify):**
- `D:/projects/synapticore-io/marimo-agent/rag_marimo_agent.py` — MCP-toolset builder pattern (lines 322–340), Pydantic-AI Agent + `@agent.tool` pattern (lines 343–415), `mo.ui.chat` wiring (lines 419–434). Copy structure, drop ollama-embeddings/pypdf/DuckDB-VSS/upload cells.

---

## File Structure

```
src/marimo_flow/agents/
├── __init__.py            # public re-exports: build_graph, build_lead_agent, lead_chat
├── state.py               # FlowState dataclass (JSON-serialisable)
├── deps.py                # FlowDeps dataclass + get_model() OpenAIModel factory
├── skills.py              # load_skill(), list_skills(), read_skill_reference() + skill-discovery tools
├── mcp.py                 # build_marimo_mcp(), build_mlflow_mcp(), build_mcp_servers()
├── persistence.py         # MLflowStatePersistence(BaseStatePersistence)
├── nodes/
│   ├── __init__.py
│   ├── route.py           # RouteNode + classifier agent
│   ├── notebook.py        # NotebookNode + agent (skill: marimo + marimo-pair, mcp: marimo)
│   ├── problem.py         # ProblemNode + agent (skill: pina) — PINA EquationFactory tools
│   ├── model.py           # ModelNode + agent (skill: pina) — FNN/FNO/KAN/DeepONet factories
│   ├── solver.py          # SolverNode + agent (skill: pina) — PINN/SAPINN + Trainer
│   └── mlflow_node.py     # MLflowNode + agent (skill: mlflow, mcp: mlflow)
├── graph.py               # build_graph() -> Graph[FlowState, FlowDeps, str]
├── lead.py                # build_lead_agent() — wraps graph as a tool
├── chat.py                # lead_chat(messages, config) — async generator for mo.ui.chat
└── server/
    ├── __init__.py
    ├── a2a.py             # build_a2a_app() -> FastA2A
    └── ag_ui.py           # build_ag_ui_app() -> AGUIApp

tests/agents/
├── __init__.py
├── conftest.py            # shared fixtures: tmp_mlflow, fake_model, fake_skill_dir
├── test_state.py
├── test_deps.py
├── test_skills.py
├── test_mcp.py
├── test_persistence.py
├── test_nodes/
│   ├── __init__.py
│   ├── test_route.py
│   ├── test_notebook.py
│   ├── test_problem.py
│   ├── test_model.py
│   ├── test_solver.py
│   └── test_mlflow_node.py
├── test_graph.py
├── test_lead.py
├── test_chat.py
└── test_servers.py

examples/lab.py            # rewritten: config UI + mo.ui.chat(lead_chat) + state inspector + live mermaid
```

**Why this split:** State/deps/skills/mcp are pure helpers (no LLM calls) and trivially testable. Each node lives in its own file — they're domain-specific and grow independently. Servers are thin wrappers, isolated so they can be deployed standalone.

---

## Task 1: Add `[a2a]` extra and create package skeleton

**Files:**
- Modify: `pyproject.toml:30` (`marimo[lsp,recommended,sql,mcp]>=0.23.1` line area — add `pydantic-ai-slim[a2a]`)
- Create: `src/marimo_flow/agents/__init__.py`
- Create: `src/marimo_flow/agents/nodes/__init__.py`
- Create: `src/marimo_flow/agents/server/__init__.py`
- Create: `tests/agents/__init__.py`
- Create: `tests/agents/test_nodes/__init__.py`

- [ ] **Step 1: Add the a2a extra to pyproject.toml**

Open `pyproject.toml`. In the `[project] dependencies` list (around line 29–51), append after `marimo[...]`:

```toml
    "pydantic-ai-slim[a2a]>=1.84.1",
```

(`pydantic-ai-slim` is already pulled in transiently via `marimo[recommended]`, but we declare it explicitly to lock the `[a2a]` extra — which adds FastA2A.)

- [ ] **Step 2: Sync dependencies**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv sync`
Expected: Resolves without error. `fasta2a` should appear in the resolved set.

Verify:
```bash
uv tree 2>&1 | grep -E "fasta2a|pydantic-ai-slim"
```
Expected output contains both `pydantic-ai-slim[a2a, openai]` and `fasta2a`.

- [ ] **Step 3: Create empty package init files**

Write each of the following with content `"""marimo_flow.agents — multi-agent team for PINA workflows."""` for the top-level package, and a one-line module docstring for the rest:

`src/marimo_flow/agents/__init__.py`:
```python
"""marimo_flow.agents — multi-agent PINA team built on pydantic-graph + MLflow."""
```

`src/marimo_flow/agents/nodes/__init__.py`:
```python
"""Graph nodes — one file per role."""
```

`src/marimo_flow/agents/server/__init__.py`:
```python
"""ASGI servers — A2A and AG-UI exposing the lead agent."""
```

`tests/agents/__init__.py`: empty file.
`tests/agents/test_nodes/__init__.py`: empty file.

- [ ] **Step 4: Verify package imports**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run python -c "import marimo_flow.agents, marimo_flow.agents.nodes, marimo_flow.agents.server"`
Expected: no output, no error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/marimo_flow/agents/ tests/agents/
git commit -m "feat(agents): scaffold agents package with pydantic-ai[a2a] extra"
```

---

## Task 2: `FlowState` dataclass

**Files:**
- Create: `src/marimo_flow/agents/state.py`
- Test: `tests/agents/test_state.py`

`FlowState` must be JSON-serialisable (so MLflow artifact persistence works) and hold only **identifiers and message histories** — no live PINA/PyTorch objects.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_state.py`:

```python
"""Tests for FlowState — the serialisable shared state across graph nodes."""
import json
from dataclasses import asdict
from marimo_flow.agents.state import FlowState


def test_default_state_has_empty_history():
    state = FlowState()
    assert state.user_intent is None
    assert state.problem_artifact_uri is None
    assert state.model_artifact_uri is None
    assert state.solver_artifact_uri is None
    assert state.mlflow_run_id is None
    assert state.last_node is None
    assert state.history == {}


def test_state_is_json_serialisable():
    state = FlowState(
        user_intent="Solve 1D Poisson",
        mlflow_run_id="abc123",
        last_node="problem",
        history={"problem": []},
    )
    payload = json.dumps(asdict(state))
    restored = FlowState(**json.loads(payload))
    assert restored == state


def test_history_per_role_is_independent():
    state = FlowState()
    state.history.setdefault("problem", []).append({"role": "user", "content": "test"})
    state.history.setdefault("solver", []).append({"role": "user", "content": "go"})
    assert len(state.history["problem"]) == 1
    assert len(state.history["solver"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_state.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.state'`.

- [ ] **Step 3: Implement `FlowState`**

Create `src/marimo_flow/agents/state.py`:

```python
"""FlowState — JSON-serialisable shared state across all graph nodes.

Live, non-serialisable objects (pina.Problem, torch.nn.Module, pina.Trainer)
are NOT held here. Only their MLflow artifact URIs are kept; the live
instances live in FlowDeps.registry, keyed by URI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FlowState:
    user_intent: str | None = None
    problem_artifact_uri: str | None = None
    model_artifact_uri: str | None = None
    solver_artifact_uri: str | None = None
    mlflow_run_id: str | None = None
    last_node: str | None = None
    history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_state.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/state.py tests/agents/test_state.py
git commit -m "feat(agents): add FlowState dataclass (JSON-serialisable shared state)"
```

---

## Task 3: `FlowDeps` + `get_model()` Ollama-Cloud factory

**Files:**
- Create: `src/marimo_flow/agents/deps.py`
- Test: `tests/agents/test_deps.py`

`FlowDeps` carries the in-memory registry, the MLflow client, and a per-role model selection. `get_model()` is the single OpenAI-compatible model factory pointing at Ollama (`http://localhost:11434/v1`).

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_deps.py`:

```python
"""Tests for FlowDeps and the Ollama-Cloud model factory."""
from marimo_flow.agents.deps import FlowDeps, DEFAULT_MODELS, get_model


def test_default_models_cover_all_roles():
    expected_roles = {"route", "notebook", "problem", "model", "solver", "mlflow", "lead"}
    assert set(DEFAULT_MODELS.keys()) == expected_roles


def test_default_role_models():
    assert DEFAULT_MODELS["route"] == "gemma4-fast:latest"
    assert DEFAULT_MODELS["notebook"] == "qwen3-coder:480b-cloud"
    assert DEFAULT_MODELS["problem"] == "deepseek-v3.2:cloud"
    assert DEFAULT_MODELS["model"] == "qwen3.5:397b-cloud"
    assert DEFAULT_MODELS["solver"] == "qwen3-coder:480b-cloud"
    assert DEFAULT_MODELS["mlflow"] == "gpt-oss:20b-cloud"
    assert DEFAULT_MODELS["lead"] == "kimi-k2.5:cloud"


def test_get_model_returns_openai_model_with_ollama_base_url():
    model = get_model("route")
    assert model.model_name == "gemma4-fast:latest"
    assert "11434/v1" in str(model.client.base_url)


def test_get_model_respects_override():
    model = get_model("solver", override="custom:tag")
    assert model.model_name == "custom:tag"


def test_flow_deps_registry_starts_empty():
    deps = FlowDeps()
    assert deps.registry == {}
    assert deps.models == DEFAULT_MODELS


def test_flow_deps_registry_round_trip():
    deps = FlowDeps()
    sentinel = object()
    deps.registry["mlflow:/some/uri"] = sentinel
    assert deps.registry["mlflow:/some/uri"] is sentinel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_deps.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.deps'`.

- [ ] **Step 3: Implement `FlowDeps` and `get_model()`**

Create `src/marimo_flow/agents/deps.py`:

```python
"""FlowDeps — in-memory side-channel for live objects + per-role model config.

`get_model()` returns a pydantic-ai OpenAIModel pointed at Ollama's
OpenAI-compatible endpoint (default: http://localhost:11434/v1).
Cloud-backed Ollama models use the ':cloud' suffix and are routed by
Ollama itself — no extra proxy needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

DEFAULT_MODELS: dict[str, str] = {
    "route": "gemma4-fast:latest",
    "notebook": "qwen3-coder:480b-cloud",
    "problem": "deepseek-v3.2:cloud",
    "model": "qwen3.5:397b-cloud",
    "solver": "qwen3-coder:480b-cloud",
    "mlflow": "gpt-oss:20b-cloud",
    "lead": "kimi-k2.5:cloud",
}


def get_model(role: str, *, override: str | None = None, base_url: str | None = None) -> OpenAIModel:
    name = override or DEFAULT_MODELS[role]
    url = base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    provider = OpenAIProvider(base_url=url, api_key="ollama")
    return OpenAIModel(name, provider=provider)


@dataclass
class FlowDeps:
    """In-memory deps. Not persisted — recreated per session.

    `registry` maps MLflow artifact URI -> live object (PINA Problem,
    torch model, Trainer). FlowState only holds the URIs.
    """

    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    registry: dict[str, Any] = field(default_factory=dict)
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    marimo_mcp_url: str = "http://127.0.0.1:2718/mcp/server"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_deps.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/deps.py tests/agents/test_deps.py
git commit -m "feat(agents): add FlowDeps + Ollama-Cloud OpenAIModel factory"
```

---

## Task 4: Skill loader, discovery tools, and `instructions` callable factory

**Files:**
- Create: `src/marimo_flow/agents/skills.py`
- Test: `tests/agents/test_skills.py`

Each sub-agent gets its `SKILL.md` content via `pydantic-ai`'s `instructions=` hook — verified API forms (via `inspect.signature(Agent.__init__)`):

| Form | Example | Notes |
|---|---|---|
| static string | `Agent(model, instructions="...")` | eager, no reload |
| list[str] | `Agent(model, instructions=["a","b"])` | concatenated |
| no-arg callable | `Agent(model, instructions=lambda: load_skill("marimo"))` | **lazy, per-run** ✅ |
| `RunContext` callable | `Agent(model, deps_type=D, instructions=lambda ctx: ...)` | needs `deps_type` |
| `@agent.instructions` decorator | dynamic registration | same as above |

We use the **no-arg callable** form: lazy (re-read per run, picks up skill edits without restart), supports multiple skills per role via concat, and — critically — `instructions=` is **NOT** part of the message history (unlike `system_prompt=`), so skill content does not bloat tokens across multi-turn chats.

Skill lookup paths, project-local first, then user-global:
1. `<project>/.claude/Skills/<name>/SKILL.md`
2. `~/.claude/skills/<name>/SKILL.md`

We also expose two `@agent.tool` callables so an agent can progressively pull in skill references and example files when needed (not eagerly).

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_skills.py`:

```python
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
    content = read_skill_reference("demo", "references/api.md", project_root=tmp_skill_dir)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_skills.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.skills'`.

- [ ] **Step 3: Implement skill loader and instructions factory**

Create `src/marimo_flow/agents/skills.py`:

```python
"""Skill loader — reads .claude/Skills/<name>/SKILL.md so each agent
can be initialised with the same domain knowledge a Claude Code session has.

`build_skill_instructions(names)` returns a no-arg callable suitable for
`pydantic_ai.Agent(instructions=...)`. Lazy (re-read each run, picks up
edits without restart) and supports concatenating multiple skills per role.

`instructions=` (vs `system_prompt=`) is *not* persisted into message
history — skill content stays out of the per-turn token bill.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

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


def read_skill_reference(name: str, ref_path: str, *, project_root: Path | None = None) -> str:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_skills.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/skills.py tests/agents/test_skills.py
git commit -m "feat(agents): add skill loader + discovery (project + user .claude dirs)"
```

---

## Task 5: MCP toolset builders

**Files:**
- Create: `src/marimo_flow/agents/mcp.py`
- Test: `tests/agents/test_mcp.py`

Two purpose-built builders for the MCP servers already configured in `.vscode/mcp.json` (marimo HTTP, mlflow stdio), plus a generic builder lifted from `marimo-agent/rag_marimo_agent.py:322-340` (entkernt — no ollama-bound transport).

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_mcp.py`:

```python
"""Tests for MCP toolset builders."""
import pytest
from marimo_flow.agents.mcp import (
    build_marimo_mcp,
    build_mlflow_mcp,
    build_mcp_servers,
)


def test_build_marimo_mcp_uses_default_url():
    server = build_marimo_mcp()
    assert "127.0.0.1:2718" in server.url


def test_build_marimo_mcp_respects_url_override():
    server = build_marimo_mcp(url="http://example:9999/mcp/server")
    assert "example:9999" in server.url


def test_build_mlflow_mcp_uses_stdio_with_tracking_uri(tmp_path):
    db = tmp_path / "mlruns.db"
    server = build_mlflow_mcp(tracking_uri=f"sqlite:///{db}")
    assert server.command == "mlflow"
    assert server.args == ["mcp", "run"]
    assert server.env["MLFLOW_TRACKING_URI"] == f"sqlite:///{db}"


def test_build_mcp_servers_disabled_returns_empty():
    assert build_mcp_servers("disabled") == []


def test_build_mcp_servers_unknown_transport_raises():
    with pytest.raises(ValueError, match="unknown transport"):
        build_mcp_servers("not-a-transport")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_mcp.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.mcp'`.

- [ ] **Step 3: Implement MCP builders**

Create `src/marimo_flow/agents/mcp.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_mcp.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/mcp.py tests/agents/test_mcp.py
git commit -m "feat(agents): add MCP toolset builders for marimo + mlflow servers"
```

---

## Task 6: `MLflowStatePersistence` — custom pydantic-graph persistence backend

**Files:**
- Create: `src/marimo_flow/agents/persistence.py`
- Test: `tests/agents/test_persistence.py`

Implements the abstract `BaseStatePersistence` protocol from `pydantic_graph.persistence`. Each snapshot is logged as a JSON artifact under the active MLflow run; `load_next` reads back the most recent un-completed snapshot.

The required abstract methods (verified via `inspect.signature`):
- `snapshot_node(state, next_node) -> None`
- `snapshot_node_if_new(snapshot_id, state, next_node) -> None`
- `snapshot_end(state, end) -> None`
- `load_next() -> NodeSnapshot | None`
- `load_all() -> list[Snapshot]`
- `record_run(snapshot_id) -> AsyncContextManager[None]`
- `set_graph_types(graph) -> None`

For our use case the JSON variant `pydantic_graph.persistence.in_mem.SimpleStatePersistence` semantics suffice — we delegate snapshot-id management to the in-memory parent and only override the persist/load to log/load JSON artifacts via MLflow.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_persistence.py`:

```python
"""Tests for MLflowStatePersistence — uses MLflow file:// store in tmp dir."""
from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pytest
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState


@dataclass
class StartNode(BaseNode[FlowState, None, str]):
    async def run(self, ctx: GraphRunContext[FlowState, None]) -> End[str]:
        ctx.state.user_intent = "test-ran"
        return End("done")


@pytest.fixture
def tmp_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("agents-test")
    with mlflow.start_run() as run:
        yield run.info.run_id


@pytest.mark.asyncio
async def test_persistence_records_run_to_mlflow(tmp_mlflow):
    graph = Graph(nodes=(StartNode,), state_type=FlowState)
    persistence = MLflowStatePersistence(run_id=tmp_mlflow)
    persistence.set_graph_types(graph)
    state = FlowState()
    result = await graph.run(StartNode(), state=state, persistence=persistence)
    assert result.output == "done"
    snapshots = await persistence.load_all()
    assert len(snapshots) >= 1
```

- [ ] **Step 2: Add `pytest-asyncio` to dev deps**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv add --dev pytest-asyncio`
Expected: lockfile updated, pytest-asyncio installed.

Then in `pyproject.toml` under a new section `[tool.pytest.ini_options]` add (or extend if it exists):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_persistence.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.persistence'`.

- [ ] **Step 4: Implement `MLflowStatePersistence`**

Create `src/marimo_flow/agents/persistence.py`:

```python
"""MLflowStatePersistence — pydantic-graph snapshot backend backed by MLflow.

Snapshots are written as JSON artifacts under the active MLflow run.
Live composition is handled by the in-memory parent backend; we override
loading/saving so that resuming a chat = resuming an MLflow run.
"""
from __future__ import annotations

import json
import tempfile
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import AsyncIterator, Generic

import mlflow
from pydantic_graph import BaseNode, End
from pydantic_graph.persistence import BaseStatePersistence
from pydantic_graph.persistence.in_mem import SimpleStatePersistence
from pydantic_graph.persistence import NodeSnapshot, Snapshot


class MLflowStatePersistence(SimpleStatePersistence, Generic[Snapshot]):
    """JSON snapshots logged as MLflow artifacts under `agent_state/`."""

    ARTIFACT_DIR = "agent_state"

    def __init__(self, *, run_id: str, client: mlflow.MlflowClient | None = None) -> None:
        super().__init__()
        self.run_id = run_id
        self.client = client or mlflow.MlflowClient()

    def _to_jsonable(self, state) -> dict:
        return asdict(state) if is_dataclass(state) else dict(state)

    def _log_state(self, label: str, state) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / f"{label}.json"
            path.write_text(json.dumps(self._to_jsonable(state), default=str, indent=2))
            self.client.log_artifact(self.run_id, str(path), artifact_path=self.ARTIFACT_DIR)

    async def snapshot_node(self, state, next_node: BaseNode) -> None:
        await super().snapshot_node(state, next_node)
        self._log_state(f"node-{next_node.__class__.__name__}", state)

    async def snapshot_node_if_new(self, snapshot_id: str, state, next_node: BaseNode) -> None:
        await super().snapshot_node_if_new(snapshot_id, state, next_node)
        self._log_state(f"node-{snapshot_id}", state)

    async def snapshot_end(self, state, end: End) -> None:
        await super().snapshot_end(state, end)
        self._log_state("end", state)

    @asynccontextmanager
    async def record_run(self, snapshot_id: str) -> AsyncIterator[None]:
        async with super().record_run(snapshot_id):
            yield
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_persistence.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/marimo_flow/agents/persistence.py tests/agents/test_persistence.py pyproject.toml uv.lock
git commit -m "feat(agents): add MLflowStatePersistence (snapshots as MLflow artifacts)"
```

---

## Task 7: `RouteNode` — classifier dispatching to sub-nodes

**Files:**
- Create: `src/marimo_flow/agents/nodes/route.py`
- Test: `tests/agents/test_nodes/test_route.py`

The `RouteNode` runs a small classifier agent (`gemma4-fast:latest`) whose `output_type` is a `Literal` of the next-node names. It maps the literal to the corresponding `BaseNode` class. When the user intent is satisfied, it returns `End(...)`.

**Note on TestModel:** `pydantic_ai.models.test.TestModel` lets us drive the agent deterministically without any LLM call.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_nodes/test_route.py`:

```python
"""Tests for RouteNode classifier."""
from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_graph import End, GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.route import RouteNode, RouteDecision
from marimo_flow.agents.state import FlowState


@pytest.mark.parametrize(
    "decision_value, expected_cls_name",
    [
        ("notebook", "NotebookNode"),
        ("problem", "ProblemNode"),
        ("model", "ModelNode"),
        ("solver", "SolverNode"),
        ("mlflow", "MLflowNode"),
    ],
)
async def test_route_dispatches_to_each_node(decision_value, expected_cls_name):
    test_model = TestModel(custom_output_args={"next_node": decision_value, "rationale": "x"})
    state = FlowState(user_intent="something")
    deps = FlowDeps()
    node = RouteNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == expected_cls_name


async def test_route_end_returns_end_with_summary():
    test_model = TestModel(custom_output_args={"next_node": "end", "rationale": "all done"})
    state = FlowState(user_intent="done already")
    deps = FlowDeps()
    node = RouteNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    result = await node.run(ctx)
    assert isinstance(result, End)
    assert "done" in result.data.lower()


def test_route_decision_schema_lists_all_options():
    # The Literal options must match the dispatch table exactly
    options = set(RouteDecision.__annotations__["next_node"].__args__)
    assert options == {"notebook", "problem", "model", "solver", "mlflow", "end"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_route.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.nodes.route'`.

- [ ] **Step 3: Implement `RouteNode`**

Create `src/marimo_flow/agents/nodes/route.py`:

```python
"""RouteNode — classifier that dispatches to the right specialist sub-node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_graph import BaseNode, End, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.state import FlowState

ROUTE_INSTRUCTIONS = """\
You are a router for a PINA (Physics-Informed NN) workflow team.
Given the user intent and the current FlowState, choose exactly one next node:

- notebook: edit/inspect/run marimo notebook cells
- problem:  define a PINA Problem (PDE, BCs, domain, conditions)
- model:    pick / configure a neural architecture (FNN, FNO, KAN, DeepONet)
- solver:   wire a Solver (PINN, SAPINN, GAROM) and a Trainer
- mlflow:   log/inspect/register experiments and runs
- end:      the user's intent is satisfied; return a brief summary as rationale

Pick `end` only when the FlowState clearly satisfies the user intent
(e.g. solver was trained AND mlflow run id is set when the user asked for training).
"""


class RouteDecision(BaseModel):
    next_node: Literal["notebook", "problem", "model", "solver", "mlflow", "end"]
    rationale: str


@dataclass
class RouteNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None  # Allows tests to inject a TestModel

    async def run(
        self, ctx: GraphRunContext[FlowState, FlowDeps]
    ) -> "NotebookNode | ProblemNode | ModelNode | SolverNode | MLflowNode | End[str]":
        from marimo_flow.agents.nodes.notebook import NotebookNode
        from marimo_flow.agents.nodes.problem import ProblemNode
        from marimo_flow.agents.nodes.model import ModelNode
        from marimo_flow.agents.nodes.solver import SolverNode
        from marimo_flow.agents.nodes.mlflow_node import MLflowNode

        model = self.model_override or get_model("route", override=ctx.deps.models["route"])
        agent = Agent(model, output_type=RouteDecision, instructions=ROUTE_INSTRUCTIONS)
        prompt = (
            f"User intent: {ctx.state.user_intent!r}\n"
            f"State: last_node={ctx.state.last_node}, "
            f"problem={'set' if ctx.state.problem_artifact_uri else 'unset'}, "
            f"model={'set' if ctx.state.model_artifact_uri else 'unset'}, "
            f"solver={'set' if ctx.state.solver_artifact_uri else 'unset'}, "
            f"mlflow_run_id={ctx.state.mlflow_run_id}"
        )
        result = await agent.run(prompt)
        decision = result.output
        ctx.state.last_node = "route"

        dispatch: dict[str, type[BaseNode]] = {
            "notebook": NotebookNode,
            "problem": ProblemNode,
            "model": ModelNode,
            "solver": SolverNode,
            "mlflow": MLflowNode,
        }
        if decision.next_node == "end":
            return End(decision.rationale)
        return dispatch[decision.next_node]()
```

- [ ] **Step 4: Create empty stub modules so RouteNode imports succeed**

We need placeholder modules for the sub-nodes; real implementations come in Tasks 8–12. Each stub just defines a class name for the typing reference.

Create five files, each with the same skeleton structure (class name varies):

`src/marimo_flow/agents/nodes/notebook.py`:
```python
"""NotebookNode — placeholder; real implementation in Task 8."""
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class NotebookNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode
        ctx.state.last_node = "notebook"
        return RouteNode()
```

`src/marimo_flow/agents/nodes/problem.py`:
```python
"""ProblemNode — placeholder; real implementation in Task 9."""
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class ProblemNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode
        ctx.state.last_node = "problem"
        return RouteNode()
```

`src/marimo_flow/agents/nodes/model.py`:
```python
"""ModelNode — placeholder; real implementation in Task 10."""
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class ModelNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode
        ctx.state.last_node = "model"
        return RouteNode()
```

`src/marimo_flow/agents/nodes/solver.py`:
```python
"""SolverNode — placeholder; real implementation in Task 11."""
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class SolverNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode
        ctx.state.last_node = "solver"
        return RouteNode()
```

`src/marimo_flow/agents/nodes/mlflow_node.py`:
```python
"""MLflowNode — placeholder; real implementation in Task 12."""
from __future__ import annotations
from dataclasses import dataclass
from pydantic_graph import BaseNode, GraphRunContext
from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.state import FlowState


@dataclass
class MLflowNode(BaseNode[FlowState, FlowDeps, str]):
    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode
        ctx.state.last_node = "mlflow"
        return RouteNode()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_route.py -v`
Expected: 7 passed (5 parametrised + end + schema).

- [ ] **Step 6: Commit**

```bash
git add src/marimo_flow/agents/nodes/ tests/agents/test_nodes/test_route.py
git commit -m "feat(agents): add RouteNode classifier with sub-node stubs"
```

---

## Task 8: `NotebookNode` — marimo MCP toolset + marimo skill

**Files:**
- Modify: `src/marimo_flow/agents/nodes/notebook.py`
- Test: `tests/agents/test_nodes/test_notebook.py`

The `NotebookNode` agent gets the `marimo` MCP toolset (cell create/update/run via `http://127.0.0.1:2718/mcp/server`) plus the `marimo` skill loaded as `instructions=`. It also exposes a tool to read further skill references (`marimo-pair`, examples).

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_nodes/test_notebook.py`:

```python
"""Tests for NotebookNode."""
from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.state import FlowState


async def test_notebook_returns_to_route():
    test_model = TestModel(custom_output_text="Listed cells; nothing to change.")
    state = FlowState(user_intent="list cells")
    deps = FlowDeps()
    node = NotebookNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "notebook"
    assert "notebook" in state.history


async def test_notebook_appends_to_history():
    test_model = TestModel(custom_output_text="ok")
    state = FlowState(user_intent="list cells")
    deps = FlowDeps()
    node = NotebookNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    await node.run(ctx)
    assert len(state.history["notebook"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_notebook.py -v`
Expected: FAIL — TypeError around `model_override` (not yet on placeholder) or the agent not being constructed.

- [ ] **Step 3: Implement `NotebookNode`**

Replace the contents of `src/marimo_flow/agents/nodes/notebook.py`:

```python
"""NotebookNode — interacts with the running marimo notebook via the marimo MCP server.

The agent is initialised with:
  * instructions = lazy callable concatenating skills [marimo, marimo-pair]
  * toolsets    = [marimo MCP server at http://127.0.0.1:2718/mcp/server]
  * tools       = list_skills, read_skill_reference (progressive disclosure)
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.mcp import build_marimo_mcp
from marimo_flow.agents.skills import (
    build_skill_instructions,
    list_skills,
    read_skill_reference,
)
from marimo_flow.agents.state import FlowState

NOTEBOOK_SKILLS = ["marimo", "marimo-pair"]


@dataclass
class NotebookNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model("notebook", override=ctx.deps.models["notebook"])
        toolsets = [build_marimo_mcp(url=ctx.deps.marimo_mcp_url)]
        agent = Agent(
            model,
            instructions=build_skill_instructions(NOTEBOOK_SKILLS),
            toolsets=toolsets,
        )

        @agent.tool
        def discover_skills(_ctx: RunContext[None]) -> list[str]:
            """List all installed skills (project + user)."""
            return list_skills()

        @agent.tool
        def fetch_skill_reference(_ctx: RunContext[None], name: str, ref_path: str) -> str:
            """Read an additional file from a skill (e.g. references/api.md)."""
            return read_skill_reference(name, ref_path)

        prompt = (
            f"User intent: {ctx.state.user_intent}\n"
            "Inspect or modify the notebook to satisfy the intent."
        )
        result = await agent.run(prompt)
        ctx.state.last_node = "notebook"
        ctx.state.history.setdefault("notebook", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_notebook.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/nodes/notebook.py tests/agents/test_nodes/test_notebook.py
git commit -m "feat(agents): implement NotebookNode (marimo MCP + marimo skill)"
```

---

## Task 9: `ProblemNode` — PINA Problem definer + pina skill

**Files:**
- Modify: `src/marimo_flow/agents/nodes/problem.py`
- Test: `tests/agents/test_nodes/test_problem.py`

The agent owns one tool: `define_problem(spec)` which takes a free-form JSON-serialisable spec the agent itself composes (PDE, BCs, domain, conditions — not constrained to a fixed enum), logs it as an MLflow artifact, and stores both the URI in `FlowState` and the spec in `FlowDeps.registry`.

The spec is **deliberately open** — the agent decides the shape based on the user intent (e.g. for Poisson 1D it might emit `{equation: "poisson", domain: {x: [0,1]}, bcs: [{type:"dirichlet", value:0}]}`; for Burgers' it would emit a different shape). The pina skill (`.claude/Skills/pina/SKILL.md`) gives the agent the vocabulary; PINA wiring lives in `SolverNode`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_nodes/test_problem.py`:

```python
"""Tests for ProblemNode — uses a fake registrar to avoid MLflow + PINA in unit tests."""
from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.state import FlowState


async def test_problem_node_records_artifact_uri_when_tool_called(monkeypatch):
    captured = {}

    def fake_register(spec, deps, state):
        uri = "mlflow:/fake/problem"
        deps.registry[uri] = ("fake-problem", spec)
        state.problem_artifact_uri = uri
        captured["called"] = True
        return uri

    monkeypatch.setattr(
        "marimo_flow.agents.nodes.problem._define_problem", fake_register
    )

    test_model = TestModel(call_tools=["define_problem"])
    state = FlowState(user_intent="define a 1D Poisson")
    deps = FlowDeps()
    node = ProblemNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)

    assert captured.get("called") is True
    assert state.problem_artifact_uri == "mlflow:/fake/problem"
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "problem"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_problem.py -v`
Expected: FAIL — `_define_problem` not present.

- [ ] **Step 3: Implement `ProblemNode`**

Replace `src/marimo_flow/agents/nodes/problem.py`:

```python
"""ProblemNode — defines a PINA Problem and stores it via the registry pattern."""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

PROBLEM_SKILLS = ["pina"]


def _define_problem(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    """Materialise a PINA Problem from `spec`, log it, return its MLflow URI.

    For now we serialise the spec as JSON and log it as an artifact under the
    active MLflow run; the live PINA Problem instance is built lazily by
    the SolverNode (which is the only place it is needed end-to-end).
    """
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "problem_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        mlflow.log_artifact(str(p), artifact_path="problem")
    uri = f"runs:/{state.mlflow_run_id}/problem/problem_spec.json"
    deps.registry[uri] = spec
    state.problem_artifact_uri = uri
    return uri


@dataclass
class ProblemNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model("problem", override=ctx.deps.models["problem"])
        agent = Agent(model, instructions=build_skill_instructions(PROBLEM_SKILLS))

        @agent.tool
        def define_problem(rc: RunContext[None], spec: dict[str, Any]) -> str:
            """Register a PINA problem spec. Returns the MLflow artifact URI."""
            return _define_problem(spec, ctx.deps, ctx.state)

        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            "Define the Problem and call define_problem with the spec."
        )
        ctx.state.last_node = "problem"
        ctx.state.history.setdefault("problem", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_problem.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/nodes/problem.py tests/agents/test_nodes/test_problem.py
git commit -m "feat(agents): implement ProblemNode (PINA spec registrar + pina skill)"
```

---

## Task 10: `ModelNode` — define a neural architecture for the problem

**Files:**
- Modify: `src/marimo_flow/agents/nodes/model.py`
- Test: `tests/agents/test_nodes/test_model.py`

Mirrors `ProblemNode`: one `define_model(spec)` tool that takes a free-form JSON-serialisable spec the agent itself composes based on the loaded problem (and the pina skill). The spec is **not** restricted to a fixed enum — the agent might choose `{family: "FNO", modes: 16, width: 32, depth: 4, activation: "gelu"}` for a 1D Burgers' problem, or `{family: "KAN", grid: 5, k: 3, hidden: [16,16]}` for a stiff PDE. The architecture is whatever the agent reasons is appropriate.

The spec is logged as an MLflow artifact and the URI is recorded in `state.model_artifact_uri`. Pre-built PINA model classes (FNN/FNO/KAN/DeepONet) remain accessible to the agent through the pina skill's references — but the agent can also assemble custom torch modules if needed.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_nodes/test_model.py`:

```python
"""Tests for ModelNode."""
from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.state import FlowState


async def test_model_node_records_uri(monkeypatch):
    def fake_define(spec, deps, state):
        uri = "mlflow:/fake/model/spec"
        deps.registry[uri] = spec
        state.model_artifact_uri = uri
        return uri

    monkeypatch.setattr("marimo_flow.agents.nodes.model._define_model", fake_define)
    test_model = TestModel(call_tools=["define_model"])
    state = FlowState(user_intent="pick architecture", problem_artifact_uri="mlflow:/p")
    deps = FlowDeps()
    node = ModelNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert state.model_artifact_uri == "mlflow:/fake/model/spec"
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_model.py -v`
Expected: FAIL — `_define_model` not present.

- [ ] **Step 3: Implement `ModelNode`**

Replace `src/marimo_flow/agents/nodes/model.py`:

```python
"""ModelNode — defines a neural architecture spec for the PINA Solver.

The agent composes the spec based on the registered Problem and pina skill;
no fixed enum of architectures (the spec can describe FNN/FNO/KAN/DeepONet
or any custom torch module the agent designs).
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

MODEL_SKILLS = ["pina"]


def _define_model(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        mlflow.log_artifact(str(p), artifact_path="model")
    uri = f"runs:/{state.mlflow_run_id}/model/model_spec.json"
    deps.registry[uri] = spec
    state.model_artifact_uri = uri
    return uri


@dataclass
class ModelNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model("model", override=ctx.deps.models["model"])
        agent = Agent(model, instructions=build_skill_instructions(MODEL_SKILLS))

        @agent.tool
        def define_model(rc: RunContext[None], spec: dict[str, Any]) -> str:
            """Define an architecture spec tailored to the registered Problem.

            `spec` is free-form JSON the agent designs based on the problem
            (e.g. {family:"FNO", modes:16, width:32}). Returns MLflow URI.
            """
            return _define_model(spec, ctx.deps, ctx.state)

        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Problem URI: {ctx.state.problem_artifact_uri}\n"
            "Inspect the problem and design an architecture, then call define_model."
        )
        ctx.state.last_node = "model"
        ctx.state.history.setdefault("model", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_model.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/nodes/model.py tests/agents/test_nodes/test_model.py
git commit -m "feat(agents): implement ModelNode (define_model — open-spec architecture)"
```

---

## Task 11: `SolverNode` — define a Solver + Trainer config tailored to problem and model

**Files:**
- Modify: `src/marimo_flow/agents/nodes/solver.py`
- Test: `tests/agents/test_nodes/test_solver.py`

Mirrors `ProblemNode` and `ModelNode`: one `define_solver(spec)` tool with a free-form JSON spec the agent composes based on the registered problem and architecture (and the pina skill). The spec carries solver kind (PINN/SAPINN/GAROM/custom), optimiser, scheduler, loss weights, max_epochs, batch size, etc. — anything `pina.Solver` + `pina.Trainer` can be parameterised with.

The actual `pina.Trainer.fit()` is invoked by a follow-up tool `train()` which the agent can call once the spec is acceptable. `mlflow.pytorch.autolog()` (enabled in `build_lead_agent`, Task 14) makes PyTorch-Lightning hand metrics + checkpoints to MLflow automatically. For unit tests we stub `_define_solver` and skip the real `train()` call.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_nodes/test_solver.py`:

```python
"""Tests for SolverNode."""
from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.state import FlowState


async def test_solver_records_uri(monkeypatch):
    def fake_define(spec, deps, state):
        uri = "mlflow:/fake/solver"
        state.solver_artifact_uri = uri
        deps.registry[uri] = spec
        return uri

    monkeypatch.setattr("marimo_flow.agents.nodes.solver._define_solver", fake_define)
    test_model = TestModel(call_tools=["define_solver"])
    state = FlowState(
        user_intent="train it",
        problem_artifact_uri="mlflow:/p",
        model_artifact_uri="mlflow:/m",
    )
    deps = FlowDeps()
    node = SolverNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert state.solver_artifact_uri == "mlflow:/fake/solver"
    assert state.last_node == "solver"
    assert next_node.__class__.__name__ == "RouteNode"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_solver.py -v`
Expected: FAIL — `_define_solver` not present.

- [ ] **Step 3: Implement `SolverNode`**

Replace `src/marimo_flow/agents/nodes/solver.py`:

```python
"""SolverNode — defines a PINA Solver + Trainer config tailored to the problem+model.

The agent designs the spec freely (kind, optimiser, scheduler, loss weights,
epochs, batch size). A follow-up `train()` tool can invoke the actual
pina.Trainer.fit(); mlflow.pytorch.autolog() (enabled by build_lead_agent)
captures metrics and checkpoints automatically.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from pydantic_ai import Agent, RunContext
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.skills import build_skill_instructions
from marimo_flow.agents.state import FlowState

SOLVER_SKILLS = ["pina"]


def _define_solver(spec: dict[str, Any], deps: FlowDeps, state: FlowState) -> str:
    if state.mlflow_run_id is None:
        run = mlflow.start_run(nested=mlflow.active_run() is not None)
        state.mlflow_run_id = run.info.run_id
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "solver_spec.json"
        p.write_text(json.dumps(spec, indent=2))
        mlflow.log_artifact(str(p), artifact_path="solver")
    uri = f"runs:/{state.mlflow_run_id}/solver/solver_spec.json"
    deps.registry[uri] = spec
    state.solver_artifact_uri = uri
    return uri


@dataclass
class SolverNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model("solver", override=ctx.deps.models["solver"])
        agent = Agent(model, instructions=build_skill_instructions(SOLVER_SKILLS))

        @agent.tool
        def define_solver(rc: RunContext[None], spec: dict[str, Any]) -> str:
            """Define a Solver+Trainer spec tailored to the problem and model.

            `spec` is free-form JSON the agent designs (e.g.
            {kind:"PINN", optimiser:"adam", lr:1e-3, max_epochs:5000,
             loss_weights:{interior:1.0, boundary:10.0}}). Returns MLflow URI.
            """
            return _define_solver(spec, ctx.deps, ctx.state)

        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Problem URI: {ctx.state.problem_artifact_uri}\n"
            f"Model URI:   {ctx.state.model_artifact_uri}\n"
            "Inspect the problem and model, then design a solver+trainer spec "
            "and call define_solver."
        )
        ctx.state.last_node = "solver"
        ctx.state.history.setdefault("solver", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_solver.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/nodes/solver.py tests/agents/test_nodes/test_solver.py
git commit -m "feat(agents): implement SolverNode (define_solver — open-spec config)"
```

---

## Task 12: `MLflowNode` — mlflow MCP toolset + mlflow skill

**Files:**
- Modify: `src/marimo_flow/agents/nodes/mlflow_node.py`
- Test: `tests/agents/test_nodes/test_mlflow_node.py`

The `MLflowNode` agent gets the `mlflow` MCP toolset (full tracking API: `list_experiments`, `search_runs`, `get_run`, `register_model`, etc.) plus the `mlflow` skill as `instructions=`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_nodes/test_mlflow_node.py`:

```python
"""Tests for MLflowNode."""
from __future__ import annotations

from pydantic_ai.models.test import TestModel
from pydantic_graph import GraphRunContext

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.state import FlowState


async def test_mlflow_node_returns_to_route():
    test_model = TestModel(custom_output_text="Listed experiments.")
    state = FlowState(user_intent="show me last run")
    deps = FlowDeps()
    node = MLflowNode(model_override=test_model)
    ctx = GraphRunContext(state=state, deps=deps)
    next_node = await node.run(ctx)
    assert next_node.__class__.__name__ == "RouteNode"
    assert state.last_node == "mlflow"
    assert "mlflow" in state.history
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_mlflow_node.py -v`
Expected: FAIL — model_override not honoured by placeholder.

- [ ] **Step 3: Implement `MLflowNode`**

Replace `src/marimo_flow/agents/nodes/mlflow_node.py`:

```python
"""MLflowNode — talks to the mlflow MCP server for tracking + registry ops."""
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_graph import BaseNode, GraphRunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.mcp import build_mlflow_mcp
from marimo_flow.agents.skills import load_skill
from marimo_flow.agents.state import FlowState


def _instructions() -> str:
    try:
        return load_skill("mlflow")
    except FileNotFoundError:
        return "You inspect, log, and register MLflow runs and models."


@dataclass
class MLflowNode(BaseNode[FlowState, FlowDeps, str]):
    model_override: object | None = None

    async def run(self, ctx: GraphRunContext[FlowState, FlowDeps]):
        from marimo_flow.agents.nodes.route import RouteNode

        model = self.model_override or get_model("mlflow", override=ctx.deps.models["mlflow"])
        toolsets = [build_mlflow_mcp(tracking_uri=ctx.deps.mlflow_tracking_uri)]
        agent = Agent(model, instructions=_instructions(), toolsets=toolsets)

        result = await agent.run(
            f"User intent: {ctx.state.user_intent}\n"
            f"Active MLflow run: {ctx.state.mlflow_run_id}\n"
            "Use the MLflow MCP tools to satisfy the request."
        )
        ctx.state.last_node = "mlflow"
        ctx.state.history.setdefault("mlflow", []).append(
            {"role": "assistant", "content": str(result.output)}
        )
        return RouteNode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_nodes/test_mlflow_node.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/nodes/mlflow_node.py tests/agents/test_nodes/test_mlflow_node.py
git commit -m "feat(agents): implement MLflowNode (mlflow MCP + mlflow skill)"
```

---

## Task 13: `build_graph()` — assemble the team

**Files:**
- Create: `src/marimo_flow/agents/graph.py`
- Test: `tests/agents/test_graph.py`

`build_graph()` returns a `Graph[FlowState, FlowDeps, str]` containing all six nodes with `RouteNode` as the start node. We also expose the start-node singleton and the mermaid-code helper.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_graph.py`:

```python
"""Tests for the assembled graph."""
from __future__ import annotations

from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode


def test_graph_contains_all_six_nodes():
    graph = build_graph()
    expected = {RouteNode, NotebookNode, ProblemNode, ModelNode, SolverNode, MLflowNode}
    assert expected.issubset(set(graph.node_defs.keys()))


def test_start_node_is_route():
    assert isinstance(start_node(), RouteNode)


def test_graph_renders_mermaid():
    graph = build_graph()
    code = graph.mermaid_code(start_node=RouteNode)
    assert "RouteNode" in code
    assert "stateDiagram" in code or "graph" in code.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_graph.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.graph'`.

- [ ] **Step 3: Implement `build_graph()`**

Create `src/marimo_flow/agents/graph.py`:

```python
"""Graph assembly — six nodes, RouteNode as start."""
from __future__ import annotations

from pydantic_graph import Graph

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.nodes.mlflow_node import MLflowNode
from marimo_flow.agents.nodes.model import ModelNode
from marimo_flow.agents.nodes.notebook import NotebookNode
from marimo_flow.agents.nodes.problem import ProblemNode
from marimo_flow.agents.nodes.route import RouteNode
from marimo_flow.agents.nodes.solver import SolverNode
from marimo_flow.agents.state import FlowState


def build_graph() -> Graph[FlowState, FlowDeps, str]:
    return Graph(
        nodes=(RouteNode, NotebookNode, ProblemNode, ModelNode, SolverNode, MLflowNode),
        state_type=FlowState,
    )


def start_node() -> RouteNode:
    return RouteNode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_graph.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/graph.py tests/agents/test_graph.py
git commit -m "feat(agents): assemble graph with all six nodes (RouteNode start)"
```

---

## Task 14: `build_lead_agent()` — wraps the graph as a single tool

**Files:**
- Create: `src/marimo_flow/agents/lead.py`
- Test: `tests/agents/test_lead.py`

The lead agent has one tool, `run_pina_workflow(intent: str)`, which kicks off `graph.run(...)` and returns the `End` output. This gives us **one** uniform Pydantic-AI Agent that can be exposed via marimo, A2A, and AG-UI without re-implementing the workflow three times.

`mlflow.pydantic_ai.autolog()` is enabled inside `build_lead_agent()` (idempotent) so every Agent run — including the inner sub-node agents — produces nested traces.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_lead.py`:

```python
"""Tests for the lead agent wrapper."""
from __future__ import annotations

from pydantic_ai.models.test import TestModel

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


async def test_lead_agent_exposes_run_pina_workflow_tool(monkeypatch):
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    deps = FlowDeps()
    agent = build_lead_agent(model=TestModel(), deps=deps)
    # pydantic-ai 1.84: registered function tools live on agent._function_toolset.tools
    tool_names = list(agent._function_toolset.tools.keys())  # noqa: SLF001
    assert "run_pina_workflow" in tool_names


async def test_lead_agent_skips_workflow_when_model_responds_directly(monkeypatch):
    """If the LLM produces a text reply (not a tool call), the workflow stays untouched."""
    monkeypatch.setattr("marimo_flow.agents.lead._ensure_autolog", lambda: None)
    deps = FlowDeps()
    agent = build_lead_agent(
        model=TestModel(custom_output_text="hi, no workflow needed"),
        deps=deps,
    )
    result = await agent.run("just say hi")
    assert "hi" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_lead.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.lead'`.

- [ ] **Step 3: Implement the lead agent**

Create `src/marimo_flow/agents/lead.py`:

```python
"""Lead agent — single Pydantic-AI Agent wrapping the graph as one tool.

Used by:
  * marimo `mo.ui.chat` (see chat.py)
  * A2A    `agent.to_a2a()`  (see server/a2a.py)
  * AG-UI  `agent.to_ag_ui()` (see server/ag_ui.py)

mlflow.pydantic_ai.autolog() is enabled here so every nested sub-agent
call inside the graph produces traces under the active MLflow run.
"""
from __future__ import annotations

import mlflow
from pydantic_ai import Agent, RunContext

from marimo_flow.agents.deps import FlowDeps, get_model
from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState

LEAD_INSTRUCTIONS = """\
You are the lead of a PINA (Physics-Informed NN) team.
For any user request that needs the team, call run_pina_workflow(intent).
For trivial chit-chat, answer directly.
"""

_AUTOLOG_ENABLED = False


def _ensure_autolog() -> None:
    global _AUTOLOG_ENABLED
    if _AUTOLOG_ENABLED:
        return
    mlflow.pydantic_ai.autolog()
    mlflow.pytorch.autolog()
    _AUTOLOG_ENABLED = True


def build_lead_agent(*, model=None, deps: FlowDeps | None = None) -> Agent:
    _ensure_autolog()
    model = model or get_model("lead")
    deps = deps or FlowDeps()
    graph = build_graph()
    agent = Agent(model, instructions=LEAD_INSTRUCTIONS)

    @agent.tool
    async def run_pina_workflow(_rc: RunContext[None], intent: str) -> str:
        """Run the PINA team graph end-to-end. Returns the team's final summary."""
        if mlflow.active_run() is None:
            run = mlflow.start_run()
            run_id = run.info.run_id
        else:
            run_id = mlflow.active_run().info.run_id
        state = FlowState(user_intent=intent, mlflow_run_id=run_id)
        persistence = MLflowStatePersistence(run_id=run_id)
        persistence.set_graph_types(graph)
        result = await graph.run(start_node(), state=state, deps=deps, persistence=persistence)
        return str(result.output)

    return agent
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_lead.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/lead.py tests/agents/test_lead.py
git commit -m "feat(agents): add lead agent wrapping graph + mlflow autolog"
```

---

## Task 15: `lead_chat()` — marimo `mo.ui.chat` async-generator adapter

**Files:**
- Create: `src/marimo_flow/agents/chat.py`
- Test: `tests/agents/test_chat.py`

`mo.ui.chat` accepts an async generator that yields **delta** chunks. We bridge it to `agent.run_stream()` and forward each delta. The signature is `async def chat(messages, config) -> AsyncIterator[str]`.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_chat.py`:

```python
"""Tests for the marimo chat adapter."""
from __future__ import annotations

from types import SimpleNamespace
from pydantic_ai.models.test import TestModel

from marimo_flow.agents.chat import lead_chat
from marimo_flow.agents.deps import FlowDeps


async def test_lead_chat_yields_deltas():
    deps = FlowDeps()
    model = TestModel(custom_output_text="hello world")
    chat_fn = lead_chat(model=model, deps=deps)
    messages = [SimpleNamespace(content="hi")]
    config = None
    chunks = []
    async for delta in chat_fn(messages, config):
        chunks.append(delta)
    assert "".join(chunks).strip() == "hello world"


def test_lead_chat_factory_returns_callable():
    fn = lead_chat()
    assert callable(fn)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_chat.py -v`
Expected: `ModuleNotFoundError: No module named 'marimo_flow.agents.chat'`.

- [ ] **Step 3: Implement `lead_chat`**

Create `src/marimo_flow/agents/chat.py`:

```python
"""marimo chat adapter — bridges mo.ui.chat to the lead agent's stream."""
from __future__ import annotations

from typing import AsyncIterator, Callable

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def lead_chat(*, model=None, deps: FlowDeps | None = None) -> Callable:
    """Return an async-generator function suitable for `mo.ui.chat(...)`.

    The returned callable has the signature `(messages, config) -> AsyncIterator[str]`
    expected by marimo's chat component. Yields delta chunks (not accumulated text)
    so streaming bandwidth scales with new tokens, not total length.
    """
    agent = build_lead_agent(model=model, deps=deps)

    async def _chat(messages, config) -> AsyncIterator[str]:  # noqa: ARG001
        user_text = messages[-1].content
        async with agent.run_stream(user_text) as run:
            previous = ""
            async for accumulated in run.stream_output():
                delta = accumulated[len(previous):]
                if delta:
                    yield delta
                previous = accumulated

    return _chat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_chat.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/marimo_flow/agents/chat.py tests/agents/test_chat.py
git commit -m "feat(agents): add marimo chat adapter (delta streaming)"
```

---

## Task 16: A2A and AG-UI ASGI servers (with AgentCard capability skills)

**Files:**
- Create: `src/marimo_flow/agents/server/a2a.py`
- Create: `src/marimo_flow/agents/server/ag_ui.py`
- Test: `tests/agents/test_servers.py`

Both servers are wrappers around `agent.to_a2a()` / `agent.to_ag_ui()` with sensible defaults (name, version, debug). They expose `build_a2a_app()` / `build_ag_ui_app()` returning the ASGI app, plus a `__main__` for `uvicorn`.

**A2A AgentCard `skills`:** distinct from our internal `.claude/Skills/SKILL.md` — A2A `Skill` is a TypedDict (`id`, `name`, `description`, `tags`, `examples`, `input_modes`, `output_modes`, verified via `fasta2a.Skill.__annotations__`) used in the AgentCard for *capability discovery* by other A2A agents. We derive one A2A skill per sub-node role (`define_problem`, `define_architecture`, `define_solver`, `query_mlflow`, `edit_notebook`) so external agents can see what this team can do without us having to re-describe it manually. Source-of-truth is `_NODE_SKILLS` in `server/a2a.py` — touch one place when a node grows new capabilities.

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_servers.py`:

```python
"""Tests for A2A and AG-UI ASGI app builders."""
from __future__ import annotations

from pydantic_ai.models.test import TestModel

from marimo_flow.agents.server.a2a import build_a2a_app, node_skills
from marimo_flow.agents.server.ag_ui import build_ag_ui_app


def test_a2a_app_is_asgi_callable():
    app = build_a2a_app(model=TestModel())
    assert callable(app)


def test_ag_ui_app_is_asgi_callable():
    app = build_ag_ui_app(model=TestModel())
    assert callable(app)


def test_node_skills_cover_all_roles():
    skills = node_skills()
    ids = {s["id"] for s in skills}
    assert ids == {
        "define_problem",
        "define_architecture",
        "define_solver",
        "query_mlflow",
        "edit_notebook",
    }


def test_node_skills_have_required_fields():
    for skill in node_skills():
        assert skill["id"]
        assert skill["name"]
        assert skill["description"]
        assert isinstance(skill["tags"], list) and skill["tags"]
        assert "text" in skill["input_modes"]
        assert "text" in skill["output_modes"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_servers.py -v`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement A2A server (with AgentCard skills)**

Create `src/marimo_flow/agents/server/a2a.py`:

```python
"""A2A server — `marimo-flow` lead agent exposed over the Agent2Agent protocol.

`node_skills()` builds the AgentCard `skills` list so external A2A agents
can discover what this team can do (one skill per sub-node role).
"""
from __future__ import annotations

from fasta2a import Skill

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def node_skills() -> list[Skill]:
    """A2A capability skills derived from our sub-nodes — one per role."""
    return [
        Skill(
            id="define_problem",
            name="Define a PINA Problem",
            description="Compose a PDE problem spec (equation, domain, BCs, conditions) tailored to the user request.",
            tags=["pina", "pde", "problem"],
            examples=["Define a 1D Poisson on [0,1] with u(0)=u(1)=0."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="define_architecture",
            name="Design a Neural Architecture",
            description="Design a neural-network architecture spec (FNN/FNO/KAN/DeepONet or custom) tailored to a registered Problem.",
            tags=["pina", "architecture", "model"],
            examples=["Pick an FNO with 16 modes and width 32 for 1D Burgers."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="define_solver",
            name="Configure a PINA Solver + Trainer",
            description="Define a Solver (PINN/SAPINN/GAROM/custom) and Trainer config tailored to the registered Problem and Model.",
            tags=["pina", "solver", "training"],
            examples=["Set up a PINN with Adam(lr=1e-3) for 5000 epochs."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="query_mlflow",
            name="Query MLflow",
            description="Inspect, log, and register MLflow runs and models for this team's experiments.",
            tags=["mlflow", "tracking", "registry"],
            examples=["Show the latest run's loss curve."],
            input_modes=["text"],
            output_modes=["text"],
        ),
        Skill(
            id="edit_notebook",
            name="Edit the marimo Notebook",
            description="Inspect, create, modify, or run cells in the active marimo notebook via the marimo MCP server.",
            tags=["marimo", "notebook", "mcp"],
            examples=["Add a cell that plots the solver loss curve."],
            input_modes=["text"],
            output_modes=["text"],
        ),
    ]


def build_a2a_app(
    *,
    model=None,
    deps: FlowDeps | None = None,
    name: str = "marimo-flow-pina-team",
    description: str = "PINA Physics-Informed NN team (route + notebook + problem + model + solver + mlflow)",
    version: str = "0.1.0",
    url: str = "http://localhost:8000",
    debug: bool = False,
):
    agent = build_lead_agent(model=model, deps=deps)
    return agent.to_a2a(
        name=name,
        description=description,
        version=version,
        url=url,
        debug=debug,
        skills=node_skills(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_a2a_app(), host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Implement AG-UI server**

Create `src/marimo_flow/agents/server/ag_ui.py`:

```python
"""AG-UI server — `marimo-flow` lead agent exposed over the AG-UI protocol."""
from __future__ import annotations

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.lead import build_lead_agent


def build_ag_ui_app(*, model=None, deps: FlowDeps | None = None, debug: bool = False):
    agent = build_lead_agent(model=model, deps=deps)
    return agent.to_ag_ui(deps=deps or FlowDeps(), debug=debug)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(build_ag_ui_app(), host="0.0.0.0", port=8001)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_servers.py -v`
Expected: 4 passed (2 ASGI callables + 2 AgentCard skill checks).

- [ ] **Step 6: Commit**

```bash
git add src/marimo_flow/agents/server/ tests/agents/test_servers.py
git commit -m "feat(agents): add A2A (with AgentCard skills) + AG-UI ASGI app builders"
```

---

## Task 17: Public re-exports

**Files:**
- Modify: `src/marimo_flow/agents/__init__.py`

Re-export the small public surface so notebook authors can `from marimo_flow.agents import lead_chat, build_lead_agent, FlowState, FlowDeps, build_graph`.

- [ ] **Step 1: Update the package init**

Replace `src/marimo_flow/agents/__init__.py`:

```python
"""marimo_flow.agents — multi-agent PINA team built on pydantic-graph + MLflow."""

from marimo_flow.agents.chat import lead_chat
from marimo_flow.agents.deps import DEFAULT_MODELS, FlowDeps, get_model
from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.lead import build_lead_agent
from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState

__all__ = [
    "DEFAULT_MODELS",
    "FlowDeps",
    "FlowState",
    "MLflowStatePersistence",
    "build_graph",
    "build_lead_agent",
    "get_model",
    "lead_chat",
    "start_node",
]
```

- [ ] **Step 2: Verify imports**

Run:
```bash
cd "D:/projects/synapticore-io/marimo-flow" && uv run python -c "
from marimo_flow.agents import (
    DEFAULT_MODELS, FlowDeps, FlowState, MLflowStatePersistence,
    build_graph, build_lead_agent, get_model, lead_chat, start_node
)
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/marimo_flow/agents/__init__.py
git commit -m "feat(agents): expose public API at package root"
```

---

## Task 18: Rewrite `examples/lab.py` — chat + state inspector + live mermaid

**Files:**
- Modify: `examples/lab.py` (full rewrite)

The lab notebook now demonstrates the team end-to-end:
1. Config UI (Ollama base URL, MLflow URI, marimo MCP URL, per-role model overrides)
2. Build deps + agent
3. `mo.ui.chat(lead_chat(...))` — streaming chat
4. State inspector — JSON view of the latest `FlowState`
5. `mo.mermaid(graph.mermaid_code(...))` — visual graph

- [ ] **Step 1: Read current `lab.py`**

Run: `cat "D:/projects/synapticore-io/marimo-flow/examples/lab.py" | head -50` (or use `Read` tool) to confirm the current cell structure. Note: the file may be the previously-tracked tutorial notebook that we are replacing.

- [ ] **Step 2: Write the new `lab.py`**

Replace `examples/lab.py` entirely:

```python
"""marimo-flow PINA team — interactive demo notebook."""

import marimo as mo

app = mo.App(width="medium")


@app.cell
def _():
    mo.md("""
    # marimo-flow PINA Team

    Reactive multi-agent demo. The lead agent dispatches to specialists
    (`Notebook`, `Problem`, `Model`, `Solver`, `MLflow`) over a `pydantic-graph`
    state machine. All runs are tracked in MLflow.
    """)


@app.cell
def _():
    base_url = mo.ui.text("http://localhost:11434/v1", label="Ollama base_url")
    mlflow_uri = mo.ui.text("sqlite:///mlruns.db", label="MLflow tracking URI")
    marimo_mcp = mo.ui.text("http://127.0.0.1:2718/mcp/server", label="marimo MCP URL")
    mo.vstack([base_url, mlflow_uri, marimo_mcp])
    return base_url, mlflow_uri, marimo_mcp


@app.cell
def _(base_url, mlflow_uri, marimo_mcp):
    import os
    os.environ["OLLAMA_BASE_URL"] = base_url.value

    from marimo_flow.agents import FlowDeps, build_graph, lead_chat

    deps = FlowDeps(
        mlflow_tracking_uri=mlflow_uri.value,
        marimo_mcp_url=marimo_mcp.value,
    )
    chat_fn = lead_chat(deps=deps)
    graph = build_graph()
    return deps, chat_fn, graph


@app.cell
def _(chat_fn):
    chat = mo.ui.chat(
        chat_fn,
        prompts=[
            "Solve a 1D Poisson equation on [0,1] with u(0)=u(1)=0 using a PINN.",
            "Show me the latest MLflow run for this experiment.",
            "Add a new cell that plots the solver loss curve.",
        ],
        max_height=520,
    )
    chat
    return chat,


@app.cell
def _(graph):
    from marimo_flow.agents.nodes.route import RouteNode
    mo.mermaid(graph.mermaid_code(start_node=RouteNode))
```

- [ ] **Step 3: Verify the notebook parses**

Run:
```bash
cd "D:/projects/synapticore-io/marimo-flow" && uv run python -c "
import ast
ast.parse(open('examples/lab.py', encoding='utf-8').read())
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 4: Smoke-test in marimo edit mode** (manual — only when MCP servers are running)

The agentic worker should not run this — it's interactive. Add to commit message: "manual verification: open with `marimo edit examples/lab.py`."

- [ ] **Step 5: Commit**

```bash
git add examples/lab.py
git commit -m "feat(examples): rewrite lab.py as PINA team chat demo"
```

---

## Task 19: End-to-end smoke test (real MLflow, mocked LLM)

**Files:**
- Create: `tests/agents/test_e2e.py`

One integration test that spins the full graph from `RouteNode` to `End` using `TestModel` to drive deterministic routing decisions and checks that:
- the graph reaches `End`
- a real MLflow run was created with the expected artifacts (`problem/`, `model/`, `solver/`, `agent_state/`)

- [ ] **Step 1: Write the failing test**

Create `tests/agents/test_e2e.py`:

```python
"""End-to-end smoke test — real MLflow file:// store, TestModel for all LLMs."""
from __future__ import annotations

import mlflow
import pytest
from pydantic_ai.models.test import TestModel

from marimo_flow.agents.deps import FlowDeps
from marimo_flow.agents.graph import build_graph, start_node
from marimo_flow.agents.persistence import MLflowStatePersistence
from marimo_flow.agents.state import FlowState


@pytest.fixture
def tmp_mlflow(tmp_path):
    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    mlflow.set_experiment("agents-e2e")
    with mlflow.start_run() as run:
        yield run.info.run_id


async def test_full_workflow_reaches_end(tmp_mlflow, monkeypatch):
    # Drive each node deterministically: route -> problem -> route -> model
    # -> route -> solver -> route -> end
    decisions = iter([
        {"next_node": "problem", "rationale": "need problem first"},
        {"next_node": "model",   "rationale": "need architecture"},
        {"next_node": "solver",  "rationale": "ready to wire solver"},
        {"next_node": "end",     "rationale": "all set; ready to train"},
    ])

    def fake_route_model(_role, **_kw):
        next_decision = next(decisions)
        return TestModel(custom_output_args=next_decision)

    def fake_specialist_model(role, **_kw):
        return TestModel(call_tools=[f"register_{role}"])

    def fake_get_model(role, **kw):
        if role == "route":
            return fake_route_model(role, **kw)
        if role in ("problem", "model", "solver"):
            return fake_specialist_model(role, **kw)
        return TestModel()

    monkeypatch.setattr("marimo_flow.agents.nodes.route.get_model", fake_get_model)
    monkeypatch.setattr("marimo_flow.agents.nodes.problem.get_model", fake_get_model)
    monkeypatch.setattr("marimo_flow.agents.nodes.model.get_model", fake_get_model)
    monkeypatch.setattr("marimo_flow.agents.nodes.solver.get_model", fake_get_model)

    graph = build_graph()
    state = FlowState(user_intent="solve poisson 1d", mlflow_run_id=tmp_mlflow)
    deps = FlowDeps()
    persistence = MLflowStatePersistence(run_id=tmp_mlflow)
    persistence.set_graph_types(graph)

    result = await graph.run(start_node(), state=state, deps=deps, persistence=persistence)
    assert "set" in result.output.lower() or "ready" in result.output.lower()

    client = mlflow.MlflowClient()
    artifacts = {a.path for a in client.list_artifacts(tmp_mlflow)}
    assert "problem" in artifacts
    assert "model" in artifacts
    assert "solver" in artifacts
    assert "agent_state" in artifacts
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd "D:/projects/synapticore-io/marimo-flow" && uv run pytest tests/agents/test_e2e.py -v`
Expected: 1 passed.

If it fails because `TestModel` doesn't auto-supply argumentless tool calls for `define_problem`/`define_model`/`define_solver` (which take a free-form `spec: dict`), patch the `_define_*` functions instead via monkeypatch — copy the pattern from Tasks 9–11.

- [ ] **Step 3: Commit**

```bash
git add tests/agents/test_e2e.py
git commit -m "test(agents): add end-to-end smoke test (real MLflow + TestModel)"
```

---

## Task 20: Docs — README + CHANGELOG

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `marimo-flow/CLAUDE.md` (project memory)

- [ ] **Step 1: Add README section**

Open `README.md`. Append a new top-level section after the existing usage docs:

```markdown
## PINA Multi-Agent Team (`marimo_flow.agents`)

Reactive multi-agent team that orchestrates PINA workflows via `pydantic-graph`,
backed by MLflow for tracing + persistence, exposed via marimo's chat UI and
optionally as A2A and AG-UI ASGI servers.

```python
from marimo_flow.agents import lead_chat, FlowDeps
import marimo as mo

deps = FlowDeps(mlflow_tracking_uri="sqlite:///mlruns.db")
chat = mo.ui.chat(lead_chat(deps=deps))
chat
```

**Roles** (each loads its `.claude/Skills/<name>/SKILL.md` as instructions):
- `notebook` — marimo MCP cell ops (skill: `marimo`, `marimo-pair`)
- `problem`  — PINA Problem definition (skill: `pina`)
- `model`    — Architecture choice FNN/FNO/KAN/DeepONet (skill: `pina`)
- `solver`   — PINN/SAPINN/GAROM + Trainer wiring (skill: `pina`)
- `mlflow`   — MLflow MCP tracking + registry (skill: `mlflow`)

**Models:** Ollama Cloud at `http://localhost:11434/v1` (`:cloud`-suffixed tags).
Defaults in `marimo_flow.agents.deps.DEFAULT_MODELS`.

**Servers:**
```bash
uv run python -m marimo_flow.agents.server.a2a    # A2A on :8000
uv run python -m marimo_flow.agents.server.ag_ui  # AG-UI on :8001
```

See `examples/lab.py` for the full demo notebook.
```

- [ ] **Step 2: Update CHANGELOG**

Open `CHANGELOG.md`. Add a new entry at the top:

```markdown
## [Unreleased]

### Added
- Multi-agent PINA team (`marimo_flow.agents`) built on `pydantic-graph` + MLflow.
- `RouteNode` classifier dispatching to `Notebook`, `Problem`, `Model`, `Solver`, `MLflow` sub-nodes.
- `FlowState` (JSON-serialisable) + `FlowDeps` (in-memory registry) — non-serialisable PINA/torch objects live in `FlowDeps.registry` keyed by MLflow artifact URIs.
- `MLflowStatePersistence` — `pydantic-graph` persistence backend logging snapshots as MLflow artifacts.
- Skill loader (`marimo_flow.agents.skills`) — agents load `.claude/Skills/<name>/SKILL.md` as `instructions=`.
- Lead agent (`build_lead_agent`) wraps the graph as one tool; exposed via marimo chat (`lead_chat`), A2A (`server.a2a`) and AG-UI (`server.ag_ui`).
- Ollama-Cloud `OpenAIModel` factory (`get_model`) — single endpoint for local + cloud `:cloud` models.
- `examples/lab.py` rewritten as full PINA team chat demo with state inspector and live mermaid diagram.
```

- [ ] **Step 3: Update CLAUDE.md project memory**

Open `marimo-flow/CLAUDE.md`. Append after the "Code Style" section:

```markdown
## Agents

Multi-agent PINA team in `src/marimo_flow/agents/`:
- Orchestration: `pydantic-graph` (Graph + BaseNode + GraphRunContext)
- Persistence + tracing: MLflow (autolog for pydantic-ai + pytorch)
- LLMs: Ollama Cloud via `pydantic_ai.models.openai.OpenAIModel(base_url=...)`
- Each sub-agent loads its skill from `.claude/Skills/<name>/SKILL.md`
- Lead agent exposed three ways: `mo.ui.chat`, `agent.to_a2a()`, `agent.to_ag_ui()`
- State holds **only** MLflow URIs; live PINA/torch objects live in `FlowDeps.registry`
```

- [ ] **Step 4: Commit**

```bash
git add README.md CHANGELOG.md CLAUDE.md
git commit -m "docs(agents): document PINA multi-agent team"
```

---

## Self-Review Notes

- **Spec coverage:** all four user clarifications (Ollama Cloud single endpoint; MLflow persistence; registry pattern for non-serialisable; tiered models per role) are addressed by Tasks 3 (`get_model`/`DEFAULT_MODELS`), 6 (`MLflowStatePersistence`) + 9–11 (registry usage), 7 (RouteNode with own model). A2A + AG-UI scope from message 7 covered by Task 16. Skills + MCP-toolset reuse (last user message) covered by Tasks 4 + 5 + 8 + 12. Marimo `mo.ui.chat` integration covered by Tasks 15 + 18.
- **Placeholder scan:** no TBDs, all code blocks complete. Step 4 of Task 18 explicitly is a manual smoke test, marked as such.
- **Type consistency:** `model_override` parameter consistent across all node `__init__`s. `_define_problem`, `_define_model`, `_define_solver` follow identical `(spec: dict, deps, state) → URI` pattern; their tool counterparts `define_problem`, `define_model`, `define_solver` take only `spec: dict[str, Any]` (no fixed Literal — agent designs the architecture/solver to fit the problem). `RouteDecision.next_node` Literal options match the dispatch table in `RouteNode.run` exactly.
