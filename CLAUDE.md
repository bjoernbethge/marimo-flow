# CLAUDE.md

## Project: marimo-flow

Reactive Python notebooks with MLflow tracking and PINA physics-informed neural networks.

## Commands

- `uv sync` - Install dependencies
- `marimo edit examples/<name>.py` - Open notebook
- `mlflow ui` - Start MLflow dashboard
- `ruff format . && ruff check --fix .` - Format/lint

## MCP Servers

- **marimo**: start manually with `marimo edit --mcp --no-token --port 2718 --headless`
- **mlflow**: `mlflow mcp run` (set `MLFLOW_TRACKING_URI`)
- VS Code config: `.vscode/mcp.json` (uses `servers` key, not `mcpServers`)
- No `.mcp.json` for Claude Code — MCP is wired in some other way here.

## context7 Library IDs

- `/marimo-team/marimo` - marimo docs
- `/mlflow/mlflow` - mlflow docs
- `/mathlab/pina` - PINA docs

## Cross-Platform Hooks

Pattern: `bash -c '...' 2>/dev/null || powershell ...`

## Code Style

- Use `uv` workflow (uv add, uv sync, uv tool install)
- No god classes - split into smaller modules
- No unnecessary wrappers - use core library features directly
- Skills in `.claude/Skills/` are git-tracked

## Agents (`src/marimo_flow/agents/`)

Multi-agent PINA team on `pydantic-graph` with 9 nodes:
`TriageNode` (start) → `RouteNode` (dispatcher) → `ProblemNode`, `ModelNode`,
`SolverNode`, `TrainingNode`, `ValidationNode`, `MLflowNode`, `NotebookNode`.

Layout (SPEC-driven):
- `schemas/` — 15 Pydantic models: `TaskSpec`, `ProblemSpec`, `DomainSpec`,
  `BoundaryConditionSpec`, `InitialConditionSpec`, `MaterialSpec`, `ModelSpec`,
  `SolverPlan`, `RunConfig`, `DatasetBinding`, `ArtifactRef`, `AgentDecision`,
  `HandoffRecord`, `ValidationReport`, `ExperimentRecord`. Kinds are `Literal`s
  tied 1:1 to `core.ProblemManager/ModelManager/SolverManager.available()`.
- `toolsets/` — `problem`, `model`, `solver`, `training`, `validation`, `data`,
  `skills`, `lead`. Each is a `FunctionToolset[FlowDeps]` singleton.
- `services/` — `ProvenanceStore` (DuckDB, 13 tables), orchestrator policy
  helpers (`check_escalation`, `default_experiment_status`,
  `requires_human_review`), experiment lifecycle (`start_experiment`,
  `complete_experiment`).
- `nodes/` — one module per graph node. `triage` builds a `TaskSpec` from
  free-form intent (fast-paths if one is already set). `validation` grades a
  training run and writes a `ValidationReport`. `route` emits `HandoffRecord`
  on every dispatch and short-circuits to `End` on escalate/reject verdicts.

Infrastructure:
- Orchestration: `pydantic-graph` (Graph + BaseNode + GraphRunContext)
- Persistence + tracing: MLflow (`mlflow.pytorch.autolog()`; pydantic-ai
  autolog is opt-in via `MLFLOW_PYDANTIC_AI_AUTOLOG=1` until mlflow >= 3.11.2)
- Provenance: DuckDB at `./provenance.duckdb` (configurable via
  `MARIMO_FLOW_PROVENANCE_DB` or `config.yaml`'s `provenance.db_path`).
  DuckDB 1.5.2 ships transitively via `marimo[sql]` — no extra project dep.
- LLMs: pydantic-ai `infer_model` with provider-prefixed specs; role→spec in
  `agents.deps.DEFAULT_MODELS` (10 roles: route, triage, notebook, problem,
  model, solver, training, validation, mlflow, lead).
- Each sub-agent loads its skill from `.claude/Skills/<name>/SKILL.md` via
  `build_skill_instructions()` — lazy, no message-history bloat.
- Lead agent (`build_lead_agent`) exposed three ways: `mo.ui.chat`,
  `agent.to_a2a()`, `agent.to_ag_ui()`. `run_pina_workflow` wraps every
  graph run in an `ExperimentRecord` (running → completed / failed).
- `FlowState` holds MLflow URIs **and** the typed specs; live
  PINA/torch objects live in `FlowDeps.registry` keyed by URI.
  `FlowState.to_jsonable()` renders Pydantic fields via
  `model_dump(mode="json")` so snapshots round-trip through JSON.

Testing:
- `tests/agents/test_*.py` — 167 passing, 1 xfailed.
- Nodes that use MCP toolsets (Notebook, MLflow): stub `build_*_mcp` with
  `pydantic_ai.toolsets.FunctionToolset()` to avoid live server connections,
  and pin `TestModel(call_tools=[])`.
- Toolset unit tests: call `toolset.tools["name"].function(ctx, ...)` directly
  with a plain `ctx` wrapper and `FlowDeps(provenance_db_path=":memory:")`
  — no graph, no LLM.
- Spec-setting in the build-* toolsets is wrapped in `contextlib.suppress`
  so TestModel's dummy kinds (e.g. `"a"`) don't break the MLflow-only path.
- Use `provenance_db_path=":memory:"` in tests to avoid leaving DB files.
