# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-22

### Added
- Multi-agent PINA team (`marimo_flow.agents`) built on `pydantic-graph` + MLflow.
- `RouteNode` classifier dispatching to `Notebook`, `Problem`, `Model`, `Solver`, `MLflow` sub-nodes.
- `FlowState` (JSON-serialisable) + `FlowDeps` (in-memory registry) — non-serialisable PINA/torch objects live in `FlowDeps.registry` keyed by MLflow artifact URIs.
- `MLflowStatePersistence` — `pydantic-graph` persistence backend logging snapshots as MLflow artifacts.
- Skill loader (`marimo_flow.agents.skills`) — agents load `.claude/Skills/<name>/SKILL.md` as lazy `instructions=` callables; supports concatenating multiple skills per role.
- `_define_problem` / `_define_model` / `_define_solver` — open-form spec tools (no fixed enum); the agent designs the spec to fit the problem.
- Lead agent (`build_lead_agent`) wraps the graph as one tool; exposed via marimo chat (`lead_chat`), A2A (`server.a2a`), and AG-UI (`server.ag_ui`).
- A2A AgentCard with one `Skill` per sub-node role for capability discovery by external agents.
- Ollama-Cloud `OpenAIChatModel` factory (`get_model`) — single endpoint for local + cloud `:cloud` models, no separate proxy.
- `examples/lab.py` rewritten as full PINA team chat demo with state inspector and live mermaid diagram.
- `CITATION.cff` for Zenodo DOI integration on GitHub Releases.

### Changed
- `pydantic-ai-slim` upgraded to `[a2a, ag-ui, openai]` extras for protocol support.
- `OpenAIModel` → `OpenAIChatModel` (deprecation in pydantic-ai 1.84).
- `pyproject.toml` description + keywords updated to reflect PINA, agents, MCP, and Ollama integration.
- README: dropped marketing claims (`Production Ready` subsection, etc.), removed obsolete `docs/RESEARCH_SUMMARY.md`.

### Fixed
- `_define_*` helpers now use `MlflowClient` with explicit `state.mlflow_run_id` to avoid silent artifact misroute when no module-level active run exists (e.g. inside `await graph.run(...)`).

## [0.2.0] - 2026-03-26

### Added
- Multi-platform Docker images (CPU, CUDA, XPU) published to GHCR
- PINA integration: ProblemManager, ModelFactory, SolverManager, WalrusAdapter, Optuna visualization helpers
- MCP integration: marimo, mlflow, context7 servers pre-configured
- Claude Code skills for marimo, mlflow, and pina
- Tag-based PyPI publish workflow with auto GitHub Release
- Security: pinned minimum versions for all Dependabot-flagged transitive deps (authlib, pillow, cryptography)
- Dependabot auto-merge workflow

### Changed
- Examples reduced to 3 focused notebooks (MLflow Console, PINA Walrus Solver, PINA Live Monitoring)
- Dependencies simplified to 3 core (marimo, mlflow, pina-mathlab) + optional `[all]` extra for torch/torch-geometric

### Removed
- Redundant tutorials and snippets module
- CONTRIBUTING.md and docs/ reference documentation

## [0.1.3] - 2025-11-23

### Added
- Complete example progression (00-08) covering ML lifecycle
- Example 07: LoRA fine-tuning for Large Language Models
- Example 08: Graph Neural Networks with PyTorch Geometric
- New snippets: agent.py, duckdb_sql.py, openvino_1.py, rag.py
- Tools directory with ollama_manager.py and openvino_manager.py
- Comprehensive reference documentation in docs/ directory:
  - marimo-quickstart.md
  - polars-quickstart.md
  - plotly-quickstart.md
  - pina-quickstart.md
  - integration-patterns.md
- Docker helper scripts (marimo-flow-agent, marimo-flow-code)
- .marimo.toml configuration for runtime settings

### Changed
- Reorganized examples with progressive numbering (00-08)
- Restructured project layout for better clarity
- Updated README.md with complete project structure
- Enhanced .gitignore for user-specific configs and cache files
- Improved Docker configuration

### Removed
- experimental/ directory (split into examples/ and tools/)
- Domain-specific apps (astrophotography, cosmos analysis)
- Redundant PyTorch and snippet files
- configs/ empty directory
- Old example files with inconsistent naming

### Fixed
- Git repository structure consistency
- File organization in root directory
- Documentation alignment with actual structure

## [0.1.2] - 2025-10-18

### Added
- PINA (Physics-Informed Neural Networks) integration
- PyTorch Geometric snippets for graph neural networks
- Enhanced Dockerfile with PyG dependencies

### Changed
- Updated dependencies to latest versions
- Improved Docker build process

## [0.1.1] - 2025-07-14

### Added
- Docker configuration with docker-compose
- Custom CSS for Marimo UI styling
- GitHub Pages workflow (Jekyll)
- Python publish workflow for PyPI

### Changed
- Updated Python version requirement to 3.11+
- Cleaned up configuration files
- Refactored Marimo Flow project structure

### Fixed
- Configuration file consistency
- Project metadata alignment

## [0.1.0] - 2025-07-08

### Added
- Initial release of Marimo Flow
- Core ML pipeline examples (00-06)
- MLflow integration for experiment tracking
- Marimo reactive notebooks
- Basic snippets for common patterns:
  - mlflow_setup.py
  - interactive_params.py
  - data_loading.py
  - altair_visualization.py
- Docker support with docker-compose
- SQLite backend for MLflow
- Progressive example structure
- MIT License
- Basic README.md documentation

### Features
- Reactive notebook development with Marimo
- Seamless MLflow experiment tracking
- Interactive parameter tuning
- Model registry and versioning
- Production pipeline examples
- Git-friendly .py notebooks
- Docker one-command setup

---

## Version History Summary

- **0.3.0** (2026-04-21) - Multi-agent PINA team (`pydantic-graph` + MLflow + Ollama Cloud), CITATION.cff
- **0.2.0** (2026-03-26) - Multi-platform Docker, PINA integration, MCP servers, simplified deps
- **0.1.3** (2025-11-23) - Major restructuring, advanced examples, comprehensive docs
- **0.1.2** (2025-10-18) - PINA and PyG integration
- **0.1.1** (2025-07-14) - Docker and CI/CD improvements
- **0.1.0** (2025-07-08) - Initial release

[Unreleased]: https://github.com/synapticore-io/marimo-flow/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/synapticore-io/marimo-flow/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/synapticore-io/marimo-flow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/synapticore-io/marimo-flow/releases/tag/v0.1.0
