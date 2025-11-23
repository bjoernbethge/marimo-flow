# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CONTRIBUTING.md with comprehensive contribution guidelines
- CHANGELOG.md for tracking version history

### Changed
- Improved .gitignore with Python cache directories
- Moved data files to organized locations (data/ directory)

## [0.1.3] - 2025-11-23

### Added
- Complete example progression (00-08) covering ML lifecycle
- Example 07: LoRA fine-tuning for Large Language Models
- Example 08: Graph Neural Networks with PyTorch Geometric
- New snippets: agent.py, duckdb_sql.py, openvino_1.py, rag.py
- Tools directory with ollama_manager.py and openvino_manager.py
- Comprehensive reference documentation in refs/ directory:
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

- **0.1.3** (2025-11-23) - Major restructuring, advanced examples, comprehensive docs
- **0.1.2** (2025-10-18) - PINA and PyG integration
- **0.1.1** (2025-07-14) - Docker and CI/CD improvements
- **0.1.0** (2025-07-08) - Initial release

[Unreleased]: https://github.com/bjoernbethge/marimo-flow/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/bjoernbethge/marimo-flow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/bjoernbethge/marimo-flow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/bjoernbethge/marimo-flow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/bjoernbethge/marimo-flow/releases/tag/v0.1.0
