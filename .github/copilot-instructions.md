# Marimo Flow - GitHub Copilot Instructions

This is a Python-based interactive machine learning notebook project combining Marimo (reactive notebooks), MLflow (experiment tracking), and PINA (Physics-Informed Neural Networks). The project emphasizes reactive programming, reproducible ML workflows, and AI-first development with Model Context Protocol (MCP) integration.

## Project Overview

**Purpose**: Interactive ML notebooks with reactive updates, AI assistance, and MLflow tracking
**Target Users**: ML researchers, data scientists, and developers building reactive ML pipelines
**Key Features**:
- Reactive notebook execution with automatic dependency tracking
- MLflow integration for experiment tracking and model registry
- MCP (Model Context Protocol) support for AI-powered development assistance
- PINA integration for Physics-Informed Neural Networks
- Multi-agent development system (Planner/Worker/Judge architecture)
- Docker-based deployment for reproducibility

## Agent Architecture

This project implements a **multi-agent development system** inspired by [Cursor's autonomous agent research](https://cursor.com/blog/agents):

- **Planner Agent** - Explores codebase, breaks down work into tasks
- **Worker Agents** - Specialized executors (Notebook, MLflow, PINA, Data, Testing)
- **Judge Agent** - Evaluates work quality, decides ship/iterate/escalate

**Core Principles**:
1. Clear role separation (hierarchical, not flat)
2. Workers self-coordinate conflicts (no integrator)
3. Model-role matching (different models for different tasks)
4. Prompts > Infrastructure
5. Workers own hard problems end-to-end

**Documentation**: See `.github/agents/README.md` for full architecture details and prompt engineering guidelines.

## Technology Stack

### Core Technologies
- **Python**: 3.11+ (primary language)
- **Marimo**: Reactive notebook framework (git-friendly `.py` notebooks)
- **MLflow**: ML experiment tracking and model registry
- **PINA**: Physics-Informed Neural Networks library

### Data & ML Libraries
- **Polars**: Preferred over pandas for data manipulation (better performance)
- **DuckDB**: In-process analytical database for data exploration
- **Altair**: Declarative statistical visualization (preferred for 2D interactive viz)
- **Plotly Express**: Interactive 3D visualizations
- **PyTorch**: Deep learning framework (via PINA)
- **Optuna**: Hyperparameter optimization
- **scikit-learn**: Traditional ML algorithms

### Development Tools
- **Package Manager**: `uv` (preferred) or `pip`
- **Formatter**: Black (line length: 79 for notebooks, 88 for library code)
- **Linter**: Ruff
- **Type Checker**: MyPy
- **Docstrings**: Google style for library code, NumPy style for notebooks

### AI & MCP Integration
- **MCP Presets**: `context7` (universal library docs), `marimo` (notebook-specific help)
- **Local LLM**: Ollama with `gpt-oss:20b-cloud` model for code completion
- **AI Rules**: See `.marimo.toml` for inline AI rules and preferences

## Development Workflow

### Environment Setup
```bash
# Install dependencies (preferred method)
uv sync --all-extras

# Or using pip
pip install -e ".[dev,full]"
```

### Running Services

**Local Development:**
```bash
# Terminal 1: Start MLflow server
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/experiments/db/mlflow.db \
  --default-artifact-root ./data/experiments/artifacts \
  --serve-artifacts

# Terminal 2: Start Marimo
uv run marimo edit examples/
```

**Docker (Recommended for Production):**
```bash
# Build and start services
docker compose -f docker/docker-compose.yaml up --build -d

# Services available at:
# Marimo: http://localhost:2718
# MLflow: http://localhost:5000
```

### Code Quality Checks

**Before Each Commit:**
```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type check
uv run mypy .

# Run tests (if available)
uv run pytest
```

## Code Standards

### Python Style Guidelines
1. Follow PEP 8 conventions
2. Use Black for formatting (line length: 79 for notebooks, 88 for library code)
3. Use Ruff for linting with strict mode enabled
4. Include type hints for all function parameters and return values
5. Write docstrings for all public functions:
   - Library code: Google style
   - Notebooks: NumPy style with parameter descriptions
6. Handle errors with try/except blocks and provide informative error messages

### Marimo Notebook Conventions

**Reactive Programming Patterns:**
1. **Cell Dependencies**: Variables referenced in a cell create automatic dependencies
2. **State Management**: Use `mo.state()` for mutable state that needs to persist across reruns
3. **UI Elements**: Use `mo.ui.*` components for interactive parameters (sliders, dropdowns, etc.)
4. **Reactive Forms**: Group related inputs with `mo.ui.form()` for batch updates
5. **Progressive Structure**: Examples should build on concepts from previous notebooks (01 → 02 → 03, etc.)

**Example Reactive Pattern:**
```python
import marimo as mo

# Define UI element (automatically creates reactivity)
slider = mo.ui.slider(start=0, stop=100, value=50, label="Parameter")

# Use the element's value (cell will re-run when slider changes)
result = compute_something(slider.value)
```

**MLflow Integration Patterns:**
1. Always check for active run before starting a new one:
   ```python
   if mlflow.active_run():
       mlflow.end_run()
   
   with mlflow.start_run(run_name="descriptive_name"):
       mlflow.log_params(params)
       mlflow.log_metrics(metrics)
   ```
2. Use descriptive run names and tags for experiment organization
3. Log artifacts with clear naming conventions
4. Track model hyperparameters, metrics, and artifacts consistently

### Data Processing Preferences
1. **Use Polars over pandas** for better performance and expressive API
2. **Use Altair** for declarative 2D interactive visualizations
3. **Use Plotly Express** (px) for 3D visualizations and complex interactive plots
4. Include proper axis labels, titles, and color schemes in all plots
5. Handle missing data explicitly and document assumptions

### PINA (Physics-Informed Neural Networks) Guidelines
1. Use the Walrus adapter when applicable for improved convergence
2. Implement proper boundary condition handling
3. Include validation against analytical solutions when possible
4. Log physics loss components separately in MLflow
5. Visualize solution fields and residuals during training

## Repository Structure

```
marimo-flow/
├── .github/                     # GitHub configuration
│   ├── copilot-instructions.md  # This file
│   ├── agents/                  # Custom agent definitions
│   └── workflows/               # CI/CD workflows
├── src/marimo_flow/             # Core library code
│   ├── core/                    # PINA integration, training, visualization
│   └── snippets/                # Reusable patterns for import
├── examples/                    # Production-ready marimo notebooks
│   ├── 01_interactive_data_profiler.py
│   ├── 02_mlflow_experiment_console.py
│   ├── 03_pina_walrus_solver.py
│   ├── 04_hyperparameter_tuning.py
│   ├── 05_model_registry.py
│   ├── 06_production_pipeline.py
│   ├── 09_pina_live_monitoring.py
│   └── tutorials/               # 15+ focused learning notebooks
├── snippets/                    # Standalone snippet modules
│   ├── altair_visualization.py  # Chart builders and themes
│   ├── data_explorer_pattern.py # Column filtering + scatter plotting
│   └── pina_basics.py           # Walrus/PINA helpers
├── tools/                       # Utility scripts
│   ├── ollama_manager.py        # Local LLM orchestration
│   └── openvino_manager.py      # Model serving utilities
├── docs/                        # Reference documentation
│   ├── marimo-quickstart.md     # Marimo guide
│   ├── polars-quickstart.md     # Polars guide
│   ├── plotly-quickstart.md     # Plotly guide
│   └── pina-quickstart.md       # PINA guide
├── data/experiments/            # MLflow storage (gitignored)
│   ├── db/                      # SQLite database
│   └── artifacts/               # Model artifacts
├── docker/                      # Docker configuration
│   ├── docker-compose.yaml      # Service orchestration
│   └── start.sh                 # Container startup script
├── pyproject.toml              # Dependencies and build config
├── .marimo.toml                # Marimo editor configuration
└── README.md                   # User-facing documentation
```

### Key File Types

**Marimo Notebooks (`.py` files in `examples/`)**:
- Git-friendly Python scripts that run as reactive notebooks
- Number-prefixed for progressive learning (01, 02, etc.)
- Should be self-contained with markdown documentation cells
- Include MLflow tracking for ML experiments

**Snippet Modules (`snippets/*.py`)**:
- Reusable code patterns for import into notebooks
- Should be well-documented with docstrings
- Avoid external dependencies where possible

**Core Library (`src/marimo_flow/`)**:
- Properly typed, tested, and documented Python modules
- Follow standard library development practices
- Suitable for `from marimo_flow import ...` usage

## Important Guidelines

### When Adding Examples
1. Follow the progressive numbering scheme (01, 02, etc.)
2. Include clear markdown cells explaining concepts
3. Integrate MLflow tracking for relevant metrics
4. Use interactive UI elements (`mo.ui.*`) for parameters
5. Set random seeds for reproducibility: `torch.manual_seed(42)`
6. Document expected runtime and resource requirements

### When Modifying Core Library
1. Maintain backward compatibility unless major version bump
2. Add type hints to all public APIs
3. Write unit tests for new functionality (table-driven tests preferred)
4. Update docstrings following Google style
5. Consider impact on existing notebooks

### When Writing Documentation
1. Keep examples up-to-date with current API
2. Include code snippets that can be copy-pasted
3. Document environment variables and configuration
4. Explain "why" not just "how" for complex patterns

### Model Context Protocol (MCP) Usage
- `context7` preset provides access to library documentation for Polars, Plotly, Altair, etc.
- `marimo` preset provides marimo-specific patterns and best practices
- Configure in `.marimo.toml` under `[mcp]` section
- See [MCP-SETUP.md](../docs/MCP-SETUP.md) for detailed configuration

### Security Considerations
1. Never commit secrets or API keys (use `.env` files, listed in `.gitignore`)
2. Validate user inputs in interactive notebooks
3. Use environment variables for sensitive configuration
4. Review MLflow artifacts before sharing

## Common Tasks

### Creating a New Example Notebook
```bash
# Start marimo in edit mode
uv run marimo edit examples/

# Create new file with appropriate number prefix
# Follow existing examples for structure
# Include: imports, config, data loading, model training, visualization, MLflow tracking
```

### Adding a New Snippet Module
```bash
# Create in snippets/ directory
# Keep focused on a single pattern or utility
# Document with docstrings
# Test by importing in a notebook
```

### Debugging Reactive Issues
1. Check variable dependencies with marimo's dataflow graph
2. Use `mo.stop()` to prevent downstream execution during debugging
3. Verify state is managed correctly (avoid global mutable state)
4. Check that UI elements are properly awaited (`.value` property)

### Working with MLflow
1. Start MLflow server before running notebooks with tracking
2. Check `MLFLOW_TRACKING_URI` environment variable is set correctly
3. Use meaningful experiment names and run names
4. Tag runs with relevant metadata (model type, dataset, etc.)

## Testing Guidelines

- Tests should be placed in `tests/` directory when added (currently no formal test infrastructure exists)
- Use pytest for test execution: `uv run pytest`
- Test naming convention: `test_*.py` for files, `test_*` for functions
- For notebooks: Validate they execute without errors
- For library code: Write comprehensive unit tests with good coverage

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines including:
- Branch naming conventions
- Commit message format (conventional commits)
- Pull request process
- Code review expectations

## Additional Resources

- **Marimo Documentation**: https://docs.marimo.io/
- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **PINA Documentation**: https://mathlab.github.io/PINA/
- **Model Context Protocol**: https://modelcontextprotocol.io/
- **Project README**: [README.md](../README.md)
- **Changelog**: [CHANGELOG.md](../CHANGELOG.md)

## Questions or Issues?

- Check existing documentation in `docs/` directory
- Review similar examples in `examples/` and `examples/tutorials/`
- Open a GitHub Issue for bugs or feature requests
- Refer to `CONTRIBUTING.md` for development questions
