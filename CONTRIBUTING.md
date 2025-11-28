# Contributing to Marimo Flow

Thank you for your interest in contributing to Marimo Flow! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/marimo-flow.git
   cd marimo-flow
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/bjoernbethge/marimo-flow.git
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Docker (optional, for containerized development)

### Local Installation

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or using pip
pip install -e ".[dev,full]"
```

### Start Development Services

```bash
# Terminal 1: Start MLflow server
uv run mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri sqlite:///data/mlflow/db/mlflow.db \
  --default-artifact-root ./data/mlflow/artifacts

# Terminal 2: Start Marimo
uv run marimo edit examples/
```

### Using Docker

```bash
# Build and start services
docker-compose up -d

# Access services
# Marimo: http://localhost:2718
# MLflow: http://localhost:5000
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-example` - New features
- `fix/mlflow-connection` - Bug fixes
- `docs/update-readme` - Documentation updates
- `refactor/cleanup-snippets` - Code refactoring

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(examples): add hyperparameter tuning notebook

Add new example demonstrating Optuna integration with MLflow
for automated hyperparameter optimization.

fix(mlflow): resolve tracking URI connection issue

Update MLflow client initialization to properly handle
environment variables and default configurations.
```

## Code Standards

### Python Style

- **Formatter**: Black (line length: 88)
- **Linter**: Ruff
- **Type Checker**: MyPy
- **Docstrings**: Google style

Run formatting and linting:

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Type check
uv run mypy .
```

### Marimo Notebook Guidelines

1. **Progressive Structure**: Examples should build on previous notebooks
2. **Interactive Elements**: Use `mo.ui` components for parameters
3. **Documentation**: Include clear markdown cells explaining concepts
4. **MLflow Integration**: Track relevant experiments and metrics
5. **Reproducibility**: Set random seeds where applicable

### Code Organization

```python
# Good: Clear imports, type hints, docstrings
import marimo as mo
import mlflow
from typing import Dict, List

def train_model(params: Dict[str, float]) -> float:
    """Train a model with given parameters.

    Args:
        params: Dictionary of hyperparameters

    Returns:
        Model accuracy score
    """
    with mlflow.start_run():
        # Implementation
        pass
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=marimo_flow

# Run specific test file
uv run pytest tests/test_examples.py
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names

```python
def test_example_notebook_executes():
    """Test that example notebook runs without errors."""
    # Test implementation
    pass
```

## Pull Request Process

### Before Submitting

1. **Update from upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run quality checks**:
   ```bash
   # Format
   uv run black .

   # Lint
   uv run ruff check .

   # Type check
   uv run mypy .

   # Tests
   uv run pytest
   ```

3. **Update documentation**: If adding features, update README.md

### Creating Pull Request

1. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub** with:
   - Clear title describing the change
   - Description of what changed and why
   - Link to related issues (if applicable)
   - Screenshots/examples (for UI changes)

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
Describe how you tested your changes

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## Types of Contributions

### Adding Examples

New example notebooks should:
1. Fit the progressive learning structure (00-08)
2. Demonstrate practical ML workflows
3. Include MLflow tracking
4. Be well-documented with markdown cells
5. Follow existing naming conventions

### Improving Documentation

- Fix typos or unclear explanations
- Add code examples
- Improve setup instructions
- Translate documentation

### Reporting Bugs

Use GitHub Issues with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)
- Relevant logs/screenshots

### Suggesting Features

Open a GitHub Issue describing:
- The problem your feature solves
- Proposed solution
- Alternative approaches considered
- Example use cases

## Questions?

- Open a GitHub Issue for questions
- Check existing issues and discussions
- Review documentation in `docs/` directory

## Recognition

Contributors will be acknowledged in:
- CHANGELOG.md
- GitHub contributors page
- Release notes (for significant contributions)

Thank you for contributing to Marimo Flow!
