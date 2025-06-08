# Marimo Flow 🌊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Marimo](https://img.shields.io/badge/Marimo-Latest-orange?logo=python&logoColor=white)](https://marimo.io)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black)](https://github.com/psf/black)

**Modern reactive ML development with Marimo notebooks and MLflow experiment tracking**

## Why Marimo Flow is Powerful 🚀

**Marimo Flow** combines the best of reactive notebook development with robust ML experiment tracking:

- **🔄 Reactive Development**: Marimo's dataflow graph ensures your notebooks are always consistent - change a parameter and watch your entire pipeline update automatically
- **🤖 AI-Enhanced Workflow**: Built-in GitHub Copilot support and AI assistants accelerate your ML development
- **📊 Seamless ML Pipeline**: MLflow integration tracks every experiment, model, and metric without breaking your flow

This combination eliminates the reproducibility issues of traditional notebooks while providing enterprise-grade experiment tracking.

## Features ✨

- **📓 Marimo Reactive Notebooks**: Git-friendly `.py` notebooks with automatic dependency tracking
- **🔬 MLflow Experiment Tracking**: Complete ML lifecycle management with model registry
- **🐳 Docker Deployment**: One-command setup with docker-compose
- **💾 SQLite Backend**: Lightweight, file-based storage for experiments
- **🎯 Interactive ML Development**: Real-time parameter tuning with instant feedback

## Quick Start 🏃‍♂️

### With Docker (Recommended)

```bash
# Clone and start services
git clone <repository-url>
cd marimo-flow
docker-compose up -d

# Access services
# Marimo: http://localhost:2718
# MLflow: http://localhost:5000
```

### Local Development

```bash
# Install dependencies
uv sync

# Start MLflow server
uv run mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///data/mlflow/db/mlflow.db --default-artifact-root ./data/mlflow/artifacts

# Start Marimo (in another terminal)
uv run marimo edit examples/
```

## Example Notebooks 📚

Explore our focused, production-ready notebooks:

### 🚀 [Basic ML Workflow](examples/01_basic_ml_workflow.py)
- Simple classification with Random Forest
- Real-time parameter tuning
- MLflow experiment tracking
- Interactive visualizations

### 🏆 [Model Comparison](examples/02_model_comparison.py)
- Compare multiple algorithms (RF, GB, LR, SVM)
- Cross-validation analysis
- Performance benchmarking
- Built-in datasets (Wine, Iris, Breast Cancer)

### 🔍 [Data Exploration](examples/03_data_exploration.py)
- Interactive statistical analysis
- Correlation heatmaps
- Distribution visualizations
- PCA and t-SNE dimensionality reduction

Each notebook demonstrates reactive development principles and follows Marimo best practices for maintainable, reproducible ML code.

## Project Structure 📁

```
marimo-flow/
├── examples/                    # Marimo notebooks
│   ├── 01_basic_ml_workflow.py     # Basic ML pipeline
│   ├── 02_model_comparison.py      # Multi-model comparison
│   └── 03_data_exploration.py      # Interactive data analysis
├── data/
│   └── mlflow/                  # MLflow storage
│       ├── artifacts/           # Model artifacts
│       ├── db/                  # SQLite database
│       └── prompts/             # Prompt templates
├── docker-compose.yaml          # Service orchestration
├── Dockerfile                   # Container definition
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

## Configuration ⚙️

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URL (default: `http://localhost:5000`)
- `MLFLOW_BACKEND_STORE_URI`: Database connection string
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: Artifact storage location

### Docker Services

- **Marimo**: Port 2718 - Interactive notebook environment
- **MLflow**: Port 5000 - Experiment tracking UI

## Pre-installed ML & Data Science Stack 📦

### Machine Learning & Scientific Computing
- **[scikit-learn](https://scikit-learn.org/)** `^1.5.2` - Machine learning library
- **[NumPy](https://numpy.org/)** `^2.1.3` - Numerical computing
- **[pandas](https://pandas.pydata.org/)** `^2.2.3` - Data manipulation and analysis
- **[PyArrow](https://arrow.apache.org/docs/python/)** `^18.0.0` - Columnar data processing
- **[SciPy](https://scipy.org/)** `^1.14.1` - Scientific computing
- **[matplotlib](https://matplotlib.org/)** `^3.9.2` - Plotting library

### High-Performance Data Processing
- **[Polars](https://pola.rs/)** `^1.12.0` - Lightning-fast DataFrame library
- **[DuckDB](https://duckdb.org/)** `^1.1.3` - In-process analytical database
- **[Altair](https://altair-viz.github.io/)** `^5.4.1` - Declarative statistical visualization

### AI & LLM Integration
- **[OpenAI](https://platform.openai.com/docs/)** `^1.54.4` - GPT API integration
- **[FastAPI](https://fastapi.tiangolo.com/)** `^0.115.4` - Modern web framework
- **[Pydantic](https://docs.pydantic.dev/)** `^2.10.2` - Data validation

### Database & Storage
- **[SQLAlchemy](https://www.sqlalchemy.org/)** `^2.0.36` - SQL toolkit and ORM
- **[Alembic](https://alembic.sqlalchemy.org/)** `^1.14.0` - Database migrations
- **[SQLGlot](https://sqlglot.com/)** `^25.30.2` - SQL parser and transpiler

### Web & API
- **[Starlette](https://www.starlette.io/)** `^0.41.2` - ASGI framework
- **[Uvicorn](https://www.uvicorn.org/)** `^0.32.0` - ASGI server
- **[httpx](https://www.python-httpx.org/)** `^0.27.2` - HTTP client

### Development Tools
- **[Black](https://black.readthedocs.io/)** `^24.10.0` - Code formatter
- **[Ruff](https://docs.astral.sh/ruff/)** `^0.7.4` - Fast Python linter
- **[pytest](https://docs.pytest.org/)** `^8.3.3` - Testing framework
- **[MyPy](https://mypy.readthedocs.io/)** `^1.13.0` - Static type checker

## API Endpoints 🔌

### MLflow REST API
- **Experiments**: `GET /api/2.0/mlflow/experiments/list`
- **Runs**: `GET /api/2.0/mlflow/runs/search`
- **Models**: `GET /api/2.0/mlflow/registered-models/list`

### Marimo Server
- **Notebooks**: `GET /` - File browser and editor
- **Apps**: `GET /run/<notebook>` - Run notebook as web app

## Contributing 🤝

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the coding standards
4. Test your changes: `uv run pytest`
5. Submit a pull request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ using Marimo and MLflow**
