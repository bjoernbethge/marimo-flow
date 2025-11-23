# Marimo Flow 🌊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Marimo](https://img.shields.io/badge/Marimo-Latest-orange?logo=python&logoColor=white)](https://marimo.io)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue?logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Marimo Flow Demo](https://raw.githubusercontent.com/bjoernbethge/marimo-flow/main/asset/marimo-flow.mp4)

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

**Progressive ML Pipeline** - Notebooks that build upon each other:

### 📊 [00 Data Exploration](examples/00_data_exploration.py)
- Interactive dataset analysis
- Statistical summaries and distributions
- Correlation heatmaps
- PCA and t-SNE dimensionality reduction
- Multiple built-in datasets (Wine, Iris, Breast Cancer, Diabetes)

### 🔧 [01 Feature Engineering](examples/01_feature_engineering.py)
- Feature selection techniques
- Scaling and normalization
- Polynomial features
- Feature importance analysis
- Interactive parameter tuning

### 🚀 [02 Basic ML Workflow](examples/02_basic_ml_workflow.py)
- End-to-end ML pipeline with Random Forest
- Real-time parameter tuning
- MLflow experiment tracking
- Interactive visualizations
- Model evaluation and metrics

### 🏆 [03 Model Comparison](examples/03_model_comparison.py)
- Compare multiple algorithms (RF, GB, LR, SVM)
- Cross-validation analysis
- Performance benchmarking
- Side-by-side metric comparison
- Best model selection

### 🎯 [04 Hyperparameter Tuning](examples/04_hyperparameter_tuning.py)
- Automated optimization with Optuna
- Bayesian hyperparameter search
- MLflow integration for all trials
- Real-time optimization progress
- Parameter importance analysis

### 📦 [05 Model Registry](examples/05_model_registry.py)
- Model versioning and registration
- Stage management (Staging → Production)
- Model search and discovery
- Model loading and inference
- Production deployment workflow

### 🚀 [06 Production Pipeline](examples/06_production_pipeline.py)
- Complete end-to-end ML pipeline
- Data validation and quality checks
- Model validation gates
- Automated deployment
- Production monitoring

### 🎓 [07 LoRA Fine-tuning](examples/07_lora_finetuning.py)
- Large Language Model fine-tuning
- LoRA (Low-Rank Adaptation) techniques
- Efficient parameter-efficient training
- MLflow integration for LLM experiments

### 🔗 [08 Graph Neural Networks](examples/08_gnn_hetero_demo.py)
- Heterogeneous graph neural networks
- PyTorch Geometric integration
- Advanced GNN architectures
- Graph-based learning workflows

**Workflow**: Start with 00 and progress through 08 for a complete ML lifecycle from basics to advanced topics.

## Project Structure 📁

```
marimo-flow/
├── examples/                    # Progressive ML pipeline notebooks (00-08)
│   ├── 00_data_exploration.py      # Data analysis and exploration
│   ├── 01_feature_engineering.py   # Feature engineering techniques
│   ├── 02_basic_ml_workflow.py     # Basic ML pipeline with MLflow
│   ├── 03_model_comparison.py      # Multi-model comparison
│   ├── 04_hyperparameter_tuning.py # Optuna optimization
│   ├── 05_model_registry.py        # Model registry & deployment
│   ├── 06_production_pipeline.py   # End-to-end production pipeline
│   ├── 07_lora_finetuning.py       # LLM fine-tuning with LoRA
│   └── 08_gnn_hetero_demo.py       # Graph Neural Networks
├── snippets/                   # Reusable code patterns
│   ├── mlflow_setup.py             # MLflow configuration
│   ├── interactive_params.py       # Interactive controls
│   ├── data_loading.py             # Data utilities
│   ├── altair_visualization.py     # Visualization patterns
│   ├── agent.py                    # AI agent integration
│   ├── duckdb_sql.py               # DuckDB query patterns
│   ├── openvino_1.py               # OpenVINO inference
│   └── rag.py                      # RAG pipeline patterns
├── tools/                       # Utility tools
│   ├── ollama_manager.py           # Local LLM orchestration
│   └── openvino_manager.py         # Model serving utilities
├── refs/                        # Reference documentation
│   ├── marimo-quickstart.md        # Marimo guide
│   ├── polars-quickstart.md        # Polars guide
│   ├── plotly-quickstart.md        # Plotly guide
│   ├── pina-quickstart.md          # PINA guide
│   └── integration-patterns.md     # Integration examples
├── data/
│   └── mlflow/                  # MLflow storage
│       ├── artifacts/           # Model artifacts
│       ├── db/                  # SQLite database
│       └── prompts/             # Prompt templates
├── docker/                      # Docker configuration
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

### 📝 About Snippets

The `snippets/` directory contains reusable code patterns that can be imported into Marimo notebooks. These are optional utilities - all functionality is already integrated into the main examples. Use them if you want to extract common patterns for reuse across multiple notebooks.

### 🛠️ About Tools

The `tools/` directory contains standalone utility scripts for managing external services:
- **ollama_manager.py**: Manage local LLM deployments with Ollama
- **openvino_manager.py**: Model serving and inference with OpenVINO

### 📚 About References

The `refs/` directory contains comprehensive LLM-friendly documentation for key technologies:
- Quick-start guides for Marimo, Polars, Plotly, and PINA
- Integration patterns and best practices
- Code examples and common workflows

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
