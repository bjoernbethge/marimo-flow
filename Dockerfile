FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create marimo-flow user
RUN groupadd --gid 1000 marimo-flow \
    && useradd --uid 1000 --gid marimo-flow --shell /bin/bash --create-home marimo-flow

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.7.8 /uv /uvx /bin/

# Set working directory and give ownership to marimo-flow user
WORKDIR /app
RUN chown -R marimo-flow:marimo-flow /app

# Copy project files and set ownership
COPY --chown=marimo-flow:marimo-flow pyproject.toml uv.lock ./

# Switch to marimo-flow user for dependency installation
USER marimo-flow

# Install dependencies
RUN uv sync --frozen --no-dev

# Switch back to root to create directories with proper permissions
USER root

# Create MLflow directories with proper ownership
RUN mkdir -p /mlflow/db /mlflow/artifacts /mlflow/models /mlflow/experiments /mlflow/runs /mlflow/prompts \
    && chown -R marimo-flow:marimo-flow /mlflow

# Switch back to marimo-flow user for runtime
USER marimo-flow

# Expose ports
EXPOSE 2718 5000

# Default command (can be overridden by docker-compose)
CMD ["uv", "run", "mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]

