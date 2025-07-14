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
COPY --chown=marimo-flow:marimo-flow pyproject.toml uv.lock README.md ./

# Create necessary directories with proper ownership
RUN mkdir -p /app/data/experiments/db /app/data/experiments/artifacts /app/data/experiments/models /app/data/experiments/runs \
    && chown -R marimo-flow:marimo-flow /app

# Copy startup script
COPY --chown=marimo-flow:marimo-flow docker/start.sh docker/.marimo.toml docker/custom.css ./
RUN chmod +x ./start.sh

# Switch to marimo-flow user for dependency installation
USER marimo-flow

# install packages
RUN uv sync --no-dev

RUN uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html



# Expose ports
EXPOSE 2718 5000

# Default command (can be overridden by docker-compose)
CMD ["./start.sh"]
