# PyG wheels (torch-scatter/sparse/cluster) for torch 2.10 CPU are published for cp312 on Linux, not cp313.
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create marimo-flow user
RUN groupadd --gid 1000 marimo-flow \
    && useradd --uid 1000 --gid marimo-flow --shell /bin/bash --create-home marimo-flow

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /bin/

# Set working directory and give ownership to marimo-flow user
WORKDIR /app
RUN chown -R marimo-flow:marimo-flow /app

# Copy project files and set ownership
COPY --chown=marimo-flow:marimo-flow pyproject.toml uv.lock README.md ./
COPY --chown=marimo-flow:marimo-flow src/ ./src/

# Create necessary directories with proper ownership (created by start.sh)
RUN mkdir -p /app/data \
    && chown -R marimo-flow:marimo-flow /app

# Copy startup script
COPY --chown=marimo-flow:marimo-flow ./docker/start.sh ./docker/.marimo.toml ./docker/custom.css ./
RUN chmod +x ./start.sh

# Switch to marimo-flow user for dependency installation
USER marimo-flow

# Install dependencies (includes torch + torch-geometric)
RUN uv sync --no-dev --extra all

# Install PyG packages for CPU (no CUDA)
RUN uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.11.0+cpu.html

# Activate virtualenv by default
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Expose ports
EXPOSE 2718 5000

# Default command (can be overridden by docker-compose)
CMD ["./start.sh"]
