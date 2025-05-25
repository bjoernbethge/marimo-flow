# Multi-stage build for MLflow with Astral's uv following official best practices
FROM python:3.11-slim-bookworm AS builder

# Install uv from official Astral image - ALWAYS pin version!
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Environment variables for optimal build
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency files first for better layer caching
COPY pyproject.toml ./

# Install dependencies using the lockfile - creates .venv automatically
RUN --mount=type=cache,target=/root/.cache/uv \
    uv lock && \
    uv sync --locked --no-install-project


# Production stage - minimal runtime image
FROM python:3.11-slim-bookworm

# Metadata labels following OCI spec
LABEL org.opencontainers.image.title="MLflow Server" \
      org.opencontainers.image.description="MLflow Server with SQLite backend optimized with uv" \
      org.opencontainers.image.source="https://github.com/bjoernbethge/mlflow-server" \
      org.opencontainers.image.version="latest"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    TZ=Europe/Berlin

# Install only necessary runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Set working directory to /app (where volumes will be mounted)
WORKDIR /app

# Expose default MLflow port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl --fail --silent --show-error http://localhost:8080/health || exit 1

# Default command - paths relative to /app workdir
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080", "--backend-store-uri", "sqlite:///app/db/mlflow.db", "--default-artifact-root", "/app/artifacts"]