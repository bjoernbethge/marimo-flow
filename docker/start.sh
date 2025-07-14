#!/bin/bash

# Activate venv for CLI tools and Python modules
source /app/.venv/bin/activate

# Create necessary directories if they don't exist
mkdir -p /app/data/experiments/db
mkdir -p /app/data/experiments/artifacts
mkdir -p /app/data/experiments/models
mkdir -p /app/data/experiments/runs

# Set MLflow environment variables
export MLFLOW_BACKEND_STORE_URI=sqlite:////app/data/experiments/db/mlflow.db
export MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/data/experiments/artifacts
export MLFLOW_SERVE_ARTIFACTS=true
export MLFLOW_HOST=0.0.0.0
export MLFLOW_PORT=5000

# Start MLflow server in background
echo "Starting MLflow server..."
mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --serve-artifacts &

# Wait for MLflow to start
sleep 5

# Start Marimo in foreground
echo "Starting Marimo..."
marimo edit --host 0.0.0.0 --port 2718 --no-token
