#!/bin/bash
set -e

echo "=== Marimo-Flow Container Startup ==="

# Create necessary directories if they don't exist
echo "Setting up data directories..."
mkdir -p /app/data/experiments/{db,artifacts,models,runs}

# Set MLflow environment variables
export MLFLOW_BACKEND_STORE_URI=sqlite:////app/data/experiments/db/mlflow.db
export MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/data/experiments/artifacts
export MLFLOW_SERVE_ARTIFACTS=true
export MLFLOW_HOST=0.0.0.0
export MLFLOW_PORT=5000

# Start MLflow server in background
echo "Starting MLflow server on port ${MLFLOW_PORT}..."
mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --serve-artifacts > /tmp/mlflow.log 2>&1 &

MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"

# Wait for MLflow to start
sleep 5

# Verify MLflow is running
if ! kill -0 $MLFLOW_PID 2>/dev/null; then
    echo "ERROR: MLflow server failed to start!"
    cat /tmp/mlflow.log
    exit 1
fi

echo "✓ MLflow server started successfully"

# Start Marimo in foreground
echo "Starting Marimo on port 2718..."
echo "=== Access the interface at: http://localhost:2718 ==="
echo "=== MLflow Dashboard at: http://localhost:5000 ==="
echo ""

marimo edit --host 0.0.0.0 --port 2718 --no-token
