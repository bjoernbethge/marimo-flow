#!/bin/bash
set -e

echo "=== Marimo-Flow Container Startup ==="

# Create necessary directories if they don't exist
echo "Setting up data directories..."
mkdir -p /app/data/mlflow/{db,artifacts,models,runs}

# Set MLflow environment variables
export MLFLOW_BACKEND_STORE_URI=sqlite:////app/data/mlflow/db/mlflow.db
export MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/data/mlflow/artifacts
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

# Wait for MLflow to be ready
echo "Waiting for MLflow to be ready..."
for i in {1..30}; do
    if curl -sf http://localhost:5000/health >/dev/null 2>&1; then
        echo "✓ MLflow is ready"
        break
    fi
    if ! kill -0 $MLFLOW_PID 2>/dev/null; then
        echo "ERROR: MLflow server process died!"
        cat /tmp/mlflow.log
        exit 1
    fi
    echo "  Attempt $i/30..."
    sleep 2
done

if ! curl -sf http://localhost:5000/health >/dev/null 2>&1; then
    echo "ERROR: MLflow did not become ready in time!"
    cat /tmp/mlflow.log
    exit 1
fi

echo "✓ MLflow server started successfully"

# Start Marimo in foreground
echo "Starting Marimo on port 2718..."
echo "=== Access the interface at: http://localhost:2718 ==="
echo "=== MLflow Dashboard at: http://localhost:5000 ==="
echo ""

# Use examples if available (via volume mount), otherwise fall back to /app
if [ -d /app/examples ]; then
    marimo edit --host 0.0.0.0 --port 2718 --no-token /app/examples
else
    marimo edit --host 0.0.0.0 --port 2718 --no-token /app
fi
