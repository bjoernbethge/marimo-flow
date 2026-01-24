# Production Deployment with MLflow

Complete guide for deploying and monitoring ML/GenAI models in production.

## FastAPI Integration

```python
from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
@mlflow.trace
def handle_chat(request: Request, chat_request: ChatRequest):
    # Extract context from headers
    session_id = request.headers.get("X-Session-ID")
    user_id = request.headers.get("X-User-ID")

    # Update trace with context
    mlflow.update_current_trace(
        tags={
            "mlflow.trace.session": session_id,
            "mlflow.trace.user": user_id,
            "environment": "production"
        }
    )

    # Your LLM logic here
    response = process_chat(chat_request.message)

    return {"response": response}
```

## Async Logging

Enable asynchronous logging for production to minimize latency impact:

```bash
# Enable async logging for production
export MLFLOW_ENABLE_ASYNC_TRACE_LOGGING=true
export MLFLOW_TRACE_SAMPLING_RATIO=1.0
```

## Production Context

Always add production context to traces:

```python
import mlflow
import os

mlflow.update_current_trace(
    tags={
        "mlflow.trace.session": session_id,
        "mlflow.trace.user": user_id,
        "environment": "production",
        "region": os.getenv("REGION"),
        "app_version": os.getenv("APP_VERSION")
    }
)
```

## Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlflow-config
data:
  MLFLOW_TRACKING_URI: 'http://mlflow-server:5000'
  MLFLOW_EXPERIMENT_NAME: 'production-genai-app'
  MLFLOW_ENABLE_ASYNC_TRACE_LOGGING: 'true'
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genai-app
spec:
  template:
    spec:
      containers:
        - name: app
          image: your-app:latest
          envFrom:
            - configMapRef:
                name: mlflow-config
```

## Serve Models

```bash
# Serve model locally
mlflow models serve -m models:/MyModel/Production -p 5001

# Test served model
curl -X POST http://localhost:5001/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs": [[1, 2, 3, 4]]}'
```

## Docker Deployment

```bash
# Build Docker image
mlflow models build-docker -m models:/MyModel/Production -n my-model

# Run container
docker run -p 5001:8080 my-model
```

## Complete FastAPI Production Example

```python
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import mlflow
import logging
from datetime import datetime

app = FastAPI()
logger = logging.getLogger(__name__)

# Configure MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("production-api")

# Load production model
model = mlflow.pyfunc.load_model("models:/MyModel@champion")

class PredictionRequest(BaseModel):
    features: list[float]
    user_id: str
    session_id: str

class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    trace_id: str

@app.post("/predict", response_model=PredictionResponse)
@mlflow.trace
async def predict(request: Request, pred_request: PredictionRequest):
    try:
        # Update trace with context
        mlflow.update_current_trace(
            tags={
                "mlflow.trace.user": pred_request.user_id,
                "mlflow.trace.session": pred_request.session_id,
                "environment": "production",
                "endpoint": "/predict"
            }
        )

        # Make prediction
        prediction = model.predict([pred_request.features])[0]

        # Get trace ID
        trace_id = mlflow.get_last_active_trace_id()

        return PredictionResponse(
            prediction=prediction,
            model_version="champion",
            trace_id=trace_id
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
```

## Cloud Deployment - AWS SageMaker

```python
import mlflow.sagemaker as mfs

# Deploy to SageMaker
app_name = "my-app"
model_uri = "models:/MyModel/Production"
region = "us-west-2"

mfs.deploy(
    app_name=app_name,
    model_uri=model_uri,
    region_name=region,
    mode="create"
)
```

## Cloud Deployment - Azure ML

```python
from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice, Webservice
import mlflow.azureml

# Connect to workspace
ws = Workspace.from_config()

# Build Azure ML model from MLflow model
azure_model, azure_model_path = mlflow.azureml.build_image(
    model_uri="models:/MyModel/Production",
    workspace=ws,
    model_name="my-model",
    image_name="my-model-image",
    synchronous=True
)

# Deploy
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    tags={"model": "production"},
    description="Production ML model"
)

service = Model.deploy(
    workspace=ws,
    name="my-model-service",
    models=[azure_model],
    deployment_config=aci_config
)

service.wait_for_deployment(show_output=True)
```

## Monitoring Production Models

```python
import mlflow
from datetime import datetime, timedelta

def monitor_production_model():
    """Monitor production model performance"""

    # Get traces from last 24 hours
    yesterday = datetime.now() - timedelta(days=1)

    traces = mlflow.search_traces(
        filter_string=f"tags.environment = 'production' AND timestamp > {yesterday.timestamp()}",
        max_results=10000
    )

    # Analyze metrics
    latencies = [trace.info.execution_time_ms for trace in traces]
    avg_latency = sum(latencies) / len(latencies)

    # Log monitoring metrics
    with mlflow.start_run(run_name=f"monitoring_{datetime.now().date()}"):
        mlflow.log_metric("avg_latency_ms", avg_latency)
        mlflow.log_metric("total_requests", len(traces))
        mlflow.log_metric("error_rate", calculate_error_rate(traces))

    # Alert if degradation
    if avg_latency > 1000:  # 1 second
        send_alert(f"High latency detected: {avg_latency}ms")
```

## Load Balancing Multiple Models

```python
from fastapi import FastAPI
import mlflow
import random

app = FastAPI()

# Load multiple model versions
champion = mlflow.pyfunc.load_model("models:/MyModel@champion")
challenger = mlflow.pyfunc.load_model("models:/MyModel@challenger")

@app.post("/predict")
@mlflow.trace
async def predict_with_ab_test(features: list[float], user_id: str):
    # A/B test: 90% champion, 10% challenger
    if random.random() < 0.9:
        model = champion
        model_version = "champion"
    else:
        model = challenger
        model_version = "challenger"

    mlflow.update_current_trace(
        tags={
            "model_version": model_version,
            "mlflow.trace.user": user_id
        }
    )

    prediction = model.predict([features])[0]
    return {"prediction": prediction, "model_version": model_version}
```

## Graceful Model Updates

```python
import mlflow
from threading import Lock

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.lock = Lock()
        self.load_latest_model()

    def load_latest_model(self):
        """Load latest production model"""
        with self.lock:
            self.model = mlflow.pyfunc.load_model("models:/MyModel@champion")
            # Get model version info
            client = mlflow.MlflowClient()
            model_info = client.get_model_version_by_alias("MyModel", "champion")
            self.model_version = model_info.version

    def predict(self, features):
        """Thread-safe prediction"""
        with self.lock:
            return self.model.predict(features)

    def refresh_if_updated(self):
        """Check and reload if model updated"""
        client = mlflow.MlflowClient()
        latest_info = client.get_model_version_by_alias("MyModel", "champion")

        if latest_info.version != self.model_version:
            logger.info(f"New model version detected: {latest_info.version}")
            self.load_latest_model()

# Use in FastAPI
model_manager = ModelManager()

@app.post("/predict")
async def predict(features: list[float]):
    # Periodically check for updates
    model_manager.refresh_if_updated()
    return {"prediction": model_manager.predict([features])[0]}
```

## Troubleshooting Production Issues

```python
# Check if tracing is enabled
import mlflow

print(f"Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {mlflow.get_experiment_by_name('production-api')}")

# Manual trace cleanup
mlflow.end_run()  # End any stuck runs

# Test trace logging
@mlflow.trace
def test_trace():
    return "test"

result = test_trace()
trace_id = mlflow.get_last_active_trace_id()
print(f"Trace ID: {trace_id}")
```
