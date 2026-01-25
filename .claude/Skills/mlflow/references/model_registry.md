# Model Registry with MLflow

Complete guide for managing model lifecycle with the MLflow Model Registry.

## Register Models

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# Register during training
with mlflow.start_run():
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="MyModel"
    )

# Set alias for deployment
client.set_registered_model_alias(
    name="MyModel",
    alias="champion",
    version=1
)

# Load model by alias
model = mlflow.pyfunc.load_model("models:/MyModel@champion")
```

## Model Versioning

```python
from mlflow import MlflowClient

client = MlflowClient()

# Transition model stage
client.transition_model_version_stage(
    name="MyModel",
    version=2,
    stage="Production"
)

# Set tags
mlflow.set_tags({
    "model_type": "random_forest",
    "dataset": "sales_data",
    "framework": "sklearn"
})

# Load specific version
model = mlflow.pyfunc.load_model("models:/MyModel/2")

# Load by stage
model = mlflow.pyfunc.load_model("models:/MyModel/Production")
```

## Model Aliases

Aliases provide a flexible way to reference model versions:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Set aliases for different deployment stages
client.set_registered_model_alias(
    name="MyModel",
    alias="champion",  # Current production model
    version=3
)

client.set_registered_model_alias(
    name="MyModel",
    alias="challenger",  # Model being tested
    version=4
)

client.set_registered_model_alias(
    name="MyModel",
    alias="baseline",  # Baseline for comparison
    version=1
)

# Load models by alias
champion = mlflow.pyfunc.load_model("models:/MyModel@champion")
challenger = mlflow.pyfunc.load_model("models:/MyModel@challenger")
baseline = mlflow.pyfunc.load_model("models:/MyModel@baseline")
```

## Model Stages

MLflow provides predefined stages for model lifecycle:

```python
from mlflow import MlflowClient

client = MlflowClient()

# Available stages: None, Staging, Production, Archived

# Promote to Staging
client.transition_model_version_stage(
    name="MyModel",
    version=5,
    stage="Staging"
)

# Promote to Production
client.transition_model_version_stage(
    name="MyModel",
    version=5,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="MyModel",
    version=3,
    stage="Archived"
)
```

## Model Metadata and Tags

```python
from mlflow import MlflowClient

client = MlflowClient()

# Set model version tags
client.set_model_version_tag(
    name="MyModel",
    version=5,
    key="validation_accuracy",
    value="0.95"
)

client.set_model_version_tag(
    name="MyModel",
    version=5,
    key="training_date",
    value="2024-01-15"
)

# Set registered model tags
client.set_registered_model_tag(
    name="MyModel",
    key="task",
    value="classification"
)

# Update model description
client.update_model_version(
    name="MyModel",
    version=5,
    description="Random Forest classifier trained on updated dataset with improved preprocessing"
)

# Update registered model description
client.update_registered_model(
    name="MyModel",
    description="Production classifier for fraud detection"
)
```

## Search and Query Models

```python
from mlflow import MlflowClient

client = MlflowClient()

# Get all registered models
models = client.search_registered_models()
for model in models:
    print(f"Name: {model.name}")
    print(f"Latest versions: {model.latest_versions}")

# Search with filter
models = client.search_registered_models(
    filter_string="name LIKE 'MyModel%'"
)

# Get specific model version
version_info = client.get_model_version(
    name="MyModel",
    version=5
)

print(f"Stage: {version_info.current_stage}")
print(f"Run ID: {version_info.run_id}")
print(f"Status: {version_info.status}")

# Get model version by alias
version_info = client.get_model_version_by_alias(
    name="MyModel",
    alias="champion"
)

# Get all versions of a model
versions = client.search_model_versions("name='MyModel'")
for version in versions:
    print(f"Version {version.version}: {version.current_stage}")
```

## Delete Models

```python
from mlflow import MlflowClient

client = MlflowClient()

# Delete a specific model version
client.delete_model_version(
    name="MyModel",
    version=1
)

# Delete entire registered model (all versions)
client.delete_registered_model(name="OldModel")
```

## Model Registry Workflow

Complete workflow from training to production:

```python
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# 1. Train and register model
with mlflow.start_run() as run:
    # Train model
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Register model
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="FraudDetector"
    )

    run_id = run.info.run_id

# 2. Get the registered version
model_version = client.search_model_versions(f"run_id='{run_id}'")[0]
version = model_version.version

# 3. Add metadata
client.set_model_version_tag(
    name="FraudDetector",
    version=version,
    key="validation_accuracy",
    value=str(accuracy)
)

client.update_model_version(
    name="FraudDetector",
    version=version,
    description=f"Trained on {datetime.now().date()}, accuracy: {accuracy:.3f}"
)

# 4. Transition to Staging
client.transition_model_version_stage(
    name="FraudDetector",
    version=version,
    stage="Staging"
)

# 5. Test in staging environment
staging_model = mlflow.pyfunc.load_model("models:/FraudDetector/Staging")
test_results = evaluate_model(staging_model, staging_data)

# 6. If tests pass, promote to Production
if test_results["accuracy"] > 0.90:
    client.transition_model_version_stage(
        name="FraudDetector",
        version=version,
        stage="Production"
    )

    # Set as champion
    client.set_registered_model_alias(
        name="FraudDetector",
        alias="champion",
        version=version
    )

# 7. Archive old production model
old_production_versions = client.search_model_versions(
    "name='FraudDetector' AND current_stage='Production'"
)
for old_version in old_production_versions:
    if old_version.version != version:
        client.transition_model_version_stage(
            name="FraudDetector",
            version=old_version.version,
            stage="Archived"
        )
```

## Registry URI Configuration

```python
import mlflow

# Set separate registry URI (optional)
mlflow.set_registry_uri("sqlite:////tmp/registry.db")

# Or use same as tracking
mlflow.set_tracking_uri("http://localhost:5000")
# Registry will use same URI by default

# Get URIs
tracking_uri = mlflow.get_tracking_uri()
registry_uri = mlflow.get_registry_uri()

print(f"Tracking: {tracking_uri}")
print(f"Registry: {registry_uri}")
```

## Model Deployment Patterns

### Pattern 1: Champion/Challenger

```python
from mlflow import MlflowClient

client = MlflowClient()

# Deploy new model as challenger
client.set_registered_model_alias(
    name="MyModel",
    alias="challenger",
    version=10
)

# After validation, promote to champion
client.set_registered_model_alias(
    name="MyModel",
    alias="champion",
    version=10
)

# Load in production
champion = mlflow.pyfunc.load_model("models:/MyModel@champion")
```

### Pattern 2: Blue/Green Deployment

```python
from mlflow import MlflowClient

client = MlflowClient()

# Blue (current production)
client.set_registered_model_alias(
    name="MyModel",
    alias="blue",
    version=8
)

# Green (new version)
client.set_registered_model_alias(
    name="MyModel",
    alias="green",
    version=9
)

# After testing, switch traffic
# Update load balancer to point to green

# Once stable, green becomes new blue
client.set_registered_model_alias(
    name="MyModel",
    alias="blue",
    version=9
)
```

### Pattern 3: Canary Deployment

```python
from mlflow import MlflowClient
import random

client = MlflowClient()

# Stable version
stable = mlflow.pyfunc.load_model("models:/MyModel@champion")

# Canary version
canary = mlflow.pyfunc.load_model("models:/MyModel@challenger")

def predict_with_canary(features, canary_percentage=10):
    """Route canary_percentage% of traffic to new model"""
    if random.random() * 100 < canary_percentage:
        return canary.predict(features)
    else:
        return stable.predict(features)
```

## Troubleshooting Registry

```python
from mlflow import MlflowClient

client = MlflowClient()

# Check registry connection
try:
    models = client.search_registered_models()
    print(f"Found {len(models)} registered models")
except Exception as e:
    print(f"Registry connection error: {e}")

# Verify model exists
try:
    model_info = client.get_registered_model("MyModel")
    print(f"Model exists: {model_info.name}")
except Exception as e:
    print(f"Model not found: {e}")

# Check model version status
version_info = client.get_model_version("MyModel", version=5)
if version_info.status == "READY":
    print("Model version is ready")
else:
    print(f"Model version status: {version_info.status}")
```
