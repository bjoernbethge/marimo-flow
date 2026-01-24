# GenAI Evaluation with MLflow

Complete guide for evaluating GenAI models and LLM applications.

## Built-in Scorers

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines
import pandas as pd

# Create evaluation dataset
eval_data = pd.DataFrame({
    "inputs": {
        "question": ["What is MLflow?", "How do I track experiments?"]
    },
    "expectations": {
        "expected_response": [
            "MLflow is an ML lifecycle platform",
            "Use mlflow.log_* functions"
        ]
    }
})

# Evaluate with built-in scorers
results = mlflow.evaluate(
    model=my_model,
    data=eval_data,
    scorers=[
        Correctness(model="anthropic:/claude-sonnet-4-20250514"),
        RelevanceToQuery(),
        Guidelines(guidelines="Be concise and accurate")
    ]
)

# View results
print(results.metrics)
```

## Available Built-in Metrics

```python
import mlflow

# GenAI-specific metrics
scorers = [
    mlflow.metrics.toxicity(),
    mlflow.metrics.latency(),
    mlflow.metrics.flesch_kincaid_grade_level(),
]

# Use in evaluation
results = mlflow.evaluate(
    model=my_model,
    data=eval_data,
    extra_metrics=scorers
)
```

## Custom Scorers

```python
from mlflow.genai.scorers import scorer

@scorer
def response_length_checker(outputs) -> bool:
    """Check if response is not too long"""
    return len(outputs) <= 500

# Register for automated evaluation
registered = response_length_checker.register(
    name="length_checker",
    experiment_id="12345"
)

# Use in evaluation
results = mlflow.evaluate(
    model=my_model,
    data=eval_data,
    scorers=[response_length_checker]
)
```

## Advanced Custom Scorers

```python
from mlflow.genai.scorers import scorer
import anthropic

@scorer
def domain_accuracy_scorer(outputs, expectations):
    """Custom domain-specific accuracy checker"""
    client = anthropic.Anthropic()

    prompt = f"""
    Compare the output with expected response for domain accuracy.

    Output: {outputs}
    Expected: {expectations}

    Rate accuracy from 0-10.
    """

    response = client.messages.create(
        model="claude-sonnet-4",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse score from response
    score = int(response.content[0].text.strip())
    return score / 10.0

# Use in evaluation
results = mlflow.evaluate(
    model=my_model,
    data=eval_data,
    scorers=[domain_accuracy_scorer]
)
```

## Compare Model Versions

```python
import mlflow
import pandas as pd

eval_data = pd.DataFrame({
    "inputs": test_questions,
    "expected_categories": ["account", "billing", "support"]
})

# Evaluate version 1
results_v1 = mlflow.evaluate(
    model_uri="models:/MyModel@v1",
    data=eval_data,
    extra_metrics=[
        mlflow.metrics.toxicity(),
        mlflow.metrics.latency(),
        mlflow.metrics.flesch_kincaid_grade_level()
    ]
)

# Evaluate version 2
results_v2 = mlflow.evaluate(
    model_uri="models:/MyModel@v2",
    data=eval_data,
    extra_metrics=[
        mlflow.metrics.toxicity(),
        mlflow.metrics.latency(),
        mlflow.metrics.flesch_kincaid_grade_level()
    ]
)

# Compare in UI or programmatically
print(f"V1 Toxicity: {results_v1.metrics['toxicity/v1/mean']}")
print(f"V2 Toxicity: {results_v2.metrics['toxicity/v1/mean']}")
```

## Batch Evaluation

```python
import mlflow
import pandas as pd

# Large evaluation dataset
eval_data = pd.DataFrame({
    "inputs": list_of_1000_questions,
    "expected_categories": list_of_1000_categories
})

# Evaluate with multiple scorers
results = mlflow.evaluate(
    model=model,
    data=eval_data,
    scorers=[
        Correctness(model="anthropic:/claude-sonnet-4"),
        RelevanceToQuery(),
        mlflow.metrics.toxicity(),
        mlflow.metrics.latency(),
        custom_scorer
    ]
)

# Analyze results
print(f"Mean correctness: {results.metrics['correctness/mean']}")
print(f"Mean latency: {results.metrics['latency/mean']}")
print(f"Toxicity rate: {results.metrics['toxicity/rate']}")
```

## Evaluation Best Practices

```python
# Use multiple scorers for comprehensive evaluation
scorers = [
    Correctness(model="anthropic:/claude-sonnet-4"),
    RelevanceToQuery(),
    mlflow.metrics.toxicity(),
    mlflow.metrics.latency(),
    custom_domain_scorer
]

# Run regular evaluations
results = mlflow.evaluate(
    model=model,
    data=test_set,
    scorers=scorers
)

# Log evaluation results
with mlflow.start_run():
    for metric_name, metric_value in results.metrics.items():
        mlflow.log_metric(f"eval_{metric_name}", metric_value)
```

## A/B Testing Models

```python
import mlflow
import pandas as pd

# Evaluation data
eval_data = pd.DataFrame({
    "inputs": test_inputs,
    "expectations": expected_outputs
})

# Evaluate model A
with mlflow.start_run(run_name="model_a_eval"):
    results_a = mlflow.evaluate(
        model="models:/MyModel@champion",
        data=eval_data,
        scorers=[Correctness(), RelevanceToQuery()]
    )
    mlflow.log_metrics({
        "correctness": results_a.metrics["correctness/mean"],
        "relevance": results_a.metrics["relevance/mean"]
    })

# Evaluate model B
with mlflow.start_run(run_name="model_b_eval"):
    results_b = mlflow.evaluate(
        model="models:/MyModel@challenger",
        data=eval_data,
        scorers=[Correctness(), RelevanceToQuery()]
    )
    mlflow.log_metrics({
        "correctness": results_b.metrics["correctness/mean"],
        "relevance": results_b.metrics["relevance/mean"]
    })

# Compare in MLflow UI
```

## Continuous Evaluation

```python
import mlflow
from datetime import datetime

def continuous_evaluation_pipeline():
    """Run evaluation regularly on production model"""

    # Get production model
    model = mlflow.pyfunc.load_model("models:/MyModel@champion")

    # Get fresh test data
    eval_data = fetch_latest_test_data()

    # Run evaluation
    with mlflow.start_run(run_name=f"eval_{datetime.now().isoformat()}"):
        results = mlflow.evaluate(
            model=model,
            data=eval_data,
            scorers=[
                Correctness(),
                RelevanceToQuery(),
                mlflow.metrics.toxicity(),
                mlflow.metrics.latency()
            ]
        )

        # Log results with timestamp
        mlflow.log_metrics({
            "correctness": results.metrics["correctness/mean"],
            "toxicity_rate": results.metrics["toxicity/rate"],
            "avg_latency": results.metrics["latency/mean"]
        })

        # Alert if metrics degrade
        if results.metrics["correctness/mean"] < 0.85:
            send_alert("Model quality degradation detected")

# Schedule to run daily
```
