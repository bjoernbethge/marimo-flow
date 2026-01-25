# Framework Integrations

Complete guide for integrating MLflow with popular GenAI and ML frameworks.

## LangChain

```python
import mlflow
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

# Enable autologging
mlflow.langchain.autolog()

# Set active model for linking traces
mlflow.set_active_model(name="langchain_model")

# Build chain
prompt = ChatPromptTemplate.from_template("Answer: {question}")
chain = prompt | ChatOpenAI(temperature=0.7) | StrOutputParser()

# Use chain - automatically traced
response = chain.invoke({"question": "What is MLflow?"})

# Log chain as model
mlflow.langchain.log_model(
    lc_model=chain,
    name="qa_chain",
    params={"temperature": 0.7},
    model_type="agent",
    registered_model_name="QAChain"
)
```

## LlamaIndex

```python
import mlflow

# Enable autologging
mlflow.llama_index.autolog()

# Use LlamaIndex as usual
chat_engine = index.as_chat_engine()
response = chat_engine.chat("What is MLflow?")

# Log LlamaIndex model
mlflow.llama_index.log_model(
    llama_index_model="workflow/model.py",
    model_config={"retrievers": ["vector_search", "bm25"]},
    code_paths=["workflow"],
    name="model"
)
```

## DSPy

```python
import mlflow

# Enable autologging for DSPy
mlflow.dspy.autolog()

# Use DSPy modules - automatically traced
# All DSPy interactions logged automatically
```

## CrewAI

```python
import mlflow

# Enable autologging for CrewAI
mlflow.crewai.autolog()

# Use CrewAI agents - automatically traced
# All agent interactions and tool calls logged
```

## Scikit-learn

```python
import mlflow

# Scikit-learn autologging
mlflow.sklearn.autolog()

# Train as usual - everything logged automatically
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Autologging captures:
# - Model parameters
# - Training metrics
# - Model artifacts
# - Dependencies
# - Signatures and input examples
```

## PyTorch

```python
import mlflow

# PyTorch autologging
mlflow.pytorch.autolog()

# Train model
model = MyNeuralNet()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    train_loss = train_epoch(model, optimizer, train_loader)
    # Metrics automatically logged
```

## TensorFlow/Keras

```python
import mlflow

# TensorFlow/Keras autologging
mlflow.tensorflow.autolog()

# Build and train model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X_train, y_train, epochs=10)
# Everything logged automatically
```

## XGBoost

```python
import mlflow

# XGBoost autologging
mlflow.xgboost.autolog()

# Train XGBoost model
dtrain = xgb.DMatrix(X_train, label=y_train)
params = {'max_depth': 3, 'eta': 0.1}
model = xgb.train(params, dtrain, num_boost_round=100)
# Parameters and metrics automatically logged
```

## OpenAI

```python
import mlflow
from openai import OpenAI

# Enable OpenAI autologging
mlflow.openai.autolog()

client = OpenAI()

# All API calls automatically traced
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
# Tokens, latency, cost automatically logged
```

## Anthropic

```python
import mlflow
from anthropic import Anthropic

# Enable Anthropic autologging
mlflow.anthropic.autolog()

client = Anthropic()

# All API calls automatically traced
response = client.messages.create(
    model="claude-sonnet-4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}]
)
# Tokens, latency, cost automatically logged
```

## LiteLLM (Multi-Provider)

```python
import mlflow
import litellm

# Enable LiteLLM autologging
mlflow.litellm.autolog()

# Works with any provider
response = litellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Also works with Claude
response = litellm.completion(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# All providers automatically traced
```

## Debugging Autologging

```python
import mlflow

# Disable autologging if conflicts arise
mlflow.sklearn.autolog(disable=True)

# Re-enable with specific options
mlflow.sklearn.autolog(
    log_models=False,  # Don't log models
    log_input_examples=True,
    log_model_signatures=True
)
```
