# Shared Templates

This directory contains reusable code templates that can be used across different skills (marimo, mlflow, pina, etc.).

## Available Templates

### 1. `training_loop.py.template`
**Purpose**: Generic training loop structure for ML frameworks

**Features**:
- Customizable training/validation steps
- Checkpoint saving/loading
- Training history tracking
- Device management
- Works with PyTorch, PINA, and other frameworks

**Use Cases**:
- Starting a new training pipeline
- Standardizing training across projects
- Quick prototyping with consistent structure

### 2. `model_definition.py.template`
**Purpose**: Common neural network architecture patterns

**Includes**:
- Feedforward networks (MLP)
- Residual networks (ResNet-style)
- Fourier feature networks (for coordinate-based problems)
- Multi-task networks (shared backbone)
- Hard constraint models (for physics-informed problems)

**Use Cases**:
- Building new models quickly
- Physics-informed neural networks
- Multi-task learning
- Coordinate-based networks

### 3. `configuration.py.template`
**Purpose**: Configuration management patterns

**Features**:
- Dataclass-based configuration (type-safe, recommended)
- Dictionary-based configuration (simple)
- Builder pattern (fluent interface)
- YAML/JSON serialization
- Validation logic
- MLflow integration

**Use Cases**:
- Managing experiment configurations
- Hyperparameter tracking
- Reproducible experiments
- Configuration versioning

### 4. `testing_pattern.py.template`
**Purpose**: Testing patterns for ML models and pipelines

**Includes**:
- Model tests (output shape, gradient flow, parameter count)
- Training tests (overfitting, loss decrease)
- Data pipeline tests (loader, range, NaN checks)
- Physics-informed tests (PDE residual, boundary conditions)
- Pytest integration examples

**Use Cases**:
- Validating model architecture
- Testing training pipelines
- Continuous integration (CI/CD)
- Physics-based validation

## How to Use

### Quick Start

1. **Copy template to your project**:
   ```bash
   cp .claude/Skills/_templates/training_loop.py.template my_project/train.py
   ```

2. **Customize for your use case**:
   - Replace placeholders with your specific implementations
   - Adjust parameters and configurations
   - Add framework-specific logic

3. **Run and iterate**:
   ```bash
   python my_project/train.py
   ```

### Integration with Skills

These templates are referenced in skill documentation:

- **marimo skill**: Uses templates for interactive notebooks with training loops
- **mlflow skill**: Configuration template includes MLflow integration
- **pina skill**: Model templates include physics-informed architectures

### Best Practices

1. **Don't modify templates directly** - copy them first
2. **Keep templates framework-agnostic** where possible
3. **Add type hints** for better IDE support
4. **Include docstrings** explaining customization points
5. **Test templates** work with minimal changes

## Template Patterns

### Modular Design
Each template is self-contained and can be used independently:
```python
# Use just the model
from model_definition import FeedForward
model = FeedForward(10, 1, [64, 64])

# Use just the trainer
from training_loop import GenericTrainer
trainer = GenericTrainer(model, train_loader)

# Use just the config
from configuration import ExperimentConfig
config = ExperimentConfig(...)
```

### Extensibility
Templates are designed to be extended:
```python
class MyCustomTrainer(GenericTrainer):
    def train_step(self, batch):
        # Custom training logic
        pass
```

### Composability
Templates can be combined:
```python
# Combine model + config + trainer
config = ExperimentConfig.from_yaml("config.yaml")
model = FeedForward(**config.model.to_dict())
trainer = GenericTrainer(model, **config.training.to_dict())
```

## Contributing

When adding new templates:

1. **Follow naming convention**: `{purpose}.py.template`
2. **Include comprehensive docstrings**
3. **Add usage examples** in `if __name__ == "__main__"` block
4. **Keep it simple** - templates should be easy to understand
5. **Update this README** with the new template

## Examples

### Example 1: Quick Model + Training
```python
from _templates.model_definition import FeedForward
from _templates.training_loop import GenericTrainer

# Create model
model = FeedForward(input_dim=10, output_dim=1, hidden_dims=[64, 64])

# Create trainer
trainer = GenericTrainer(
    model=model,
    train_loader=my_data_loader,
    optimizer=torch.optim.Adam(model.parameters()),
    loss_fn=torch.nn.MSELoss(),
    max_epochs=100
)

# Train
trainer.train()
```

### Example 2: Configuration-Driven Workflow
```python
from _templates.configuration import ExperimentConfig
from _templates.model_definition import FeedForward
from _templates.training_loop import GenericTrainer

# Load config
config = ExperimentConfig.from_yaml("experiment.yaml")

# Create model from config
model = FeedForward(
    input_dim=config.model.input_dim,
    output_dim=config.model.output_dim,
    hidden_dims=config.model.hidden_dims
)

# Train with config settings
trainer = GenericTrainer(
    model=model,
    train_loader=train_loader,
    max_epochs=config.training.max_epochs,
    device=config.device
)
trainer.train()
```

### Example 3: Testing Workflow
```python
from _templates.testing_pattern import ModelTester, TrainingTester

# Test model
ModelTester.test_output_shape(model, (32, 10), (32, 1))
ModelTester.test_gradient_flow(model, (32, 10))
ModelTester.test_parameter_count(model, max_params=10000)

# Test training
TrainingTester.test_overfitting_small_batch(
    model, loss_fn, optimizer, small_batch, targets
)
```

## Related Resources

- **marimo examples**: `.claude/Skills/marimo/examples/`
- **mlflow examples**: `.claude/Skills/mlflow/examples/`
- **pina examples**: `.claude/Skills/pina/examples/`
- **Skill references**: `.claude/Skills/{skill}/references/`
