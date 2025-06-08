import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# PyTorch Basics - Neural Networks & Tensors""")
    return


@app.cell
def _(mo):
    """PyTorch Configuration"""
    hidden_size = mo.ui.slider(
        start=32, stop=256, step=32, value=128,
        label="🧠 Hidden Layer Size"
    )

    learning_rate = mo.ui.slider(
        start=0.0001, stop=0.01, step=0.0001, value=0.001,
        label="📈 Learning Rate"
    )

    num_epochs = mo.ui.slider(
        start=50, stop=200, step=25, value=100,
        label="🔄 Number of Epochs"
    )

    mo.md(f"""
    ## ⚙️ PyTorch Model Configuration
    {hidden_size}
    {learning_rate}
    {num_epochs}
    """)
    return hidden_size, learning_rate, num_epochs


@app.cell
def _():
    """Setup PyTorch Environment"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import numpy as np
    import mlflow
    import mlflow.pytorch
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name()}")

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # MLflow Setup
    mlflow.set_experiment("pytorch_basics")

    return (
        F,
        StandardScaler,
        device,
        make_classification,
        mlflow,
        nn,
        np,
        optim,
        plt,
        torch,
        train_test_split,
    )


@app.cell
def _(F, nn):
    """Neural Network Definition"""
    class SimpleNN(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            self.batch_norm1 = nn.BatchNorm1d(hidden_size)
            self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)

        def forward(self, x):
            x = F.relu(self.batch_norm1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.batch_norm2(self.fc2(x)))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    print("✅ Neural Network class defined")
    return (SimpleNN,)


@app.cell
def _(
    StandardScaler,
    device,
    make_classification,
    np,
    torch,
    train_test_split,
):
    """Prepare Dataset"""
    # Create dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)

    print(f"✅ Dataset prepared: {X_train_tensor.shape[0]} train, {X_test_tensor.shape[0]} test samples")
    print(f"📊 Features: {X_train_tensor.shape[1]}, Classes: {len(np.unique(y))}")

    return X_test_tensor, X_train_tensor, y_test_tensor, y_train_tensor


@app.cell
def _(
    SimpleNN,
    X_test_tensor,
    X_train_tensor,
    device,
    hidden_size,
    learning_rate,
    mlflow,
    nn,
    num_epochs,
    optim,
    torch,
    y_test_tensor,
    y_train_tensor,
):
    """Train PyTorch Model"""
    with mlflow.start_run(run_name="pytorch_simple_nn") as run:
        # Hyperparameters
        input_size = X_train_tensor.shape[1]
        num_classes = len(torch.unique(y_train_tensor))
        batch_size = 32
        dropout_rate = 0.3

        # MLflow Parameter logging
        mlflow.log_params({
            "input_size": input_size,
            "hidden_size": hidden_size.value,
            "num_classes": int(num_classes),
            "learning_rate": learning_rate.value,
            "num_epochs": num_epochs.value,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "device": str(device)
        })

        # Create model
        model = SimpleNN(input_size, hidden_size.value, num_classes, dropout_rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate.value, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        # Training Loop
        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs.value):
            model.train()

            # Mini-batch Training
            total_loss = 0
            correct = 0
            total = 0

            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            scheduler.step()

            avg_loss = total_loss / (len(X_train_tensor) // batch_size)
            accuracy = 100 * correct / total

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            # MLflow Metrics logging
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_accuracy": accuracy,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy = 100 * (test_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

        # Final Metrics
        mlflow.log_metrics({
            "test_loss": float(test_loss.item()),
            "test_accuracy": float(test_accuracy)
        })

        # Save model
        mlflow.pytorch.log_model(model, "model")

        run_id = run.info.run_id

    print(f"✅ PyTorch Model trained")
    print(f"📈 Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"📊 Final Test Loss: {test_loss.item():.4f}")
    print(f"🔗 MLflow Run ID: {run_id}")

    return test_accuracy, train_accuracies, train_losses


@app.cell
def _(mo, plt, test_accuracy, train_accuracies, train_losses):
    """Training Visualization & Results"""
    # Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss Plot
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy Plot
    ax2.plot(train_accuracies, label='Training Accuracy', color='green')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    mo.md(f"""
    ## 📋 PyTorch Training Results

    ### Model Performance
    - **Final Test Accuracy**: {test_accuracy:.2f}%
    - **Architecture**: Multi-layer Neural Network with Batch Normalization
    - **Features**: Dropout, Learning Rate Scheduling, GPU Support

    ### Training Curves
    The plots above show the training progress over epochs.
    """)

    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
