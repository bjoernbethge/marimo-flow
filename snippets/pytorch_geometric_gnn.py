import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# PyTorch Geometric - Graph Neural Networks""")
    return


@app.cell
def _(mo):
    """GNN Configuration"""
    model_type = mo.ui.dropdown(
        options=["GCN", "GAT", "GraphSAGE"],
        value="GCN",
        label="🧠 GNN Model Type"
    )
    
    hidden_dim = mo.ui.slider(
        start=16, stop=128, step=16, value=64,
        label="🔗 Hidden Dimensions"
    )
    
    num_epochs = mo.ui.slider(
        start=50, stop=300, step=50, value=200,
        label="🔄 Training Epochs"
    )
    
    mo.md(f"""
    ## ⚙️ Graph Neural Network Configuration
    {model_type}
    {hidden_dim}
    {num_epochs}
    """)
    return model_type, hidden_dim, num_epochs


@app.cell
def _(hidden_dim, model_type, num_epochs):
    """Setup PyTorch Geometric Environment"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    import numpy as np
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.pytorch
    from sklearn.metrics import accuracy_score
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Geometric Version: {torch_geometric.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # MLflow Setup
    mlflow.set_experiment("pytorch_geometric_gnn")
    
    return (
        Data, F, GATConv, GCNConv, NormalizeFeatures, Planetoid, SAGEConv,
        accuracy_score, device, mlflow, nn, np, plt, torch, torch_geometric
    )


@app.cell
def _(NormalizeFeatures, Planetoid, device):
    """Load Graph Dataset"""
    # Load dataset - Cora Citation Network
    dataset = Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0].to(device)
    
    print(f"✅ Dataset loaded: {dataset}")
    print(f"📊 Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"🎯 Features: {dataset.num_features}, Classes: {dataset.num_classes}")
    print(f"📈 Average node degree: {data.num_edges / data.num_nodes:.2f}")
    
    return data, dataset


@app.cell
def _(F, GATConv, GCNConv, SAGEConv, nn):
    """Define GNN Models"""
    class GCN(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, num_classes)
            self.dropout = dropout
            
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    class GAT(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, heads=8, dropout=0.6):
            super(GAT, self).__init__()
            self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout)
            self.dropout = dropout
            
        def forward(self, x, edge_index):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    class GraphSAGE(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, dropout=0.5):
            super(GraphSAGE, self).__init__()
            self.conv1 = SAGEConv(num_features, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, num_classes)
            self.dropout = dropout
            
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv3(x, edge_index)
            return F.log_softmax(x, dim=1)
    
    print("✅ GNN model classes defined")
    return GAT, GCN, GraphSAGE


@app.cell
def _(accuracy_score):
    """Training and Testing Functions"""
    def train_model(model, data, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def test_model(model, data):
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            
            train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
            test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
            
        return train_acc, val_acc, test_acc
    
    print("✅ Training functions defined")
    return test_model, train_model


@app.cell
def _(
    GAT, GCN, GraphSAGE, data, dataset, device, hidden_dim, 
    mlflow, model_type, nn, num_epochs, test_model, torch, train_model
):
    """Train Selected GNN Model"""
    with mlflow.start_run(run_name=f"pyg_{model_type.value.lower()}") as run:
        # Create model
        if model_type.value == 'GAT':
            model = GAT(
                dataset.num_features, 
                hidden_dim.value // 8,  # Smaller hidden dim for GAT due to multi-head
                dataset.num_classes,
                heads=8,
                dropout=0.6
            ).to(device)
        elif model_type.value == 'GraphSAGE':
            model = GraphSAGE(
                dataset.num_features, 
                hidden_dim.value, 
                dataset.num_classes,
                dropout=0.5
            ).to(device)
        else:  # GCN
            model = GCN(
                dataset.num_features, 
                hidden_dim.value, 
                dataset.num_classes,
                dropout=0.5
            ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = nn.NLLLoss()
        
        # Hyperparameter logging
        mlflow.log_params({
            "model_type": model_type.value,
            "hidden_dim": hidden_dim.value,
            "learning_rate": 0.01,
            "weight_decay": 5e-4,
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "num_epochs": num_epochs.value
        })
        
        # Training
        train_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(num_epochs.value):
            loss = train_model(model, data, optimizer, criterion)
            train_acc, val_acc, test_acc = test_model(model, data)
            
            train_losses.append(loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # MLflow Metrics
            mlflow.log_metrics({
                "train_loss": loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc
            }, step=epoch)
        
        # Final Test
        final_train_acc, final_val_acc, final_test_acc = test_model(model, data)
        
        # Final Metrics
        mlflow.log_metrics({
            "final_train_accuracy": final_train_acc,
            "final_val_accuracy": final_val_acc,
            "final_test_accuracy": final_test_acc
        })
        
        # Save model
        mlflow.pytorch.log_model(model, f"model_{model_type.value.lower()}")
        
        run_id = run.info.run_id
    
    print(f"✅ {model_type.value} model trained")
    print(f"📈 Final Test Accuracy: {final_test_acc:.4f}")
    print(f"🔗 MLflow Run ID: {run_id}")
    
    return (
        criterion, epoch, final_test_acc, final_train_acc, final_val_acc,
        loss, model, optimizer, run_id, test_acc, train_acc, train_accs,
        train_losses, val_acc, val_accs
    )


@app.cell
def _(final_test_acc, mo, model_type, plt, train_accs, train_losses, val_accs):
    """Training Results Visualization"""
    # Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss Curve
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy Curves
    ax2.plot(train_accs, label='Training Accuracy', color='green')
    ax2.plot(val_accs, label='Validation Accuracy', color='orange')
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    mo.md(f"""
    ## 📋 {model_type.value} Training Results
    
    ### Model Performance
    - **Final Test Accuracy**: {final_test_acc:.4f}
    - **Model Type**: {model_type.value}
    - **Dataset**: Cora Citation Network
    
    ### Training Curves
    The plots above show the training progress over epochs.
    
    ### Graph Neural Network Features
    - **Node Classification** on citation network
    - **Message Passing** between connected nodes
    - **Inductive Learning** capability
    """)
    
    return ax1, ax2, fig


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run() 