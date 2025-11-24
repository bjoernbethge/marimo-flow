# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4.0",
#     "marimo",
#     "mlflow>=2.17.0",
#     "numpy>=1.26.4",
#     "polars>=1.12.0",
#     "scikit-learn>=1.5.0",
#     "torch>=2.5.0",
#     "torch-geometric>=2.6.0",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports and Setup"""
    import altair as alt
    import marimo as mo
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import polars as pl
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_geometric
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, Linear, SAGEConv, TransformerConv
    from torch_geometric.utils import negative_sampling

    mo.md(f"""
    # 🧠 Advanced GNN Demo
    
    **Heterogeneous Graph Neural Networks with PyTorch Geometric**
    
    - **PyTorch**: {torch.__version__}
    - **PyG**: {torch_geometric.__version__}
    - **Device**: {"CUDA" if torch.cuda.is_available() else "CPU"}
    """)

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLflow Setup
    mlflow.set_experiment("pytorch_geometric_advanced")

    return (
        F,
        HeteroConv,
        HeteroData,
        Linear,
        SAGEConv,
        TransformerConv,
        alt,
        average_precision_score,
        device,
        mlflow,
        mo,
        negative_sampling,
        nn,
        np,
        pl,
        roc_auc_score,
        torch,
        torch_geometric,
    )


@app.cell
def _(HeteroData, device, torch):
    """Create Heterogeneous Graph Dataset"""
    def create_movie_hetero_graph():
        """Creates a heterogeneous graph for Movie-Actor-Director relationships"""

        # Node Features
        num_movies = 100
        num_actors = 200
        num_directors = 50

        # Movie Features (Genre, Year, Rating)
        movie_features = torch.randn(num_movies, 16)

        # Actor Features (Age, Experience, Awards)
        actor_features = torch.randn(num_actors, 12)

        # Director Features (Experience, Awards, Style)
        director_features = torch.randn(num_directors, 8)

        # Edge Indices
        # Movie-Actor (acts_in)
        movie_actor_edges = torch.randint(0, num_movies, (2, 400))
        movie_actor_edges[1] = torch.randint(0, num_actors, (400,))

        # Movie-Director (directed_by)
        movie_director_edges = torch.randint(0, num_movies, (2, 100))
        movie_director_edges[1] = torch.randint(0, num_directors, (100,))

        # Actor-Actor (collaborated)
        actor_actor_edges = torch.randint(0, num_actors, (2, 300))

        # Create HeteroData
        hetero_data = HeteroData()

        # Node Features
        hetero_data['movie'].x = movie_features
        hetero_data['actor'].x = actor_features
        hetero_data['director'].x = director_features

        # Node Labels (for Classification)
        hetero_data['movie'].y = torch.randint(0, 5, (num_movies,))  # 5 Genres
        hetero_data['actor'].y = torch.randint(0, 3, (num_actors,))  # 3 Career Stages

        # Edge Indices
        hetero_data['movie', 'acts_in', 'actor'].edge_index = movie_actor_edges
        hetero_data['actor', 'acts_in', 'movie'].edge_index = movie_actor_edges.flip(0)

        hetero_data['movie', 'directed_by', 'director'].edge_index = movie_director_edges
        hetero_data['director', 'directed_by', 'movie'].edge_index = movie_director_edges.flip(0)

        hetero_data['actor', 'collaborated', 'actor'].edge_index = actor_actor_edges

        return hetero_data

    hetero_data = create_movie_hetero_graph().to(device)

    print("✅ Heterogeneous Graph created")
    print(f"📊 Node types: {hetero_data.node_types}")
    print(f"🔗 Edge types: {hetero_data.edge_types}")

    for node_type in hetero_data.node_types:
        print(f"  {node_type}: {hetero_data[node_type].num_nodes} nodes")

    return (hetero_data,)


@app.cell
def _(F, HeteroConv, Linear, SAGEConv, TransformerConv, nn, torch):
    """Define Advanced GNN Models"""
    class HeteroGNN(nn.Module):
        def __init__(self, hidden_dim=64, num_layers=2):
            super().__init__()

            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                conv = HeteroConv({
                    ('movie', 'acts_in', 'actor'): SAGEConv((-1, -1), hidden_dim),
                    ('actor', 'acts_in', 'movie'): SAGEConv((-1, -1), hidden_dim),
                    ('movie', 'directed_by', 'director'): SAGEConv((-1, -1), hidden_dim),
                    ('director', 'directed_by', 'movie'): SAGEConv((-1, -1), hidden_dim),
                    ('actor', 'collaborated', 'actor'): SAGEConv((-1, -1), hidden_dim),
                }, aggr='sum')
                self.convs.append(conv)

            # Node-type specific classifiers
            self.movie_classifier = Linear(hidden_dim, 5)  # 5 genres
            self.actor_classifier = Linear(hidden_dim, 3)  # 3 career stages

        def forward(self, x_dict, edge_index_dict):
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}

            return x_dict

    class GraphTransformer(nn.Module):
        def __init__(self, num_features, hidden_dim, num_classes, num_layers=3, heads=8, dropout=0.1):
            super().__init__()

            self.num_layers = num_layers
            self.dropout = dropout

            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()

            # Input projection
            self.input_proj = Linear(num_features, hidden_dim)

            # Transformer layers
            for _ in range(num_layers):
                self.convs.append(TransformerConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout))
                self.norms.append(nn.LayerNorm(hidden_dim))

            # Output projection
            self.output_proj = Linear(hidden_dim, num_classes)

        def forward(self, x, edge_index):
            x = self.input_proj(x)

            for conv, norm in zip(self.convs, self.norms):
                # Residual connection
                residual = x
                x = conv(x, edge_index)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = norm(x + residual)
                x = F.relu(x)

            x = self.output_proj(x)
            return F.log_softmax(x, dim=1)

    class LinkPredictor(nn.Module):
        def __init__(self, num_features, hidden_dim, num_layers=2, dropout=0.5):
            super().__init__()

            self.convs = nn.ModuleList()
            self.convs.append(SAGEConv(num_features, hidden_dim))

            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))

            self.dropout = dropout

            # Link prediction head
            self.link_predictor = nn.Sequential(
                Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                Linear(hidden_dim, 1)
            )

        def forward(self, x, edge_index):
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        def predict_links(self, x, edge_index):
            # Get node embeddings
            node_emb = self.forward(x, edge_index)

            # Concatenate source and target embeddings
            src_emb = node_emb[edge_index[0]]
            dst_emb = node_emb[edge_index[1]]
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)

            # Predict link probability
            return torch.sigmoid(self.link_predictor(edge_emb))

    print("✅ Advanced GNN model classes defined")
    return GraphTransformer, HeteroGNN, LinkPredictor


@app.cell
def _(
    GraphTransformer,
    HeteroGNN,
    LinkPredictor,
    device,
    gnn_type,
    hetero_data,
    hidden_dim,
    mlflow,
    negative_sampling,
    nn,
    num_layers,
    roc_auc_score,
    torch,
):
    """Train Selected Advanced GNN Model"""
    with mlflow.start_run(run_name=f"advanced_{gnn_type.value.lower()}") as run:

        if gnn_type.value == "HeteroGNN":
            # Heterogeneous GNN Training
            model = HeteroGNN(hidden_dim=hidden_dim.value, num_layers=num_layers.value).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()

            # Parameters
            mlflow.log_params({
                "model_type": "HeteroGNN",
                "hidden_dim": hidden_dim.value,
                "num_layers": num_layers.value,
                "learning_rate": 0.01,
                "target_node_type": "movie"
            })

            # Training
            model.train()
            train_losses = []

            for epoch in range(100):
                optimizer.zero_grad()

                # Forward pass
                out_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)

                # Classification loss for movie nodes
                logits = model.movie_classifier(out_dict['movie'])
                loss = criterion(logits, hetero_data['movie'].y)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # Final accuracy
            model.eval()
            with torch.no_grad():
                final_out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
                final_logits = model.movie_classifier(final_out['movie'])
                final_pred = final_logits.argmax(dim=1)
                final_acc = (final_pred == hetero_data['movie'].y).float().mean()

            mlflow.log_metric("final_accuracy", float(final_acc))
            result_metric = f"Final Accuracy: {final_acc:.4f}"

        elif gnn_type.value == "LinkPredictor":
            # Create simple graph for link prediction
            num_nodes = 1000
            x = torch.randn(num_nodes, 64).to(device)
            edge_index = torch.randint(0, num_nodes, (2, 2000)).to(device)

            model = LinkPredictor(64, hidden_dim.value, num_layers.value).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.BCELoss()

            mlflow.log_params({
                "model_type": "LinkPredictor",
                "hidden_dim": hidden_dim.value,
                "num_layers": num_layers.value,
                "learning_rate": 0.01,
                "num_nodes": num_nodes
            })

            # Training
            model.train()
            train_losses = []

            for epoch in range(50):
                optimizer.zero_grad()

                # Positive edges
                pos_pred = model.predict_links(x, edge_index)
                pos_labels = torch.ones(pos_pred.size(0), 1).to(device)

                # Negative edges
                neg_edges = negative_sampling(
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                    num_neg_samples=edge_index.size(1)
                ).to(device)

                neg_pred = model.predict_links(x, neg_edges)
                neg_labels = torch.zeros(neg_pred.size(0), 1).to(device)

                # Combined loss
                all_pred = torch.cat([pos_pred, neg_pred], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                loss = criterion(all_pred, all_labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # Final AUC
            model.eval()
            with torch.no_grad():
                test_pos_pred = model.predict_links(x, edge_index[:, :500])
                test_neg_pred = model.predict_links(x, neg_edges[:, :500])

                test_pred = torch.cat([test_pos_pred, test_neg_pred], dim=0).cpu().numpy()
                test_labels = torch.cat([
                    torch.ones(test_pos_pred.size(0)),
                    torch.zeros(test_neg_pred.size(0))
                ], dim=0).cpu().numpy()

                auc = roc_auc_score(test_labels, test_pred)

            mlflow.log_metric("final_auc", auc)
            result_metric = f"Final AUC: {auc:.4f}"

        else:  # GraphTransformer
            # Simple graph for transformer
            num_nodes = 500
            x = torch.randn(num_nodes, 32).to(device)
            edge_index = torch.randint(0, num_nodes, (2, 1000)).to(device)
            y = torch.randint(0, 5, (num_nodes,)).to(device)

            model = GraphTransformer(32, hidden_dim.value, 5, num_layers.value).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.NLLLoss()

            mlflow.log_params({
                "model_type": "GraphTransformer",
                "hidden_dim": hidden_dim.value,
                "num_layers": num_layers.value,
                "learning_rate": 0.01,
                "num_nodes": num_nodes
            })

            # Training
            model.train()
            train_losses = []

            for epoch in range(100):
                optimizer.zero_grad()

                out = model(x, edge_index)
                loss = criterion(out, y)

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

            # Final accuracy
            model.eval()
            with torch.no_grad():
                final_out = model(x, edge_index)
                final_pred = final_out.argmax(dim=1)
                final_acc = (final_pred == y).float().mean()

            mlflow.log_metric("final_accuracy", float(final_acc))
            result_metric = f"Final Accuracy: {final_acc:.4f}"

        # Save model
        mlflow.pytorch.log_model(model, f"model_{gnn_type.value.lower()}")

        run_id = run.info.run_id

    print(f"✅ {gnn_type.value} model trained")
    print(f"📈 {result_metric}")
    print(f"🔗 MLflow Run ID: {run_id}")

    return result_metric, train_losses


@app.cell(hide_code=True)
def _(gnn_type, mo, alt, result_metric, train_losses):
    """Advanced GNN Results Visualization"""
    # Training Loss Plot
    df_losses = pl.from_records([{"epoch": i, "loss": l} for i, l in enumerate(train_losses)])
    loss_chart = alt.Chart(df_losses).mark_line().encode(
        x="epoch",
        y="loss",
        color="Model Type:N",
        title=f'{gnn_type.value} Training Loss'
    ).properties(width=600, height=400)

    mo.md(f"""
    ## 📋 {gnn_type.value} Training Results

    ### Model Performance
    - **{result_metric}**
    - **Model Type**: {gnn_type.value}
    - **Advanced Features**: Heterogeneous graphs, attention mechanisms, link prediction

    ### Training Progress
    The plot above shows the training loss over epochs.

    ### Advanced GNN Capabilities
    - **Heterogeneous GNNs**: Multi-type nodes and edges
    - **Graph Transformers**: Self-attention for graphs
    - **Link Prediction**: Edge prediction with negative sampling
    - **Scalable Training**: Mini-batch and sampling strategies
    """)

    return


if __name__ == "__main__":
    app.run()
