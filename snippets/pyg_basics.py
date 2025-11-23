import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Graph Neural Networks with Torch Geometric""")
    return


@app.cell
def _():
    """Import PyG libraries"""
    import torch
    import torch.nn.functional as F
    from torch_geometric.datasets import KarateClub
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import to_networkx
    import networkx as nx
    import numpy as np
    import polars as pl
    import altair as alt
    
    return GCNConv, KarateClub, alt, nx, pl, torch, to_networkx, np, F


@app.cell
def _(KarateClub, mo):
    dataset = KarateClub()
    data = dataset[0]
    
    mo.md(f"""
    ## 1. Dataset: Karate Club
    
    - Nodes: {data.num_nodes}
    - Edges: {data.num_edges}
    - Features: {data.num_features}
    - Classes: {dataset.num_classes}
    """)
    return data, dataset


@app.cell
def _(alt, data, nx, pl, to_networkx):
    # Visualize Graph Structure
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare data for Altair
    nodes_df = pl.DataFrame({
        "id": list(G.nodes()),
        "x": [pos[n][0] for n in G.nodes()],
        "y": [pos[n][1] for n in G.nodes()],
        "class": data.y.numpy()
    })
    
    edges_list = []
    for u, v in G.edges():
        edges_list.append({"x": pos[u][0], "y": pos[u][1], "x2": pos[v][0], "y2": pos[v][1]})
        
    edges_df = pl.DataFrame(edges_list)
    
    # Plot edges
    edge_chart = alt.Chart(edges_df.to_pandas()).mark_rule(opacity=0.3).encode(
        x='x', y='y', x2='x2', y2='y2'
    )
    
    # Plot nodes
    node_chart = alt.Chart(nodes_df.to_pandas()).mark_circle(size=200).encode(
        x='x', y='y',
        color=alt.Color('class:N', scale=alt.Scale(scheme='category10')),
        tooltip=['id', 'class']
    )
    
    chart = (edge_chart + node_chart).properties(
        title="Karate Club Graph Structure",
        width=600, height=400
    )
    
    chart.display()
    return G, chart, edge_chart, edges_df, edges_list, node_chart, nodes_df, pos


@app.cell
def _(GCNConv, F, data, torch):
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(dataset.num_features, 16)
            self.conv2 = GCNConv(16, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    return GCN, criterion, model, optimizer


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="🚀 Train GCN")
    mo.md(f"{train_button}")
    return (train_button,)


@app.cell
def _(criterion, data, mo, model, optimizer, train_button):
    mo.stop(not train_button.value, "Click 'Train GCN' to start")
    
    losses = []
    accuracies = []
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            losses.append({"epoch": epoch, "loss": loss.item()})
            
    print("✅ Training complete")
    return accuracies, epoch, loss, losses, out


@app.cell
def _(alt, losses, pl):
    loss_df = pl.DataFrame(losses)
    
    loss_chart = alt.Chart(loss_df.to_pandas()).mark_line().encode(
        x='epoch',
        y='loss',
        tooltip=['epoch', 'loss']
    ).properties(title="Training Loss")
    
    loss_chart.display()
    return loss_chart, loss_df


if __name__ == "__main__":
    app.run()

