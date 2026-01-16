"""
Large-Scale GNN Training with NeighborLoader

This example demonstrates how to train a GNN on large graphs (1M+ nodes)
using neighborhood sampling to avoid memory issues.

Designed for: MSME Survival Prediction on 1.3M node graph
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


class ScalableGNN(torch.nn.Module):
    """
    GraphSAGE-style model designed for mini-batch training.
    
    Uses SAGEConv which is well-suited for inductive learning
    and neighborhood sampling.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def create_neighbor_loader(
    data: Data,
    batch_size: int = 1024,
    num_neighbors: list[int] = [25, 10],
    num_workers: int = 4,
) -> NeighborLoader:
    """
    Create a NeighborLoader for efficient mini-batch training.
    
    Args:
        data: PyG Data object
        batch_size: Number of target nodes per batch
        num_neighbors: Number of neighbors to sample per layer
        num_workers: Parallel data loading workers
    
    Returns:
        Configured NeighborLoader
    """
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
    )
    
    return loader


def train_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch using mini-batches."""
    
    model.train()
    total_loss = 0
    total_examples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass on sampled subgraph
        out = model(batch.x, batch.edge_index)
        
        # IMPORTANT: Only first batch_size nodes are targets
        # The rest are sampled neighbors (context only)
        target_out = out[:batch.batch_size]
        target_y = batch.y[:batch.batch_size]
        
        loss = F.cross_entropy(target_out, target_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size
    
    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Data,
    subgraph_loader: NeighborLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate model using inference-time sampling.
    
    For evaluation, we sample more neighbors for better accuracy.
    """
    model.eval()
    
    # For inference, use all neighbors (or more)
    inference_loader = NeighborLoader(
        data,
        num_neighbors=[-1],  # -1 means all neighbors (no sampling)
        batch_size=4096,
        input_nodes=None,  # All nodes
        shuffle=False,
    )
    
    # Collect predictions for all nodes
    preds = []
    for batch in inference_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)[:batch.batch_size]
        preds.append(out.argmax(dim=1).cpu())
    
    pred = torch.cat(preds, dim=0)
    y = data.y
    
    # Compute accuracies
    train_acc = (pred[data.train_mask] == y[data.train_mask]).float().mean().item()
    val_acc = (pred[data.val_mask] == y[data.val_mask]).float().mean().item()
    test_acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()
    
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
    }


def create_large_synthetic_graph(num_nodes: int = 100_000) -> Data:
    """Create a synthetic large graph for demonstration."""
    
    print(f"Creating synthetic graph with {num_nodes:,} nodes...")
    
    # Random features
    x = torch.randn(num_nodes, 64)
    
    # Random labels (binary)
    y = torch.randint(0, 2, (num_nodes,))
    
    # Create edges: each node connects to ~20 random others
    edges_per_node = 20
    src = torch.arange(num_nodes).repeat_interleave(edges_per_node)
    dst = torch.randint(0, num_nodes, (num_nodes * edges_per_node,))
    edge_index = torch.stack([src, dst])
    
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Train/val/test split
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[perm[:int(0.8 * num_nodes)]] = True
    val_mask[perm[int(0.8 * num_nodes):int(0.9 * num_nodes)]] = True
    test_mask[perm[int(0.9 * num_nodes):]] = True
    
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    
    return data


def main():
    """Main training loop for large-scale GNN."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create or load data
    data = create_large_synthetic_graph(num_nodes=100_000)
    
    # Create model
    model = ScalableGNN(
        in_channels=data.num_features,
        hidden_channels=128,
        out_channels=2,
        num_layers=2,
        dropout=0.5,
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loader
    train_loader = create_neighbor_loader(
        data,
        batch_size=1024,
        num_neighbors=[25, 10],  # Sample 25 neighbors for layer 1, 10 for layer 2
        num_workers=4,
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(1, 21):
        loss = train_epoch(model, train_loader, optimizer, device)
        
        if epoch % 5 == 0:
            metrics = evaluate(model, data, train_loader, device)
            print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | "
                  f"Train: {metrics['train_acc']:.4f} | "
                  f"Val: {metrics['val_acc']:.4f}")
    
    # Final test
    metrics = evaluate(model, data, train_loader, device)
    print(f"\nFinal Test Accuracy: {metrics['test_acc']:.4f}")


if __name__ == "__main__":
    main()
