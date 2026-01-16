"""
GNN Training Template

A complete, copy-paste ready template for training a GNN
for node classification. Includes:
- Model definition
- Training loop with early stopping
- Evaluation metrics
- Logging

Use this as a starting point for any GNN project.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Optional
import time


# ===========================================================
# Model Templates
# ===========================================================

class GCN(torch.nn.Module):
    """2-layer Graph Convolutional Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE(torch.nn.Module):
    """2-layer GraphSAGE."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    """2-layer Graph Attention Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ===========================================================
# Training Utilities
# ===========================================================

def train_epoch(model, data, optimizer, device):
    """Single training epoch."""
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, device):
    """Evaluate model on train/val/test sets."""
    model.eval()
    
    out = model(data.x.to(device), data.edge_index.to(device))
    pred = out.argmax(dim=1)
    y = data.y.to(device)
    
    accs = {}
    for split in ['train', 'val', 'test']:
        mask = getattr(data, f'{split}_mask')
        correct = (pred[mask] == y[mask]).sum()
        accs[split] = (correct / mask.sum()).item()
    
    return accs


def train_model(
    model: torch.nn.Module,
    data: Data,
    epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    patience: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """
    Complete training loop with early stopping.
    
    Args:
        model: GNN model
        data: PyG Data object with train/val/test masks
        epochs: Maximum training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        device: Training device
        verbose: Print progress
    
    Returns:
        Trained model (best validation)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_acc = 0
    best_epoch = 0
    counter = 0
    best_state = None
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, data, optimizer, device)
        accs = evaluate(model, data, device)
        
        # Early stopping check
        if accs['val'] > best_val_acc:
            best_val_acc = accs['val']
            best_epoch = epoch
            counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            counter += 1
        
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                  f"Train: {accs['train']:.4f} | Val: {accs['val']:.4f} | Test: {accs['test']:.4f}")
        
        if counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Final evaluation
    model = model.to(device)
    final_accs = evaluate(model, data, device)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Training completed in {elapsed:.1f}s")
        print(f"Best epoch: {best_epoch}")
        print(f"Final Test Accuracy: {final_accs['test']:.4f}")
    
    return model


# ===========================================================
# Main
# ===========================================================

def main():
    """Example usage with Cora dataset."""
    from torch_geometric.datasets import Planetoid
    
    # Load dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    print(f"Dataset: {dataset}")
    print(f"Nodes: {data.num_nodes}")
    print(f"Features: {data.num_features}")
    print(f"Classes: {dataset.num_classes}")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Train different models
    models = {
        'GCN': GCN(data.num_features, 64, dataset.num_classes),
        'GraphSAGE': GraphSAGE(data.num_features, 64, dataset.num_classes),
        'GAT': GAT(data.num_features, 8, dataset.num_classes),
    }
    
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}")
        print('='*50)
        
        trained_model = train_model(
            model, 
            data,
            epochs=200,
            patience=20,
            device=device,
        )
        
        # Store test accuracy
        final_accs = evaluate(trained_model, data, device)
        results[name] = final_accs['test']
    
    # Summary
    print(f"\n{'='*50}")
    print("Summary")
    print('='*50)
    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")


if __name__ == "__main__":
    main()
