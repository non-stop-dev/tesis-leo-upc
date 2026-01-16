#!/usr/bin/env python3
"""
GraphSAGE training pipeline for MSME survival prediction.

This script:
1. Loads the HeteroData graph (msme_graph.pt)
2. Trains a heterogeneous GraphSAGE model
3. Uses class weights to handle severe imbalance (96.2% survival rate)
4. Evaluates using F1-Macro (not accuracy)

Key features:
- NeighborLoader for mini-batch training on 1.3M nodes
- Class weights inversely proportional to frequency
- Train/Val/Test split already in graph object

Usage:
    python 3_train_gnn.py --epochs 10 --batch_size 1024

Output:
    - models/graphsage_msme.pt (saved model)
    - output/training_metrics.json (metrics log)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_PATH = os.path.join(BASE_DIR, "database", "msme_graph.pt")
MODEL_OUTPUT = os.path.join(BASE_DIR, "code", "models", "graphsage_msme.pt")
METRICS_OUTPUT = os.path.join(BASE_DIR, "code", "output", "training_metrics.json")


class HomogeneousGNN(torch.nn.Module):
    """
    Base GraphSAGE model that will be converted to heterogeneous.
    
    Architecture: 2-layer GraphSAGE with dropout.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin(x)
        return x


def load_graph():
    """Load the HeteroData graph object."""
    print(f"[Graph] Loading from {GRAPH_PATH}...")
    
    if not os.path.exists(GRAPH_PATH):
        print(f"[Error] Graph not found: {GRAPH_PATH}")
        print("[Tip] Run 1_build_msme_graph.py first")
        sys.exit(1)
    
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"[Graph] Loaded: {data}")
    
    return data


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights inversely proportional to frequency.
    
    From GEMINI.md Section 5.5:
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    """
    class_counts = torch.bincount(labels)
    print(f"[Class Distribution] {dict(enumerate(class_counts.tolist()))}")
    
    # Inverse frequency
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    
    print(f"[Class Weights] {dict(enumerate(class_weights.tolist()))}")
    
    return class_weights


def train_epoch(model, loader, optimizer, class_weights, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_examples = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict)
        
        # Get predictions for MSME nodes only
        msme_out = out['msme'][:batch['msme'].batch_size]
        msme_y = batch['msme'].y[:batch['msme'].batch_size]
        
        # Weighted cross-entropy loss
        loss = F.cross_entropy(msme_out, msme_y, weight=class_weights.to(device))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch['msme'].batch_size
        total_examples += batch['msme'].batch_size
    
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(batch.x_dict, batch.edge_index_dict)
        msme_out = out['msme'][:batch['msme'].batch_size]
        msme_y = batch['msme'].y[:batch['msme'].batch_size]
        
        preds = msme_out.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(msme_y.cpu().tolist())
    
    # Compute metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'f1_macro': float(f1_macro),
        'f1_per_class': [float(f) for f in f1_per_class],
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Train GraphSAGE for MSME survival prediction')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for NeighborLoader')
    parser.add_argument('--hidden', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")
    
    # Load graph
    data = load_graph()
    
    # Compute class weights from training labels
    train_labels = data['msme'].y[data['msme'].train_mask]
    class_weights = compute_class_weights(train_labels)
    
    # Create NeighborLoader for mini-batch training
    # Sample neighbors at each hop to avoid memory explosion
    train_loader = NeighborLoader(
        data,
        num_neighbors={
            ('msme', 'located_in', 'district'): [10, 5],
            ('district', 'rev_located_in', 'msme'): [10, 5],
            ('msme', 'competes_in', 'sector'): [15, 10],
            ('sector', 'rev_competes_in', 'msme'): [15, 10],
        },
        batch_size=args.batch_size,
        input_nodes=('msme', data['msme'].train_mask),
        shuffle=True,
    )
    
    val_loader = NeighborLoader(
        data,
        num_neighbors={
            ('msme', 'located_in', 'district'): [10, 5],
            ('district', 'rev_located_in', 'msme'): [10, 5],
            ('msme', 'competes_in', 'sector'): [15, 10],
            ('sector', 'rev_competes_in', 'msme'): [15, 10],
        },
        batch_size=args.batch_size,
        input_nodes=('msme', data['msme'].val_mask),
        shuffle=False,
    )
    
    # Initialize model
    # Entity nodes (district, sector) don't have features, use embedding
    # MSME nodes have features from the graph
    in_channels = data['msme'].x.shape[1]
    
    # Create homogeneous base model
    base_model = HomogeneousGNN(
        in_channels=-1,  # Infer from data
        hidden_channels=args.hidden,
        out_channels=2,  # Binary classification
        dropout=args.dropout
    )
    
    # Convert to heterogeneous
    model = to_hetero(base_model, data.metadata(), aggr='mean')
    model = model.to(device)
    
    print(f"[Model] {model}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Training loop
    metrics_history = []
    best_f1 = 0.0
    
    print(f"\n[Training] Starting {args.epochs} epochs...")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, class_weights, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
              f"Val F1-Macro: {val_metrics['f1_macro']:.4f} | "
              f"Val Precision: {val_metrics['precision']:.4f} | "
              f"Val Recall: {val_metrics['recall']:.4f}")
        
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            os.makedirs(os.path.dirname(MODEL_OUTPUT), exist_ok=True)
            torch.save(model.state_dict(), MODEL_OUTPUT)
            print(f"  → Saved best model (F1: {best_f1:.4f})")
    
    # Final evaluation on test set
    test_loader = NeighborLoader(
        data,
        num_neighbors={
            ('msme', 'located_in', 'district'): [10, 5],
            ('district', 'rev_located_in', 'msme'): [10, 5],
            ('msme', 'competes_in', 'sector'): [15, 10],
            ('sector', 'rev_competes_in', 'msme'): [15, 10],
        },
        batch_size=args.batch_size,
        input_nodes=('msme', data['msme'].test_mask),
        shuffle=False,
    )
    
    # Load best model for test evaluation
    model.load_state_dict(torch.load(MODEL_OUTPUT, weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    
    print(f"\n[Test Results]")
    print(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  F1 per class: {test_metrics['f1_per_class']}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  Confusion Matrix:")
    for row in test_metrics['confusion_matrix']:
        print(f"    {row}")
    
    # Save metrics
    os.makedirs(os.path.dirname(METRICS_OUTPUT), exist_ok=True)
    final_metrics = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'class_weights': class_weights.tolist(),
        'training_history': metrics_history,
        'test_metrics': test_metrics,
    }
    
    with open(METRICS_OUTPUT, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\n[Saved] Metrics: {METRICS_OUTPUT}")
    print(f"[Saved] Model: {MODEL_OUTPUT}")
    print("[Done] Training complete")


if __name__ == "__main__":
    main()
