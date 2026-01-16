---
name: gnn-quick-reference
description: A "Dictionary" skill for quick access to hyperparameter scales, common transforms, layer comparison, and cheatsheet data. Use for quick lookups during GNN development.
---

# GNN Quick Reference Skill

Fast lookups for common GNN patterns, hyperparameters, and PyG utilities.

## When to use this skill

- Quick lookup of layer parameters or defaults
- Choosing between GNN architectures
- Finding common transforms
- Remembering typical hyperparameter ranges

## GNN Layer Comparison

| Layer | Aggregation | Edge Features | Attention | Best For |
|-------|-------------|---------------|-----------|----------|
| `GCNConv` | Sum + Norm | ❌ | ❌ | Homophilic graphs |
| `SAGEConv` | Mean/Max/LSTM | ❌ | ❌ | Inductive learning |
| `GATConv` | Attention-weighted | ❌ | ✅ | Learning importance |
| `GATv2Conv` | Attention-weighted | ❌ | ✅ | Dynamic attention |
| `GINConv` | Sum + MLP | ❌ | ❌ | Max expressiveness |
| `EdgeConv` | Max | ✅ | ❌ | Point clouds |
| `TransformerConv` | Attention | ✅ | ✅ | Large graphs |

## Common Hyperparameters

### Model Architecture
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Hidden channels | 32-256 | Start with 64-128 |
| Number of layers | 2-4 | More can cause oversmoothing |
| Dropout | 0.3-0.6 | Higher for small datasets |
| GAT heads | 4-8 | More heads = more capacity |

### Training
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Learning rate | 0.001-0.01 | Start with 0.01 |
| Weight decay | 5e-4 to 5e-3 | Regularization |
| Epochs | 100-500 | Use early stopping |
| Batch size (sampling) | 512-2048 | Memory dependent |

### Neighbor Sampling
| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Layer 1 neighbors | 15-25 | More = better but slower |
| Layer 2 neighbors | 10-15 | Can be smaller |
| Layer 3 neighbors | 5-10 | Diminishing returns |

## Common Transforms

```python
import torch_geometric.transforms as T

# Feature preprocessing
T.NormalizeFeatures()      # L1 normalize node features
T.ToUndirected()           # Make graph undirected
T.AddSelfLoops()           # Add self-connections

# Graph augmentation
T.RandomNodeSplit()        # Create train/val/test splits
T.RandomLinkSplit()        # For link prediction
T.FeaturePropagation()     # Propagate features

# Structure modification
T.ToSparseTensor()         # Convert to SparseTensor
T.LocalDegreeProfile()     # Add degree features
T.AddRandomWalkPE(walk_length=20)  # Positional encodings
```

## Quick Code Snippets

### Train/Val/Test Split
```python
n = data.num_nodes
perm = torch.randperm(n)
data.train_mask = torch.zeros(n, dtype=torch.bool)
data.val_mask = torch.zeros(n, dtype=torch.bool)
data.test_mask = torch.zeros(n, dtype=torch.bool)

data.train_mask[perm[:int(0.8*n)]] = True
data.val_mask[perm[int(0.8*n):int(0.9*n)]] = True
data.test_mask[perm[int(0.9*n):]] = True
```

### Class Weights for Imbalanced Data
```python
class_counts = torch.bincount(data.y)
weights = 1.0 / class_counts.float()
weights = weights / weights.sum()
loss = F.cross_entropy(out, data.y, weight=weights)
```

### Early Stopping
```python
best_val_loss = float('inf')
patience = 20
counter = 0

for epoch in range(500):
    # ... training ...
    val_loss = compute_val_loss()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        counter += 1
        if counter >= patience:
            break
```

### Accuracy Computation
```python
@torch.no_grad()
def accuracy(model, data, mask):
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    return (correct / mask.sum()).item()
```

## Graph Statistics

```python
# Basic stats
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Features: {data.num_node_features}")
print(f"Avg degree: {data.num_edges / data.num_nodes:.2f}")

# Check properties
print(f"Is undirected: {data.is_undirected()}")
print(f"Has self-loops: {data.has_self_loops()}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
```

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| OOM Error | Graph too large | Use NeighborLoader |
| NaN loss | Learning rate too high | Reduce LR, add gradient clipping |
| Low accuracy | Oversmoothing | Reduce layers, add dropout |
| Slow training | No GPU/parallelism | Check device, add workers |

## Layer Input/Output Shapes

```
Input: x [N, F_in], edge_index [2, E]
       ↓
Conv1: x [N, hidden]
       ↓  
Conv2: x [N, out_channels]
       ↓
Loss: Compare x[train_mask] with y[train_mask]
```

## References

- `31_gnn_cheatsheet.md` - Full GNN cheatsheet
- `32_dataset_cheatsheet.md` - Dataset information
- `23_torch_geometric_transforms.md` - All transforms
