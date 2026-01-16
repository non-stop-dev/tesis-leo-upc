---
name: large-scale-graph-sampling
description: Critical for large datasets (1M+ nodes). Handles training when the graph doesn't fit in memory using Neighbor Sampling, mini-batching, and hierarchical sampling. Use when working with datasets like the 1.3M MSME census.
---

# Large-Scale Graph Sampling Skill

This skill provides expertise for training GNNs on graphs too large to fit in GPU memory, using neighborhood sampling and efficient data loading.

## When to use this skill

- Your graph has >100K nodes
- Full-batch training causes OOM (Out of Memory) errors
- You need to train on a subset of neighbors per layer
- You want to use mini-batch training for large graphs

## Core Concept: Neighborhood Sampling

Instead of using all neighbors, sample a fixed number per layer:

```
Layer 2: Sample 10 neighbors per node
    ↑
Layer 1: Sample 25 neighbors per node
    ↑
Target nodes (batch of seeds)
```

This creates a **subgraph** for each mini-batch.

## The NeighborLoader

PyG's `NeighborLoader` handles sampling automatically:

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 25 for layer 1, 10 for layer 2
    batch_size=1024,         # Number of target nodes per batch
    input_nodes=data.train_mask,  # Which nodes to sample from
    shuffle=True,
)

for batch in loader:
    # batch is a subgraph containing sampled nodes
    out = model(batch.x, batch.edge_index)
    loss = loss_fn(out[:batch.batch_size], batch.y[:batch.batch_size])
    # Note: only first batch_size nodes are targets
```

## Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `num_neighbors` | Samples per layer (list) | `[25, 10]` or `[15, 10, 5]` |
| `batch_size` | Target nodes per batch | `512-2048` |
| `input_nodes` | Mask for seed nodes | `train_mask` |
| `num_workers` | Parallel data loading | `4-8` |
| `pin_memory` | Faster GPU transfer | `True` |

## Memory Estimation

For a batch:
- Target nodes: `batch_size`
- Layer 1: up to `batch_size × num_neighbors[0]`
- Layer 2: up to `batch_size × num_neighbors[0] × num_neighbors[1]`

Example: `batch_size=1024, num_neighbors=[25, 10]`
- Max sampled nodes ≈ 1024 × 25 × 10 = 256K nodes

## Heterogeneous NeighborLoader

For `HeteroData`:

```python
from torch_geometric.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors={
        ('empresa', 'ubicado_en', 'distrito'): [10, 5],
        ('empresa', 'compite_con', 'empresa'): [15, 10],
    },
    batch_size=512,
    input_nodes=('empresa', data['empresa'].train_mask),
)
```

## Training Loop with Sampling

```python
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index)
        
        # Only compute loss on target nodes (first batch_size)
        loss = F.cross_entropy(
            out[:batch.batch_size],
            batch.y[:batch.batch_size]
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")
```

## Performance Tips

1. **Use multiple workers**: `num_workers=4` or more
2. **Pin memory**: `pin_memory=True` for GPU training
3. **Prefetch batches**: PyG handles this automatically
4. **Reduce neighbors for deeper networks**: Use fewer samples in later layers
5. **Profile first**: Use PyG's profiling tools to find bottlenecks

## Alternative: ClusterGCN

For very large graphs, consider `ClusterLoader`:

```python
from torch_geometric.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=100)
loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True)
```

## Decision Tree

```
Is your graph too large for GPU memory?
├── NO → Use full-batch training
└── YES → How large?
    ├── <10M edges → Use NeighborLoader with moderate sampling
    ├── <100M edges → Use NeighborLoader with aggressive sampling
    └── >100M edges → Consider ClusterGCN or distributed training
```

## Link Prediction with Sampling

Use `LinkNeighborLoader` for link prediction tasks:

```python
from torch_geometric.loader import LinkNeighborLoader

loader = LinkNeighborLoader(
    data,
    num_neighbors=[25, 10],
    edge_label_index=train_edge_index,  # Edges to predict
    edge_label=train_edge_label,        # Labels (0/1)
    batch_size=256,
)
```

## Advanced Options

- `disjoint=True`: Disable node fusion across seed nodes (use more memory but cleaner subgraphs)
- `subgraph_type="bidirectional"`: Convert directed samples to bidirectional
- `subgraph_type="induced"`: Return induced subgraph of all sampled nodes

## References

- `06c_scaling_gnns_neighbor_sampling.md` - Complete neighbor sampling tutorial
- `05b_neighbor_sampling.md` - NeighborLoader basics
- `08_advanced_mini_batching.md` - Batching deep dive
- `10_hierarchical_neighborhood_sampling.md` - HGAM sampling
- `19_torch_geometric_loader.md` - All loader APIs
