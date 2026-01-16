---
name: performance-optimization
description: Utilizing torch.compile, CPU Affinity, and memory-efficient aggregations to speed up GNN training and inference. Use when you need to optimize GNN performance, reduce memory usage, or profile bottlenecks.
---

# Performance Optimization Skill

This skill provides expertise for optimizing GNN training and inference performance using PyTorch and PyG's built-in tools.

## When to use this skill

- Training is slower than expected
- Running out of GPU/CPU memory
- Need to deploy models efficiently
- Want to leverage `torch.compile()` with GNNs
- Profiling to find bottlenecks

## torch.compile() for GNNs

PyTorch 2.0's compiler can significantly speed up GNN training:

```python
import torch

model = GCNModel(in_channels, hidden_channels, out_channels)
model = torch.compile(model)  # Compile the model

# Training proceeds as normal
for batch in loader:
    out = model(batch.x, batch.edge_index)  # First call triggers compilation
    # ...
```

### Compile Modes

| Mode | Speed | Correctness | Use Case |
|------|-------|-------------|----------|
| `default` | Good | High | General use |
| `reduce-overhead` | Better | High | Small batches |
| `max-autotune` | Best | Good | Production |

```python
model = torch.compile(model, mode='reduce-overhead')
```

### What Works with Compile

✅ Most PyG layers (GCNConv, SAGEConv, GATConv)
✅ Standard PyTorch operations
✅ Message passing with simple aggregations

⚠️ May need adjustments:
- Dynamic tensor sizes (use `dynamic=True`)
- Sparse operations (some may not compile)

## Memory-Efficient Aggregations

### Use SparseTensor instead of edge_index

```python
from torch_geometric.typing import SparseTensor

# Convert edge_index to SparseTensor
adj = SparseTensor(row=edge_index[0], col=edge_index[1])

# Use in forward
out = model(x, adj)
```

Benefits:
- More memory efficient for dense graphs
- Faster for some operations
- Better cache locality

### Gradient Checkpointing

Trade compute for memory:

```python
from torch.utils.checkpoint import checkpoint

class MemoryEfficientGNN(torch.nn.Module):
    def forward(self, x, edge_index):
        # Checkpoint intermediate layers
        x = checkpoint(self.conv1, x, edge_index, use_reentrant=False)
        x = checkpoint(self.conv2, x, edge_index, use_reentrant=False)
        return x
```

## CPU Affinity for DataLoading

Bind workers to specific CPU cores for better performance:

```python
import os

# Set affinity before creating loaders
os.sched_setaffinity(0, {0, 1, 2, 3})  # Use cores 0-3

loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],
    batch_size=1024,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,  # Keep workers alive
)
```

## Profiling

### PyG Profiler

```python
from torch_geometric.profile import profileit

@profileit()
def train_step(model, data):
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    return loss

# Run with profiling
loss = train_step(model, data)
```

### PyTorch Profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for batch in loader:
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Quick Optimization Checklist

1. **Data Loading**
   - [ ] Use `num_workers > 0`
   - [ ] Enable `pin_memory=True` for GPU
   - [ ] Use `persistent_workers=True`

2. **Model**
   - [ ] Apply `torch.compile(model)`
   - [ ] Use mixed precision (`torch.cuda.amp`)
   - [ ] Use fused optimizers

3. **Memory**
   - [ ] Use NeighborLoader for large graphs
   - [ ] Consider gradient checkpointing
   - [ ] Use SparseTensor if beneficial

4. **Training**
   - [ ] Use appropriate batch sizes
   - [ ] Enable cudnn benchmarking
   - [ ] Profile before optimizing

## Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in loader:
    optimizer.zero_grad()
    
    with autocast():  # Automatic mixed precision
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## References

- `11_compiled_graph_neural_networks.md` - torch.compile guide
- `09_memory_efficient_aggregations.md` - SparseTensor usage
- `15_cpu_affinity_for_pyg_workloads.md` - CPU optimization
- `30_torch_geometric_profile.md` - Profiling tools
