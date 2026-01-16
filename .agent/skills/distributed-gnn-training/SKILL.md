---
name: distributed-gnn-training
description: Multi-GPU and multi-node distributed training for GNNs using DistributedDataParallel (DDP), SLURM, and PyG's distributed module. Use when training on multiple GPUs or scaling across compute nodes.
---

# Distributed GNN Training Skill

This skill provides expertise for scaling GNN training across multiple GPUs and compute nodes using PyTorch's distributed training infrastructure.

## When to use this skill

- Training on multiple GPUs on a single machine
- Scaling training across multiple nodes (HPC/cluster)
- Using SLURM workload manager for distributed jobs
- Working with graphs too large for single-machine memory
- Using PyG's `torch_geometric.distributed` module

## Training Approaches Overview

| Approach | Use Case | Complexity |
|----------|----------|------------|
| Single-node Multi-GPU | 2-8 GPUs, fits in memory | Medium |
| Multi-node Multi-GPU (SLURM) | HPC clusters | High |
| PyG Distributed (partitioned) | Graphs > machine memory | High |

## Approach 1: Single-Node Multi-GPU (DDP)

Data-parallel training where each GPU runs an identical model copy with synchronized gradients.

### Key Concepts

1. **Process spawning**: Create one process per GPU
2. **Process group**: Initialize communication via `nccl`
3. **Data splitting**: Each process trains on a subset of data
4. **Gradient sync**: `DistributedDataParallel` handles gradient averaging

### Basic Template

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import NeighborLoader

def run(rank: int, world_size: int, dataset):
    # Initialize process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    data = dataset[0]
    
    # Split training indices across GPUs
    train_index = data.train_mask.nonzero().view(-1)
    train_index = train_index.split(train_index.size(0) // world_size)[rank]
    
    # Create loader for this GPU's data slice
    train_loader = NeighborLoader(
        data,
        input_nodes=train_index,
        num_neighbors=[25, 10],
        batch_size=1024,
        num_workers=4,
        shuffle=True,
    )
    
    # Create model and wrap with DDP
    model = MyGNN(...).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(10):
        model.train()
        for batch in train_loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            loss.backward()
            optimizer.step()
        
        dist.barrier()  # Sync before evaluation
    
    dist.destroy_process_group()

if __name__ == '__main__':
    dataset = load_dataset()
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
```

### Important Notes

- Initialize dataset **before** spawning (shared memory)
- Each `rank` gets a unique GPU
- Use `dist.barrier()` to synchronize processes
- Only evaluate on `rank == 0` for simplicity

## Approach 2: Multi-Node with SLURM

For HPC clusters using SLURM workload manager.

### SLURM Batch Script

```bash
#!/bin/bash
#SBATCH --job-name=pyg-train
#SBATCH --output=train.log
#SBATCH --partition=gpu
#SBATCH -N 2                    # 2 nodes
#SBATCH --ntasks=4              # 4 total processes
#SBATCH --gpus-per-task=1       # 1 GPU per process
#SBATCH --gpu-bind=none

# Set up master address
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Run training
srun python train_distributed.py
```

### Modified Training Script

```python
import os
import torch.distributed as dist

# Get rank info from SLURM environment
world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

# Initialize with NCCL backend
dist.init_process_group('nccl', world_size=world_size, rank=rank)

# Use global rank for data splitting
train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

# Use local_rank for GPU assignment
model = MyGNN(...).to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])
```

### Key Differences from Single-Node

- SLURM provides rank/world_size via environment variables
- `local_rank` = GPU on this node, `rank` = global process ID
- Master address auto-detected from SLURM node list

## Approach 3: PyG Distributed (Graph Partitioning)

For graphs **too large to fit in single machine memory**.

### Architecture

1. **Partition graph** using METIS (balanced node distribution)
2. **Local stores** hold graph topology + features per partition
3. **Distributed sampling** via RPC across partitions
4. **DDP** for model training

### Graph Partitioning

```python
from torch_geometric.distributed import Partitioner

partitioner = Partitioner(
    data=data,
    num_parts=4,  # Number of partitions
    root='./partitions',
)
partitioner.generate_partition()
```

### Training with Distributed Loaders

```python
from torch_geometric.distributed import (
    LocalGraphStore, 
    LocalFeatureStore,
    DistNeighborLoader,
)

# Initialize stores from partition files
graph_store = LocalGraphStore.from_partition('./partitions/part_0')
feature_store = LocalFeatureStore.from_partition('./partitions/part_0')

# Create distributed loader
loader = DistNeighborLoader(
    data=(feature_store, graph_store),
    num_neighbors=[15, 10, 5],
    batch_size=1024,
    input_nodes=train_mask,
)

# Training loop (similar to regular)
for batch in loader:
    out = model(batch.x, batch.edge_index)
    # ...
```

### When to Use Each Approach

```
Does the full graph fit in memory?
├── YES → How many GPUs?
│   ├── 1 GPU → Standard single-GPU training
│   ├── 2-8 GPUs (1 node) → DDP (Approach 1)
│   └── Many GPUs (multiple nodes) → SLURM DDP (Approach 2)
└── NO → Use PyG Distributed with partitioning (Approach 3)
```

## Performance Tips

1. **Load data before spawning** for shared memory efficiency
2. **Use NCCL backend** for GPU-to-GPU communication
3. **Pin memory** in data loaders (`pin_memory=True`)
4. **Profile first** to find communication bottlenecks
5. **Consider gradient accumulation** if batch size is limited

## References

- `07a_multi_gpu_vanilla.md` - Single-node multi-GPU
- `07b_multi_node_multi_gpu_slurm.md` - SLURM setup
- `07c_distributed_training_pyg.md` - PyG distributed module
- `27_torch_geometric_distributed.md` - API reference
