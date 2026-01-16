"""
Multi-GPU Distributed Training Example

This example demonstrates how to train a GNN across multiple GPUs
on a single machine using PyTorch's DistributedDataParallel (DDP).

Based on: PyG tutorial on multi-GPU training with vanilla PyTorch
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Reddit


def run(rank: int, world_size: int, dataset):
    """
    Training function executed by each GPU process.
    
    Args:
        rank: Process ID (0 to world_size-1), also used as GPU ID
        world_size: Total number of GPUs/processes
        dataset: Shared dataset (loaded before spawning)
    """
    
    # =========== Setup Process Group ===========
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    data = dataset[0]
    
    # =========== Split Data Across GPUs ===========
    # Each GPU gets a portion of the training nodes
    train_index = data.train_mask.nonzero().view(-1)
    train_index = train_index.split(train_index.size(0) // world_size)[rank]
    
    train_loader = NeighborLoader(
        data,
        input_nodes=train_index,
        num_neighbors=[25, 10],
        batch_size=1024,
        num_workers=4,
        shuffle=True,
    )
    
    # Only rank 0 handles validation
    if rank == 0:
        val_index = data.val_mask.nonzero().view(-1)
        val_loader = NeighborLoader(
            data,
            input_nodes=val_index,
            num_neighbors=[25, 10],
            batch_size=1024,
            num_workers=4,
            shuffle=False,
        )
    
    # =========== Model Setup ===========
    torch.manual_seed(12345)
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=256,
        num_layers=2,
        out_channels=dataset.num_classes,
    ).to(rank)
    
    # Wrap with DistributedDataParallel for gradient synchronization
    model = DistributedDataParallel(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # =========== Training Loop ===========
    for epoch in range(1, 11):
        model.train()
        total_loss = 0
        total_examples = 0
        
        for batch in train_loader:
            batch = batch.to(rank)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        
        # Synchronize all processes before evaluation
        dist.barrier()
        
        # =========== Validation (rank 0 only) ===========
        if rank == 0:
            avg_loss = total_loss / total_examples
            print(f'Epoch: {epoch:02d}, Train Loss: {avg_loss:.4f}')
            
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(rank)
                    out = model(batch.x, batch.edge_index)[:batch.batch_size]
                    pred = out.argmax(dim=-1)
                    correct += (pred == batch.y[:batch.batch_size]).sum().item()
                    total += batch.batch_size
            
            print(f'Validation Accuracy: {correct / total:.4f}')
        
        # Sync again before next epoch
        dist.barrier()
    
    # =========== Cleanup ===========
    dist.destroy_process_group()


def main():
    """Main entry point - spawns training processes."""
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available. This example requires GPUs.")
        return
    
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs")
    
    if world_size < 2:
        print("This example requires at least 2 GPUs for demonstration.")
        print("Running single-GPU fallback...")
        # Could add single-GPU fallback here
        return
    
    # Load dataset BEFORE spawning processes
    # This allows data to be shared in memory across processes
    print("Loading Reddit dataset...")
    dataset = Reddit('./data/Reddit')
    print(f"Dataset loaded: {dataset[0].num_nodes:,} nodes, {dataset[0].num_edges:,} edges")
    
    # Spawn training processes
    print(f"Spawning {world_size} training processes...")
    mp.spawn(
        run,
        args=(world_size, dataset),
        nprocs=world_size,
        join=True,
    )
    
    print("Training complete!")


if __name__ == '__main__':
    main()
