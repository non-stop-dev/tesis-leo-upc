"""
Multi-Node Multi-GPU Training Script

This script is designed to be launched via SLURM's srun command.
It reads rank and world_size from environment variables set by SLURM.

Usage:
    srun python train_multinode.py --hidden-channels 256 --epochs 50
    
Or submitted via sbatch with the slurm_train.sbatch script.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn import GraphSAGE
from torch_geometric.loader import NeighborLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-node GNN Training')
    parser.add_argument('--hidden-channels', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num-neighbors', type=str, default='25,10',
                        help='Comma-separated neighbor counts per layer')
    return parser.parse_args()


def get_distributed_info():
    """Get rank and world_size from environment variables set by SLURM."""
    
    # WORLD_SIZE: Total number of processes
    world_size = int(os.environ.get('WORLD_SIZE', 
                                    os.environ.get('SLURM_NTASKS', 1)))
    
    # RANK: Global rank of this process
    rank = int(os.environ.get('RANK', 
                              os.environ.get('SLURM_PROCID', 0)))
    
    # LOCAL_RANK: Rank within this node (for GPU selection)
    local_rank = int(os.environ.get('LOCAL_RANK', 
                                    os.environ.get('SLURM_LOCALID', 0)))
    
    return world_size, rank, local_rank


def run(args):
    """Main training function."""
    
    world_size, rank, local_rank = get_distributed_info()
    
    if rank == 0:
        print(f"World size: {world_size}")
        print(f"Initializing process group...")
    
    # Initialize distributed process group
    # MASTER_ADDR and MASTER_PORT should be set by SLURM script
    dist.init_process_group('nccl', world_size=world_size, rank=rank)
    
    # Set device based on local rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    if rank == 0:
        print(f"Loading dataset...")
    
    # Load dataset - each process loads independently in multi-node
    from torch_geometric.datasets import Reddit
    dataset = Reddit('./data/Reddit')
    data = dataset[0]
    
    if rank == 0:
        print(f"Dataset: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
    
    # Split training indices across all processes using GLOBAL rank
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]
    
    # Parse neighbor counts
    num_neighbors = [int(x) for x in args.num_neighbors.split(',')]
    
    train_loader = NeighborLoader(
        data,
        input_nodes=train_idx,
        num_neighbors=num_neighbors,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    
    # Create model and wrap with DDP using LOCAL_RANK for device
    torch.manual_seed(12345)
    model = GraphSAGE(
        in_channels=dataset.num_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        out_channels=dataset.num_classes,
    ).to(local_rank)
    
    model = DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if rank == 0:
        print(f"Starting training for {args.epochs} epochs...")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_examples = 0
        
        for batch in train_loader:
            batch = batch.to(local_rank)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index)[:batch.batch_size]
            loss = F.cross_entropy(out, batch.y[:batch.batch_size])
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
        
        # Synchronize before logging
        dist.barrier()
        
        if rank == 0 and epoch % 5 == 0:
            avg_loss = total_loss / total_examples
            print(f'Epoch {epoch:03d} | Loss: {avg_loss:.4f}')
    
    # Save model (only rank 0)
    if rank == 0:
        print("Saving model...")
        torch.save(model.module.state_dict(), 'model_checkpoint.pt')
        print("Training complete!")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    run(args)
