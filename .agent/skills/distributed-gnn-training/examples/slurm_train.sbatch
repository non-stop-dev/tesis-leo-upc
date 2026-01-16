#!/bin/bash
#SBATCH --job-name=pyg-gnn-train
#SBATCH --output=gnn_train_%j.log
#SBATCH --error=gnn_train_%j.err
#SBATCH --partition=gpu              # ADJUST: Your GPU partition name
#SBATCH -N 2                          # Number of nodes
#SBATCH --ntasks=4                    # Total processes (GPUs)
#SBATCH --gpus-per-task=1             # 1 GPU per process
#SBATCH --cpus-per-task=8             # CPUs for data loading
#SBATCH --mem=64G                     # Memory per node
#SBATCH --time=04:00:00               # Max runtime
#SBATCH --gpu-bind=none               # Required for NCCL

# ===========================================================
# SLURM Multi-Node Multi-GPU Training Script for GNNs
# ===========================================================
#
# This script configures a distributed training job across
# multiple nodes, each with multiple GPUs.
#
# Usage:
#   sbatch slurm_train.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f gnn_train_<jobid>.log
#
# ===========================================================

echo "=========================================="
echo "SLURM Job Configuration"
echo "=========================================="
echo "Job ID: $SLURM_JOBID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "=========================================="

# Set up master address for torch.distributed
# Use a pseudo-random port based on job ID to avoid conflicts
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Optional: Activate conda/venv environment
# source ~/.bashrc
# conda activate pyg-env
# source /path/to/venv/bin/activate

# Optional: Set CUDA visibility (if needed)
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: NCCL debugging
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

# Run the training script with srun
# srun automatically sets RANK, LOCAL_RANK, WORLD_SIZE environment variables
echo ""
echo "Starting distributed training..."
echo "=========================================="

srun python train_multinode.py \
    --hidden-channels 256 \
    --num-layers 3 \
    --batch-size 1024 \
    --epochs 50 \
    --lr 0.001

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
