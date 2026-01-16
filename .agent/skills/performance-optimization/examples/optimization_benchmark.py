"""
Performance Optimization Example for GNNs

This example demonstrates various techniques to optimize GNN training:
1. torch.compile() for model speedup
2. Mixed precision training
3. Profiling to find bottlenecks
4. Memory-efficient loading
"""

import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv


# ===========================================================
# Model
# ===========================================================

class OptimizedGNN(torch.nn.Module):
    """GNN model designed for optimization."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return x


# ===========================================================
# Benchmark Utilities
# ===========================================================

def create_synthetic_data(num_nodes: int = 50000) -> Data:
    """Create synthetic graph data for benchmarking."""
    
    x = torch.randn(num_nodes, 128)
    y = torch.randint(0, 2, (num_nodes,))
    
    # Create random edges (~20 per node)
    edges_per_node = 20
    src = torch.arange(num_nodes).repeat_interleave(edges_per_node)
    dst = torch.randint(0, num_nodes, (num_nodes * edges_per_node,))
    edge_index = torch.stack([src, dst])
    
    # Train mask
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.8 * num_nodes)] = True
    
    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask)


class Timer:
    """Simple context manager for timing."""
    
    def __init__(self, name: str):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        print(f"{self.name}: {self.elapsed:.3f}s")


# ===========================================================
# Training Variants
# ===========================================================

def train_baseline(model, data, epochs: int = 10):
    """Baseline training without optimizations."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    with Timer(f"Baseline Training ({epochs} epochs)"):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
    
    return model


def train_compiled(model, data, epochs: int = 10):
    """Training with torch.compile()."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    # Compile the model
    compiled_model = torch.compile(model, mode='reduce-overhead')
    
    optimizer = torch.optim.Adam(compiled_model.parameters(), lr=0.01)
    
    # Warmup (first call triggers compilation)
    print("Compiling model (warmup)...")
    with torch.no_grad():
        _ = compiled_model(data.x, data.edge_index)
    
    with Timer(f"Compiled Training ({epochs} epochs)"):
        for epoch in range(epochs):
            compiled_model.train()
            optimizer.zero_grad()
            out = compiled_model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
    
    return compiled_model


def train_mixed_precision(model, data, epochs: int = 10):
    """Training with automatic mixed precision."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("Mixed precision requires CUDA. Skipping.")
        return model
    
    model = model.to(device)
    data = data.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = GradScaler()
    
    with Timer(f"Mixed Precision Training ({epochs} epochs)"):
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            with autocast():
                out = model(data.x, data.edge_index)
                loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    return model


def train_with_sampling(data, epochs: int = 10, batch_size: int = 1024):
    """Training with NeighborLoader (mini-batch)."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = OptimizedGNN(data.num_features, 128, 2).to(device)
    
    loader = NeighborLoader(
        data,
        num_neighbors=[15, 10, 5],
        batch_size=batch_size,
        input_nodes=data.train_mask,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    with Timer(f"Mini-batch Training ({epochs} epochs)"):
        for epoch in range(epochs):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(
                    out[:batch.batch_size], 
                    batch.y[:batch.batch_size]
                )
                loss.backward()
                optimizer.step()
    
    return model


# ===========================================================
# Profiling
# ===========================================================

def profile_model(model, data, num_iterations: int = 5):
    """Profile model execution to find bottlenecks."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    print("\n" + "="*50)
    print("Profiling GNN Model")
    print("="*50)
    
    # Warmup
    for _ in range(3):
        _ = model(data.x, data.edge_index)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile with PyTorch profiler
    from torch.profiler import profile, ProfilerActivity
    
    with profile(
        activities=[ProfilerActivity.CPU] + 
                  ([ProfilerActivity.CUDA] if device.type == 'cuda' else []),
        record_shapes=True,
    ) as prof:
        for _ in range(num_iterations):
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
    
    # Print results
    sort_key = "cuda_time_total" if device.type == 'cuda' else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))


# ===========================================================
# Main
# ===========================================================

def main():
    print("Performance Optimization Benchmark")
    print("="*50)
    
    # Create data
    print("\nCreating synthetic graph...")
    data = create_synthetic_data(num_nodes=50000)
    print(f"Nodes: {data.num_nodes:,}, Edges: {data.num_edges:,}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Compare training methods
    print("\n--- Benchmark Results ---\n")
    
    # 1. Baseline
    model1 = OptimizedGNN(data.num_features, 128, 2)
    train_baseline(model1, data, epochs=10)
    
    # 2. Compiled (if supported)
    if hasattr(torch, 'compile'):
        model2 = OptimizedGNN(data.num_features, 128, 2)
        try:
            train_compiled(model2, data, epochs=10)
        except Exception as e:
            print(f"Compile failed: {e}")
    
    # 3. Mixed Precision (CUDA only)
    model3 = OptimizedGNN(data.num_features, 128, 2)
    train_mixed_precision(model3, data, epochs=10)
    
    # 4. Mini-batch with sampling
    train_with_sampling(data, epochs=10)
    
    # Profile
    print("\n--- Profiling ---")
    model_profile = OptimizedGNN(data.num_features, 128, 2)
    profile_model(model_profile, data)


if __name__ == "__main__":
    main()
