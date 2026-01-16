"""
GraphGPS Example

Demonstrates how to build and train a Graph Transformer (GraphGPS)
that combines local MPNN with global attention.

Based on: 06e_graph_transformer.md
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, ModuleList, Embedding
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool


class GraphGPS(torch.nn.Module):
    """
    GraphGPS model combining local MPNN with global Transformer.
    
    Architecture:
        - Node + PE embeddings
        - Stack of GPSConv layers (MPNN + Attention + FFN)
        - Global pooling + MLP head
    """
    
    def __init__(
        self,
        channels: int = 64,
        pe_dim: int = 8,
        num_layers: int = 5,
        attn_type: str = 'multihead',
        num_node_types: int = 28,
        num_edge_types: int = 4,
    ):
        super().__init__()
        
        # Input embeddings
        self.node_emb = Embedding(num_node_types, channels - pe_dim)
        self.pe_lin = Linear(20, pe_dim)  # walk_length=20
        self.pe_norm = BatchNorm1d(20)
        self.edge_emb = Embedding(num_edge_types, channels)
        
        # GPS layers
        self.convs = ModuleList()
        for _ in range(num_layers):
            # Local MPNN: GIN with edge features
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            
            # GPS layer = MPNN + Transformer + FFN
            conv = GPSConv(
                channels,
                GINEConv(nn),
                heads=4,
                attn_type=attn_type,
                dropout=0.3,
            )
            self.convs.append(conv)
        
        # Output head
        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),  # Regression output
        )
    
    def forward(self, x, pe, edge_index, edge_attr, batch):
        # Encode inputs
        x = self.node_emb(x.squeeze(-1))
        pe = self.pe_lin(self.pe_norm(pe))
        x = torch.cat([x, pe], dim=-1)
        
        edge_attr = self.edge_emb(edge_attr)
        
        # Apply GPS layers
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        
        # Global pooling + prediction
        x = global_add_pool(x, batch)
        return self.mlp(x)


def train(model, loader, optimizer, device):
    """Training epoch."""
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(
            data.x, data.pe, data.edge_index, 
            data.edge_attr, data.batch
        )
        loss = (out.squeeze() - data.y).abs().mean()  # MAE loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate MAE on test set."""
    model.eval()
    total_error = 0
    
    for data in loader:
        data = data.to(device)
        out = model(
            data.x, data.pe, data.edge_index,
            data.edge_attr, data.batch
        )
        total_error += (out.squeeze() - data.y).abs().sum().item()
    
    return total_error / len(loader.dataset)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load ZINC dataset with Random Walk PE
    print("Loading ZINC dataset with positional encodings...")
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    
    train_dataset = ZINC('./data/ZINC', subset=True, split='train', pre_transform=transform)
    val_dataset = ZINC('./data/ZINC', subset=True, split='val', pre_transform=transform)
    test_dataset = ZINC('./data/ZINC', subset=True, split='test', pre_transform=transform)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Create model
    model = GraphGPS(
        channels=64,
        pe_dim=8,
        num_layers=5,
        attn_type='multihead',  # or 'performer' for large graphs
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    print("\nStarting training...")
    best_val = float('inf')
    
    for epoch in range(1, 51):
        train_loss = train(model, train_loader, optimizer, device)
        val_mae = evaluate(model, val_loader, device)
        test_mae = evaluate(model, test_loader, device)
        
        scheduler.step(val_mae)
        
        if val_mae < best_val:
            best_val = val_mae
            torch.save(model.state_dict(), 'best_graphgps.pt')
        
        print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | '
              f'Val MAE: {val_mae:.4f} | Test MAE: {test_mae:.4f}')
    
    # Load best model
    model.load_state_dict(torch.load('best_graphgps.pt'))
    final_test = evaluate(model, test_loader, device)
    print(f'\nFinal Test MAE: {final_test:.4f}')


if __name__ == "__main__":
    main()
