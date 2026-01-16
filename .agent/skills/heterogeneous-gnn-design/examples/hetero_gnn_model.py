"""
Heterogeneous GNN for MSME Survival Prediction

This example demonstrates how to build a heterogeneous GNN model
for predicting MSME survival using multiple node types:
- Empresas (MSMEs)
- Distritos (Districts)
- Sectores (Economic Sectors)

Two approaches are shown:
1. Using to_hetero() for automatic conversion
2. Using HeteroConv for manual control
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv, Linear, to_hetero
import torch_geometric.transforms as T


# ===========================================================
# APPROACH 1: Automatic conversion with to_hetero()
# ===========================================================

class HomogeneousGNN(torch.nn.Module):
    """
    A simple GNN defined for homogeneous graphs.
    Will be converted to heterogeneous using to_hetero().
    """
    
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        # Use -1 for lazy initialization (auto-detect input size)
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def create_hetero_model_approach1(metadata: tuple, hidden: int = 64, out: int = 2):
    """
    Create heterogeneous model using to_hetero() transformation.
    
    Args:
        metadata: Tuple of (node_types, edge_types) from HeteroData
        hidden: Hidden layer size
        out: Output classes (2 for binary survival)
    
    Returns:
        Heterogeneous GNN model
    """
    model = HomogeneousGNN(hidden, out)
    model = to_hetero(model, metadata, aggr='sum')
    return model


# ===========================================================
# APPROACH 2: Manual HeteroConv layers
# ===========================================================

class HeteroGNN(torch.nn.Module):
    """
    Manually constructed heterogeneous GNN using HeteroConv.
    
    This gives more control over which layer types are used
    for each edge type.
    """
    
    def __init__(
        self, 
        hidden_channels: int, 
        out_channels: int,
        metadata: tuple,
    ):
        super().__init__()
        
        node_types, edge_types = metadata
        
        # First layer: different convolutions per edge type
        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in edge_types
        }, aggr='sum')
        
        # Second layer: use GAT for empresa-empresa edges
        conv2_dict = {}
        for edge_type in edge_types:
            src, rel, dst = edge_type
            if src == dst == 'empresa':
                # Use attention for competition edges
                conv2_dict[edge_type] = GATConv(
                    (-1, -1), out_channels, heads=1, add_self_loops=False
                )
            else:
                conv2_dict[edge_type] = SAGEConv((-1, -1), out_channels)
        
        self.conv2 = HeteroConv(conv2_dict, aggr='sum')
        
        # Linear projection for each node type
        self.lins = torch.nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in node_types
        })
    
    def forward(
        self, 
        x_dict: dict[str, Tensor], 
        edge_index_dict: dict[tuple, Tensor]
    ) -> dict[str, Tensor]:
        """
        Forward pass on heterogeneous graph.
        
        Args:
            x_dict: Dict mapping node_type -> features
            edge_index_dict: Dict mapping edge_type -> edge_index
        
        Returns:
            Dict mapping node_type -> predictions
        """
        # Initial linear projection
        x_dict = {
            node_type: self.lins[node_type](x).relu()
            for node_type, x in x_dict.items()
        }
        
        # First conv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        
        # Second conv layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        return x_dict


# ===========================================================
# Data Creation and Training
# ===========================================================

def create_sample_hetero_data() -> HeteroData:
    """Create sample heterogeneous data for demonstration."""
    
    data = HeteroData()
    
    # Node features
    num_empresas = 1000
    num_distritos = 50
    num_sectores = 20
    
    data['empresa'].x = torch.randn(num_empresas, 8)
    data['empresa'].y = torch.randint(0, 2, (num_empresas,))
    
    data['distrito'].x = torch.randn(num_distritos, 4)
    data['sector'].x = torch.randn(num_sectores, 3)
    
    # Edge indices
    data['empresa', 'ubicado_en', 'distrito'].edge_index = torch.stack([
        torch.arange(num_empresas),
        torch.randint(0, num_distritos, (num_empresas,))
    ])
    
    data['empresa', 'pertenece_a', 'sector'].edge_index = torch.stack([
        torch.arange(num_empresas),
        torch.randint(0, num_sectores, (num_empresas,))
    ])
    
    data['empresa', 'compite_con', 'empresa'].edge_index = torch.stack([
        torch.randint(0, num_empresas, (5000,)),
        torch.randint(0, num_empresas, (5000,))
    ])
    
    # Train/val/test masks
    perm = torch.randperm(num_empresas)
    data['empresa'].train_mask = torch.zeros(num_empresas, dtype=torch.bool)
    data['empresa'].val_mask = torch.zeros(num_empresas, dtype=torch.bool)
    data['empresa'].test_mask = torch.zeros(num_empresas, dtype=torch.bool)
    
    data['empresa'].train_mask[perm[:int(0.8 * num_empresas)]] = True
    data['empresa'].val_mask[perm[int(0.8 * num_empresas):int(0.9 * num_empresas)]] = True
    data['empresa'].test_mask[perm[int(0.9 * num_empresas):]] = True
    
    # Add reverse edges for bidirectional message passing
    data = T.ToUndirected()(data)
    
    return data


def train_hetero_model(model, data, epochs: int = 100):
    """Training loop for heterogeneous model."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x_dict, data.edge_index_dict)
        
        # Loss on empresa predictions only
        pred = out['empresa']
        mask = data['empresa'].train_mask
        loss = F.cross_entropy(pred[mask], data['empresa'].y[mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict)
                pred = out['empresa'].argmax(dim=1)
                
                train_acc = (pred[data['empresa'].train_mask] == 
                            data['empresa'].y[data['empresa'].train_mask]).float().mean()
                val_acc = (pred[data['empresa'].val_mask] == 
                          data['empresa'].y[data['empresa'].val_mask]).float().mean()
                
            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.4f} | Val: {val_acc:.4f}")
    
    return model


if __name__ == "__main__":
    # Create sample data
    data = create_sample_hetero_data()
    print(f"Node types: {data.node_types}")
    print(f"Edge types: {data.edge_types}")
    
    # Approach 1: Using to_hetero
    print("\n--- Approach 1: to_hetero() ---")
    model1 = create_hetero_model_approach1(data.metadata(), hidden=64, out=2)
    model1 = train_hetero_model(model1, data, epochs=50)
    
    # Approach 2: Manual HeteroConv
    print("\n--- Approach 2: HeteroConv ---")
    model2 = HeteroGNN(hidden_channels=64, out_channels=2, metadata=data.metadata())
    model2 = train_hetero_model(model2, data, epochs=50)
