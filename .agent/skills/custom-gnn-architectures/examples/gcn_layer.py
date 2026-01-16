"""
Custom GCN Layer Implementation

This example shows how to implement a Graph Convolutional Network (GCN) layer
from scratch using the MessagePassing base class.

Based on: "Semi-Supervised Classification with Graph Convolutional Networks"
          by Kipf & Welling (2017)
"""

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network layer.
    
    Formula:
        x_i^(k) = Σ_{j ∈ N(i) ∪ {i}} (1 / √(d_i * d_j)) * W * x_j + b
    
    Where d_i and d_j are the degrees of nodes i and j.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # Sum aggregation
        
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.empty(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        
        # Step 1: Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Step 2: Linear transformation
        x = self.lin(x)
        
        # Step 3: Compute normalization coefficients
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4-5: Start propagating messages (calls message + aggregate)
        out = self.propagate(edge_index, x=x, norm=norm)
        
        # Step 6: Add bias
        out = out + self.bias
        
        return out
    
    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        """
        Construct messages from neighbors.
        
        Args:
            x_j: Features of source (neighbor) nodes [num_edges, out_channels]
            norm: Normalization coefficients [num_edges]
        
        Returns:
            Normalized messages [num_edges, out_channels]
        """
        # Normalize neighbor features by degree
        return norm.view(-1, 1) * x_j


class GCNModel(torch.nn.Module):
    """
    Two-layer GCN model for node classification.
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)
        return x


if __name__ == "__main__":
    # Example usage with synthetic data
    from torch_geometric.data import Data
    
    # Create a simple graph
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ], dtype=torch.long)
    
    x = torch.randn(4, 16)  # 4 nodes, 16 features
    
    data = Data(x=x, edge_index=edge_index)
    
    # Create and run model
    model = GCNModel(in_channels=16, hidden_channels=32, out_channels=2)
    out = model(data.x, data.edge_index)
    
    print(f"Input shape: {data.x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output:\n{out}")
