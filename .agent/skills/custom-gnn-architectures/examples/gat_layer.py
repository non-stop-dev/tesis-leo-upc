"""
Custom GAT (Graph Attention Network) Layer Implementation

This example shows how to implement a Graph Attention Network layer
with multi-head attention using the MessagePassing base class.

Based on: "Graph Attention Networks" by Veličković et al. (2018)

Key Features:
- Learns attention weights for each edge
- Multi-head attention for stability
- Can incorporate edge features
"""

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class GATConv(MessagePassing):
    """
    Graph Attention Network layer with multi-head attention.
    
    The attention mechanism computes:
        α_{ij} = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
    
    Where || denotes concatenation and a is a learnable attention vector.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Linear transformation for node features
        self.lin = Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters (one per head)
        # The attention vector for computing α = a^T [Wh_i || Wh_j]
        self.att_src = Parameter(torch.empty(1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, heads, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        torch.nn.init.xavier_uniform_(self.att_src)
        torch.nn.init.xavier_uniform_(self.att_dst)
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
        
        Returns:
            Updated node features [num_nodes, heads * out_channels] or 
                                  [num_nodes, out_channels] if not concat
        """
        H, C = self.heads, self.out_channels
        
        # Linear transformation: [N, in_channels] -> [N, H * C]
        x = self.lin(x).view(-1, H, C)  # [N, H, C]
        
        # Compute attention scores for source and destination
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]
        
        # Propagate messages
        out = self.propagate(
            edge_index, 
            x=x, 
            alpha=(alpha_src, alpha_dst)
        )
        
        # Concatenate or average heads
        if self.concat:
            out = out.view(-1, H * C)  # [N, H * C]
        else:
            out = out.mean(dim=1)  # [N, C]
        
        return out
    
    def message(
        self, 
        x_j: Tensor,          # [E, H, C] - source features
        alpha_j: Tensor,      # [E, H] - source attention
        alpha_i: Tensor,      # [E, H] - target attention  
        index: Tensor,        # [E] - target node indices
        ptr: Tensor = None,
        size_i: int = None,
    ) -> Tensor:
        """
        Compute attention-weighted messages.
        
        Args:
            x_j: Source node features after linear transform
            alpha_j: Source attention scores
            alpha_i: Target attention scores
            index: Target node indices for softmax grouping
        
        Returns:
            Attention-weighted messages [E, H, C]
        """
        # Combine attention scores
        alpha = alpha_j + alpha_i  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        
        # Softmax over neighbors (edges going to same target)
        alpha = softmax(alpha, index, ptr, size_i)  # [E, H]
        
        # Apply dropout to attention weights
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weight source features by attention
        return x_j * alpha.unsqueeze(-1)  # [E, H, C]


class GATModel(torch.nn.Module):
    """
    Two-layer GAT model for node classification.
    
    Architecture:
        Input -> GAT (8 heads, concat) -> ELU -> Dropout -> GAT (1 head, mean) -> Output
    """
    
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int, 
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        
        self.dropout = dropout
        
        # First GAT layer: multi-head with concatenation
        self.conv1 = GATConv(
            in_channels, 
            hidden_channels, 
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        
        # Second GAT layer: single head (or average of heads)
        self.conv2 = GATConv(
            hidden_channels * heads,  # Input is concatenated heads
            out_channels,
            heads=1,
            concat=False,  # Average heads for final output
            dropout=dropout,
        )
    
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_gat(model, data, optimizer, epochs: int = 200):
    """Training loop for GAT model."""
    
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            # Evaluate
            model.eval()
            with torch.no_grad():
                pred = model(data.x, data.edge_index).argmax(dim=1)
                
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).float().mean()
                val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()
                
            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            model.train()
    
    return model


if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    
    # Load Cora dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    print(f"Dataset: {dataset}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Features: {data.num_features}, Classes: {dataset.num_classes}")
    
    # Create model
    model = GATModel(
        in_channels=data.num_features,
        hidden_channels=8,
        out_channels=dataset.num_classes,
        heads=8,
        dropout=0.6,
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    
    # Train
    model = train_gat(model, data, optimizer, epochs=200)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f"\nTest Accuracy: {test_acc:.4f}")
