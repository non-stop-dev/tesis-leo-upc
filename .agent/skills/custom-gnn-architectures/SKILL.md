---
name: custom-gnn-architectures
description: Specialized in designing custom GNN layers (GCN, GAT, GraphSAGE) by extending the MessagePassing base class. Use when implementing specific attention mechanisms, custom aggregation functions, or novel GNN architectures for node classification.
---

# Custom GNN Architectures Skill

This skill provides expertise for building custom Graph Neural Network layers using PyTorch Geometric's `MessagePassing` base class.

## When to use this skill

- Implementing custom GNN layers (GAT with custom attention, etc.)
- Understanding message passing fundamentals
- Creating novel aggregation schemes
- Building GNN models for node-level classification (e.g., survival prediction)

## Theoretical Foundations

> **See**: `resources/REFERENCES.md` for detailed academic background

### Spectral vs. Spatial Convolutions
- **Spectral Domain**: Convolutions defined via the Graph Fourier Transform using the Graph Laplacian $L = D - A$.
    - computationally expensive ($\mathcal{O}(N^3)$) due to eigen-decomposition.
    - **ChebNet**: Approximates spectral filters using Chebyshev polynomials $T_k(x)$ up to order $K$, reducing complexity to $\mathcal{O}(K|E|)$.
    - **GCN**: A first-order approximation ($K=1$) of ChebNet with additional constraints to avoid overfitting.

- **Spatial Domain**: Message passing directly on the graph structure.
    - **MessagePassing**: The spatial generalization where nodes aggregate information from local neighborhoods.
    - Most modern GNNs (GraphSAGE, GAT) are spatial, avoiding the need for the entire graph structure during training (inductive).

### Polynomial Filters
A graph convolution can be viewed as learning a polynomial filter $g_\theta(\Lambda) = \sum_{k=0}^K \theta_k \Lambda^k$.
- **Interpretation**: A $K$-th order polynomial captures information from $K$-hop neighbors.
- **GCN** effectively uses $K=1$ but stacks layers to increase the receptive field.

## Core Concept: Message Passing

GNNs can be described by the message passing framework:

$$
\mathbf{x}_i^{(k)} = \gamma^{(k)} \left( \mathbf{x}_i^{(k-1)}, \bigoplus_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\mathbf{x}_i^{(k-1)}, \mathbf{x}_j^{(k-1)}, \mathbf{e}_{j,i}\right) \right)
$$

Where:
- $\phi$ = `message()` function - computes messages from neighbors
- $\bigoplus$ = `aggregate()` function - combines messages (sum, mean, max)
- $\gamma$ = `update()` function - updates node embeddings

## The MessagePassing Base Class

```python
from torch_geometric.nn import MessagePassing

class MyGNNLayer(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # Aggregation: 'add', 'mean', 'max'
    
    def forward(self, x, edge_index):
        # Called when you use the layer
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j: features of source nodes (neighbors)
        return x_j
    
    def update(self, aggr_out):
        # aggr_out: aggregated messages
        return aggr_out
```

## Key Methods

| Method | Purpose | Arguments |
|--------|---------|-----------|
| `propagate()` | Starts message passing | `edge_index`, plus any kwargs |
| `message()` | Computes $\phi$ (message function) | `x_i`, `x_j`, `edge_attr`, etc. |
| `aggregate()` | Combines messages with $\bigoplus$ | Default based on `aggr` |
| `update()` | Final transformation $\gamma$ | Aggregated output |

## Variable Naming Convention

In `message()`, use suffixes to access node features:
- `x_j` → features of **source** (neighbor) nodes
- `x_i` → features of **target** (central) nodes
- `edge_attr` → edge features

## Common Architectures

### GCN (Graph Convolutional Network)
- Aggregation: Weighted sum with degree normalization
- Formula: $\mathbf{x}_i^{(k)} = \sum_{j} \frac{1}{\sqrt{d_i d_j}} \mathbf{W} \mathbf{x}_j$

### GAT (Graph Attention Network)
- Aggregation: Attention-weighted sum
- Learns which neighbors are most important
- See `examples/gat_layer.py`

### GraphSAGE
- Aggregation: Sample and aggregate neighbors
- Scalable to large graphs

## Decision Tree

```
What type of aggregation do you need?
├── Simple neighborhood averaging → Use GCNConv
├── Learn neighbor importance → Use GATConv or GATv2Conv
├── Max pooling over neighbors → Use SAGEConv with aggr='max'
└── Custom logic → Extend MessagePassing
```

## Advanced Architectures: Skip Connections
To prevent **oversmoothing** in deep GNNs (where node embeddings become indistinguishable), use skip connections:

1.  **Residual Connections (ResNet)**:
    $\mathbf{h}^{(k)} = \sigma(\text{Conv}(\mathbf{h}^{(k-1)})) + \mathbf{h}^{(k-1)}$
    Simple element-wise addition. Requires same dimension.

2.  **Jumping Knowledge Networks (JKN)**:
    Combines embeddings from *all* layers at the final step.
    $\mathbf{h}_v^{\text{final}} = \text{AGG}(\mathbf{h}_v^{(1)}, \mathbf{h}_v^{(2)}, \dots, \mathbf{h}_v^{(K)})$
    Aggregation can be concatenation, max-pooling, or LSTM-attention.
    *Effective for structure-oriented tasks where local information is preserved.*

3.  **Highway GCN**:
    Uses gating mechanisms to control information flow.
    $\mathbf{T} = \sigma(\mathbf{W}_T \mathbf{h}^{(k-1)})$, $\mathbf{h}^{(k)} = \mathbf{T} \odot \text{Conv}(\mathbf{h}^{(k-1)}) + (1 - \mathbf{T}) \odot \mathbf{h}^{(k-1)}$

## Building a Complete Model

```python
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
```

## Training Loop Template

```python
model = GNNModel(data.num_features, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

## References

- `04a_creating_message_passing_networks.md` - MessagePassing tutorial
- `17_torch_geometric_nn.md` - All available layers
- `31_gnn_cheatsheet.md` - Quick reference
