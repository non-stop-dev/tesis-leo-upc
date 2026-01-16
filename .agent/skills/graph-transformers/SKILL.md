---
name: graph-transformers
description: Building Graph Transformer architectures combining local MPNN and global attention (GraphGPS, GPSConv). Use when you need to capture long-range dependencies or overcome GNN limitations like over-squashing.
---

# Graph Transformers Skill

This skill provides expertise for building Graph Transformer architectures that combine local message passing with global self-attention.

## When to use this skill

- Need to capture long-range dependencies in graphs
- Experiencing over-smoothing or over-squashing with deep GNNs
- Want state-of-the-art performance on graph benchmarks
- Graph structure has both local and global patterns

## MPNNs vs Transformers Trade-offs

**MPNN Limitations:**
- Limited expressivity (1-WL test)
- Over-smoothing with many layers
- Over-squashing (information bottleneck)
- Cannot capture long-range dependencies

**Transformer Advantages:**
- All nodes can attend to all others
- Long-range connections naturally handled
- Decoupled from graph structure

**Transformer Disadvantages:**
- Loses locality inductive bias
- O(N²) complexity vs O(E) for MPNNs
- Requires good positional encodings

## GraphGPS Architecture

The solution: **Combine both!**

```
For each layer:
  ├── Local MPNN (GatedGCN, GIN, etc.)
  ├── Global Transformer Attention
  └── FFN + Skip Connections
```

### Layer Formula

$$
\hat{X}_M^{l+1} = \text{MPNN}^l(X^l, E^l, A)
$$

$$
\hat{X}_T^{l+1} = \text{GlobalAttn}^l(X^l)
$$

$$
X^{l+1} = \text{MLP}^l(\hat{X}_M^{l+1} + \hat{X}_T^{l+1})
$$

## GPSConv Layer

PyG provides `GPSConv` for building Graph Transformers:

```python
from torch_geometric.nn import GPSConv, GINEConv

# Define local MPNN
nn = Sequential(
    Linear(channels, channels),
    ReLU(),
    Linear(channels, channels),
)
local_mpnn = GINEConv(nn)

# Create GPS layer
gps_conv = GPSConv(
    channels=64,
    conv=local_mpnn,           # Local message passing
    heads=4,                   # Attention heads
    attn_type='multihead',     # or 'performer' for linear
    dropout=0.5,
)

# Forward pass
x = gps_conv(x, edge_index, batch, edge_attr=edge_attr)
```

## Positional Encodings

Graph Transformers need positional/structural encodings since graphs lack natural ordering.

### Types of Encodings

| Type | Examples | Captures |
|------|----------|----------|
| Local PE | Distance to cluster center | Local position |
| Global PE | Laplacian eigenvectors | Global structure |
| Local SE | Node degree, random walk diagonals | Local structure |
| Relative PE | Pairwise distances | Edge-level info |

### Adding Random Walk PE

```python
import torch_geometric.transforms as T

# Add as preprocessing transform
transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
data = transform(data)

# data.pe now contains [N, walk_length] positional features
```

### Using PE in Model

```python
class GPS(torch.nn.Module):
    def __init__(self, channels, pe_dim):
        super().__init__()
        # Project PE to smaller dimension
        self.pe_lin = Linear(20, pe_dim)
        self.pe_norm = BatchNorm1d(20)
        
        # Node embedding (leave room for PE)
        self.node_emb = Linear(in_channels, channels - pe_dim)
    
    def forward(self, x, pe, edge_index, batch):
        # Combine node features with PE
        pe = self.pe_norm(pe)
        x = torch.cat([self.node_emb(x), self.pe_lin(pe)], dim=-1)
        
        # Apply GPS layers
        for conv in self.convs:
            x = conv(x, edge_index, batch)
        
        return x
```

## Complete GraphGPS Model

```python
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d, ModuleList
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool

class GraphGPS(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels, pe_dim=8, num_layers=4):
        super().__init__()
        
        # Input projections
        self.node_emb = Linear(in_channels, hidden - pe_dim)
        self.pe_lin = Linear(20, pe_dim)  # 20 = walk_length
        self.pe_norm = BatchNorm1d(20)
        
        # GPS layers
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(hidden, hidden),
                ReLU(),
                Linear(hidden, hidden),
            )
            conv = GPSConv(
                hidden, 
                GINEConv(nn),
                heads=4,
                attn_type='multihead',
            )
            self.convs.append(conv)
        
        # Output
        self.mlp = Sequential(
            Linear(hidden, hidden // 2),
            ReLU(),
            Linear(hidden // 2, out_channels),
        )
    
    def forward(self, x, pe, edge_index, edge_attr, batch):
        # Combine features with PE
        x = torch.cat([
            self.node_emb(x),
            self.pe_lin(self.pe_norm(pe))
        ], dim=-1)
        
        # GPS layers
        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        
        # Pooling + classification
        x = global_add_pool(x, batch)
        return self.mlp(x)
```

## Attention Types

| Type | Complexity | Notes |
|------|------------|-------|
| `multihead` | O(N²) | Standard attention |
| `performer` | O(N) | Approximate, faster |

```python
# For large graphs, use Performer
gps = GPSConv(hidden, conv, attn_type='performer')
```

## Training Tips

1. **Use edge features** with GINEConv or GatedGCN as local MPNN
2. **Add PE** for better structural awareness
3. **Start with multihead**, switch to performer for large graphs
4. **Use residual connections** (built into GPSConv)
5. **Learning rate scheduling** often helps

## Decision Tree

```
Need long-range dependencies?
├── NO → Use standard GNN (GCN, GAT, etc.)
└── YES → Graph Transformer
    └── Graph size?
        ├── Small (<5K nodes) → GPSConv with multihead attention
        └── Large → GPSConv with performer attention
            └── Very large → Consider sampling + GPS
```

## References

- `06e_graph_transformer.md` - Complete GraphGPS tutorial
- `17_torch_geometric_nn.md` - GPSConv API
- [GraphGPS Paper](https://arxiv.org/abs/2205.12454)
