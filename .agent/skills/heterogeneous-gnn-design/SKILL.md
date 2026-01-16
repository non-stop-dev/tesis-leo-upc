---
name: heterogeneous-gnn-design
description: Guidance for modeling graphs with multiple node and edge types. Use when working with HeteroData objects, applying to_hetero transformations, or building models that handle MSMEs, Districts, and Sectors as different entity types.
---

# Heterogeneous GNN Design Skill

This skill provides expertise for working with heterogeneous graphs - graphs that have multiple types of nodes and edges, such as MSMEs connected to Districts and Sectors.

## When to use this skill

- Your graph has semantically different node types (e.g., empresas, distritos, sectores)
- Your edges represent different relationships (ubicado_en, compite_con, pertenece_a)
- You want to apply different transformations to different node/edge types
- You need to convert a homogeneous model to work on heterogeneous data

## Heterogeneous Graph Structure

A heterogeneous graph consists of:
- **Node types**: Different entity types (e.g., `'empresa'`, `'distrito'`, `'sector'`)
- **Edge types**: Different relationships as triplets `(source_type, relation, target_type)`

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features by type
data['empresa'].x = torch.randn(1000, 8)
data['distrito'].x = torch.randn(50, 4)

# Edge indices by relation type
data['empresa', 'ubicado_en', 'distrito'].edge_index = ...
data['empresa', 'compite_con', 'empresa'].edge_index = ...
```

## Approach 1: Convert Homogeneous Model with `to_hetero`

The easiest way to create a heterogeneous GNN is to define a homogeneous model and convert it:

```python
from torch_geometric.nn import GCNConv, to_hetero

class HomogeneousGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(-1, hidden_channels)  # -1 for lazy init
        self.conv2 = GCNConv(-1, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Convert to heterogeneous
model = HomogeneousGNN(64, 2)
model = to_hetero(model, data.metadata(), aggr='sum')
```

The `to_hetero` transformation:
- Creates separate weights for each node/edge type
- Automatically routes messages based on edge types
- Uses `aggr` to combine messages from different edge types

## Approach 2: Manual HeteroConv

For more control, use `HeteroConv` to specify different layers per edge type:

```python
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv

conv = HeteroConv({
    ('empresa', 'ubicado_en', 'distrito'): GCNConv(-1, 64),
    ('distrito', 'rev_ubicado_en', 'empresa'): GCNConv(-1, 64),
    ('empresa', 'compite_con', 'empresa'): GATConv(-1, 64, heads=4),
}, aggr='sum')

# Usage
x_dict = conv(x_dict, edge_index_dict)
```

## Important: Add Reverse Edges

For bidirectional message passing, add reverse edges:

```python
import torch_geometric.transforms as T

transform = T.ToUndirected()
data = transform(data)
```

This creates:
- `('distrito', 'rev_ubicado_en', 'empresa')` from `('empresa', 'ubicado_en', 'distrito')`

## Training on Heterogeneous Graphs

```python
# Forward pass returns dict of node embeddings
out = model(data.x_dict, data.edge_index_dict)

# Access predictions for specific node type
empresa_pred = out['empresa']

# Compute loss only on target node type
loss = F.cross_entropy(
    empresa_pred[data['empresa'].train_mask],
    data['empresa'].y[data['empresa'].train_mask]
)
```

## Lazy Initialization with `-1`

Use `in_channels=-1` for automatic feature size detection:

```python
# First forward pass will infer input dimensions
conv = GCNConv(-1, 64)
```

## Decision Tree

```
Do you have multiple node or edge types?
├── NO → Use standard homogeneous GNN
└── YES → Use HeteroData
    └── Want simple conversion?
        ├── YES → Define homogeneous model + to_hetero()
        └── NO → Use HeteroConv manually
            └── Need different architectures per edge type?
                ├── YES → Specify different conv layers in HeteroConv
                └── NO → Use same conv type with shared weights
```

## Schema for MSME Survival

```
Node Types:
  empresas: [ventas, productividad, regimen, tributos, ...]
  distritos: [poblacion, densidad, ruralidad, ...]
  sectores: [tamano_sector, concentracion, ...]

Edge Types:
  (empresa, ubicado_en, distrito)
  (empresa, pertenece_a, sector)  
  (empresa, compite_con, empresa)  # Same sector + district
```

## References

- `04b_heterogeneous_graph_learning.md` - Full tutorial
- `17_torch_geometric_nn.md` - HeteroConv API
- `28_torch_geometric_contrib.md` - Advanced hetero layers
