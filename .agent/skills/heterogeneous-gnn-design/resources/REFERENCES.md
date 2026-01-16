# Documentation Resources

This folder contains references to PyG documentation files that complement this skill.

## Primary References

- **04b_heterogeneous_graph_learning.md** - Complete tutorial on heterogeneous graphs
- **17_torch_geometric_nn.md** - HeteroConv and related layers
- **28_torch_geometric_contrib.md** - Additional heterogeneous layers

## Key Concepts from Documentation

### HeteroData Structure

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Node features by type
data['paper'].x = paper_features
data['author'].x = author_features

# Edges by relation (source_type, relation, target_type)
data['author', 'writes', 'paper'].edge_index = ...
data['paper', 'cites', 'paper'].edge_index = ...
```

### Converting Homogeneous to Heterogeneous

From `04b_heterogeneous_graph_learning.md`:

```python
from torch_geometric.nn import to_hetero

# Define model for homogeneous graph
model = HomogeneousGNN(hidden_channels=64)

# Convert to heterogeneous
model = to_hetero(model, data.metadata(), aggr='sum')

# Forward pass with dict inputs
out = model(data.x_dict, data.edge_index_dict)
```

### Metadata

`data.metadata()` returns a tuple:
1. `node_types`: List of node type strings
2. `edge_types`: List of (src, rel, dst) tuples

### HeteroConv for Manual Control

```python
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv

conv = HeteroConv({
    ('user', 'rates', 'movie'): SAGEConv(-1, 64),
    ('movie', 'rev_rates', 'user'): SAGEConv(-1, 64),
}, aggr='sum')
```

### Adding Reverse Edges

Essential for bidirectional message passing:

```python
import torch_geometric.transforms as T

data = T.ToUndirected()(data)
# Creates reverse edges like ('movie', 'rev_rates', 'user')
```
