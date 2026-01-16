---
name: pyg-data-modeling
description: Expertise in defining graph structures using PyTorch Geometric. Use when creating Data/HeteroData objects, importing CSVs/SQL to graph format, or defining node/edge features. Essential for converting tabular MSME data into graph representations.
---

# PyG Data Modeling Skill

This skill provides guidance for converting raw data (CSV, SQL, DataFrames) into PyTorch Geometric graph structures suitable for GNN training.

## When to use this skill

- Converting tabular business data (like MSME census) into graph format
- Defining node features (`x`), edge indices, and edge attributes
- Creating `HeteroData` objects for multi-type nodes/edges
- Building custom `InMemoryDataset` classes for large datasets

## Core Concepts

### The `Data` Object

A single graph in PyG is represented by `torch_geometric.data.Data`:

```python
from torch_geometric.data import Data

data = Data(
    x=node_features,           # [num_nodes, num_features]
    edge_index=edge_index,     # [2, num_edges] in COO format
    edge_attr=edge_features,   # [num_edges, num_edge_features]
    y=labels,                  # Target variable
)
```

**Key attributes:**
- `data.x`: Node feature matrix
- `data.edge_index`: Graph connectivity (source, target pairs as columns)
- `data.edge_attr`: Edge features
- `data.y`: Labels (node-level or graph-level)
- `data.pos`: Node positions (for spatial graphs)

### The `HeteroData` Object

For graphs with multiple node/edge types:

```python
from torch_geometric.data import HeteroData

data = HeteroData()

# Add node features by type
data['empresa'].x = empresa_features      # MSME nodes
data['distrito'].x = distrito_features    # District nodes

# Add edges by relation type
data['empresa', 'ubicado_en', 'distrito'].edge_index = location_edges
data['empresa', 'compite_con', 'empresa'].edge_index = competition_edges
```

## Decision Tree

```
Is your graph homogeneous (single node/edge type)?
├── YES → Use `Data` object
└── NO → Use `HeteroData` object
    └── Do you have >100K nodes?
        ├── YES → Use `InMemoryDataset` with chunked loading
        └── NO → Load directly into memory
```

## Step-by-Step: CSV to Graph

1. **Load your data** as a DataFrame
2. **Define nodes** - each unique entity becomes a node with an integer ID
3. **Define edges** - relationships become (source_id, target_id) pairs
4. **Extract features** - numerical/categorical columns become node features
5. **Build the Data object**

See `examples/csv_to_graph.py` for a complete implementation.

## Validation

Always validate your graph after construction:

```python
data.validate(raise_on_error=True)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Features per node: {data.num_node_features}")
print(f"Has isolated nodes: {data.has_isolated_nodes()}")
print(f"Is undirected: {data.is_undirected()}")
```

## Common Transforms

Apply transforms before training:

```python
import torch_geometric.transforms as T

transform = T.Compose([
    T.NormalizeFeatures(),      # Normalize node features
    T.ToUndirected(),           # Make edges bidirectional
    T.AddSelfLoops(),           # Add self-connections
])

data = transform(data)
```

## References

- `02_introduction_by_example.md` - Core Data concepts
- `05a_loading_graphs_from_csv.md` - CSV loading patterns
- `18_torch_geometric_data.md` - Full Data API reference
