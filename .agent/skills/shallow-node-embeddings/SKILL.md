---
name: shallow-node-embeddings
description: Unsupervised node embedding techniques like Node2Vec and MetaPath2Vec for learning representations from graph structure. Use when you lack node features or want to pretrain embeddings before GNN training.
---

# Shallow Node Embeddings Skill

This skill provides expertise for learning unsupervised node embeddings using techniques like Node2Vec and MetaPath2Vec.

## When to use this skill

- Your graph lacks rich node features
- You want to pretrain node embeddings before GNN training
- You need unsupervised representations that preserve graph structure
- Working with heterogeneous graphs (use MetaPath2Vec)

## Shallow vs Deep Embeddings

| Aspect | Shallow (Node2Vec) | Deep (GNNs) |
|--------|-------------------|-------------|
| Features | Not used | Required |
| Learning | Unsupervised | Supervised |
| Inductive | No (transductive) | Yes |
| Parameters | O(N × d) | O(layers × hidden) |
| Use case | Pretraining, no features | End-to-end learning |

## Core Concept

Shallow embeddings learn a lookup table of node vectors $\mathbf{z}_v$ such that nearby nodes (via random walks) have similar embeddings:

$$
\mathcal{L} = \sum_{w \in \mathcal{W}} - \log \sigma(\mathbf{z}_v^\top \mathbf{z}_w) + \sum_{w' \notin \mathcal{W}} - \log (1 - \sigma(\mathbf{z}_v^\top \mathbf{z}_{w'}))
$$

## Node2Vec (Homogeneous Graphs)

```python
from torch_geometric.nn import Node2Vec

model = Node2Vec(
    data.edge_index,
    embedding_dim=128,     # Output embedding size
    walks_per_node=10,     # Random walks per node
    walk_length=20,        # Steps per walk
    context_size=10,       # Window for positive samples
    p=1.0,                 # Return parameter (BFS/DFS control)
    q=1.0,                 # In-out parameter
    num_negative_samples=1,
).to(device)

# Create training loader
loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss = {total_loss / len(loader):.4f}")

# Get embeddings
z = model()  # All node embeddings [N, 128]
z = model(torch.tensor([0, 1, 2]))  # Specific nodes
```

## Node2Vec Parameters

| Parameter | Description | Effect |
|-----------|-------------|--------|
| `p` | Return parameter | Low p → BFS-like (local structure) |
| `q` | In-out parameter | Low q → DFS-like (global structure) |
| `walk_length` | Steps per walk | Longer → more context |
| `context_size` | Window size | Larger → more positive pairs |

**Typical settings:**
- Local structure focus: `p=1, q=0.5`
- Global structure focus: `p=1, q=2`
- Balanced: `p=1, q=1`

## MetaPath2Vec (Heterogeneous Graphs)

For heterogeneous graphs, use metapath-guided random walks:

```python
from torch_geometric.nn import MetaPath2Vec

# Define metapath (cycle through node types)
metapath = [
    ('empresa', 'ubicado_en', 'distrito'),
    ('distrito', 'rev_ubicado_en', 'empresa'),
    ('empresa', 'pertenece_a', 'sector'),
    ('sector', 'rev_pertenece_a', 'empresa'),
]

model = MetaPath2Vec(
    data.edge_index_dict,  # Dict of edge indices
    embedding_dim=128,
    metapath=metapath,
    walk_length=20,
    context_size=7,
    walks_per_node=5,
    num_negative_samples=5,
).to(device)

# Training is similar to Node2Vec
loader = model.loader(batch_size=128, shuffle=True)
# ...

# Get embeddings for specific node type
z = model('empresa')  # All empresa embeddings
```

## Using Embeddings for Downstream Tasks

### As GNN Input Features

```python
# Train Node2Vec first
node2vec_model.train()
# ... training ...

# Get embeddings
with torch.no_grad():
    embeddings = node2vec_model()

# Use as features for GNN
data.x = embeddings  # Replace or concatenate with existing features

# Train GNN
gnn_model = GCN(embeddings.size(1), 64, num_classes)
```

### For Node Classification

```python
# Train shallow embeddings
z = model()

# Train simple classifier on top
classifier = torch.nn.Linear(128, num_classes)
optimizer = torch.optim.Adam(classifier.parameters())

for epoch in range(100):
    classifier.train()
    out = classifier(z[train_mask])
    loss = F.cross_entropy(out, y[train_mask])
    loss.backward()
    optimizer.step()
```

### For Link Prediction

```python
# Edge representations via operations on node embeddings
def edge_score(z, edge_index):
    src, dst = edge_index
    # Options: dot product, Hadamard, concatenation
    return (z[src] * z[dst]).sum(dim=1)  # Dot product
```

## Decision Tree

```
Do you have rich node features?
├── YES → Consider GNNs directly
└── NO → Use shallow embeddings
    └── Is your graph heterogeneous?
        ├── NO → Use Node2Vec
        └── YES → Use MetaPath2Vec
            └── Design metapath to capture semantic relationships
```

## References

- `06d_shallow_node_embeddings.md` - Complete tutorial
- `17_torch_geometric_nn.md` - Node2Vec/MetaPath2Vec API
