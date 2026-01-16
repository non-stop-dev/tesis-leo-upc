# Documentation Resources

This file links to the PyG documentation for GNN explainability.

## Primary Reference

- **06a_gnn_explainability.md** - Complete GNN Explainability tutorial

## Key Concepts

### Explainer Framework

PyG's `Explainer` class provides a unified interface for explanation methods:

```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',           # or 'phenomenon'
    node_mask_type='attributes',         # explain feature importance
    edge_mask_type='object',             # explain edge importance
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',               # or 'edge', 'graph'
        return_type='raw',               # or 'log_probs', 'probs'
    ),
)
```

### Explanation Types

| Type | Description |
|------|-------------|
| `model` | Explain model behavior (what does the model look at?) |
| `phenomenon` | Explain the underlying phenomenon (what causes the label?) |

### Mask Types

| Mask | Description |
|------|-------------|
| `node_mask_type='attributes'` | Feature-level importance |
| `node_mask_type='object'` | Node-level importance |
| `edge_mask_type='object'` | Edge-level importance |

### Available Algorithms

1. **GNNExplainer** - Perturbation-based masks
2. **PGExplainer** - Learnable masks (train once, explain many)
3. **AttentionExplainer** - Extract attention weights (for GAT)
4. **CaptumExplainer** - Gradient-based (IntegratedGradients, etc.)

### Visualization

```python
# Visualize subgraph explanation
explanation.visualize_graph(path='explanation.png', backend='networkx')

# Visualize feature importance
explanation.visualize_feature_importance(path='features.png', top_k=10)
```
