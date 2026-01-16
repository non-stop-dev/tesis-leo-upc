---
name: gnn-interpretability
description: Expertise in using PyG's Explainability framework to understand why the model predicts specific outcomes. Use when you need to visualize which neighbors or features affect survival predictions.
---

# GNN Interpretability Skill

This skill provides expertise for explaining GNN predictions using PyG's Explainer framework. Essential for understanding which graph features and neighbor relationships drive the survival classification.

## When to use this skill

- You need to explain why an MSME was predicted to survive or fail
- You want to visualize which neighbors are most influential
- You need to identify important node features
- You're presenting model insights to stakeholders

## Core Concept: GNN Explanations

GNN explanations answer: **"Why did the model predict X for this node?"**

Two main explanation types:
1. **Feature importance**: Which node features matter?
2. **Edge importance**: Which neighbors/connections matter?

## The Explainer Framework

```python
from torch_geometric.explain import Explainer, GNNExplainer

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

# Explain a specific node
explanation = explainer(data.x, data.edge_index, index=node_idx)
```

## Key Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `explanation_type` | `'model'`, `'phenomenon'` | What to explain |
| `node_mask_type` | `'attributes'`, `'object'`, `None` | Feature masking |
| `edge_mask_type` | `'object'`, `None` | Edge masking |
| `task_level` | `'node'`, `'edge'`, `'graph'` | Prediction level |

## Available Algorithms

1. **GNNExplainer** - Original perturbation-based method
2. **PGExplainer** - Faster, learnable masks
3. **CaptumExplainer** - Gradient-based (IntegratedGradients, etc.)
4. **AttentionExplainer** - Uses attention weights from GAT

```python
from torch_geometric.explain import GNNExplainer, PGExplainer, AttentionExplainer

# Choose your algorithm
algorithm = GNNExplainer(epochs=200)
# or
algorithm = AttentionExplainer()  # For GAT models
```

## Accessing Explanation Results

```python
# Run explanation
explanation = explainer(data.x, data.edge_index, index=42)

# Node/feature importance mask
print(explanation.node_mask)  # [num_features] importance scores

# Edge importance mask  
print(explanation.edge_mask)  # [num_edges] importance scores

# Get the k-hop subgraph around the explained node
print(explanation.edge_index)  # Edges in explanation
print(explanation.x)  # Node features
```

## Visualization

```python
import matplotlib.pyplot as plt

# Visualize edge importance
explanation.visualize_graph(
    path='explanation.png',
    backend='graphviz'  # or 'networkx'
)

# Visualize feature importance
explanation.visualize_feature_importance(
    path='feature_importance.png',
    top_k=10,
)
```

## Batch Explanations

For explaining multiple nodes:

```python
from torch_geometric.explain import Explanation

explanations = []
for node_idx in [0, 10, 42, 100]:
    exp = explainer(data.x, data.edge_index, index=node_idx)
    explanations.append(exp)

# Aggregate importance across explanations
avg_edge_mask = torch.stack([e.edge_mask for e in explanations]).mean(dim=0)
```

## Decision Tree

```
What do you want to explain?
├── Single node prediction → Use Explainer with index=node_idx
├── Multiple nodes → Loop over indices or use batch explain
├── GAT attention → Use AttentionExplainer (fastest)
└── Feature importance → Set node_mask_type='attributes'

What algorithm to use?
├── Need interpretable subgraph → GNNExplainer
├── Need speed for many explains → PGExplainer (train once)
├── Have GAT model → AttentionExplainer
└── Want gradient-based → CaptumExplainer
```

## Use Case: MSME Survival Analysis

For understanding MSME survival predictions:

1. **Which competitors affect survival?** → Look at edge masks for `compite_con` edges
2. **Does district matter?** → Compare edge masks for `ubicado_en` edges
3. **What features predict survival?** → Examine node_mask (ventas, productividad, etc.)

Example interpretation:
> "MSME #42 was predicted to fail because of high competition (edge_mask shows strong connections to 3 failing competitors) and low productivity (feature_mask highlights productividad_k)."

## References

- `06a_gnn_explainability.md` - Tutorial on explanations
- `25_torch_geometric_explain.md` - Full Explainer API
