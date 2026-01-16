---
name: link-prediction-analysis
description: Link prediction for analyzing edge importance and comparing edge types. Use when you need to understand which relationships (geographic, sectoral, competition) are most predictive of outcomes.
---

# Link Prediction & Edge Analysis Skill

This skill provides expertise for link prediction tasks and analyzing the importance of different edge types in heterogeneous graphs.

## When to use this skill

- Analyzing which edge types are most predictive (Edge Sensitivity Analysis)
- Predicting missing edges or future connections
- Comparing geographic vs sectoral vs competition edges
- Understanding graph structure's impact on node outcomes

## Edge Sensitivity Analysis Framework

For your MSME survival prediction, the question is:
> *"Which edge type (ubicado_en, pertenece_a, compite_con) is most determinant for survival?"*

### Approach 1: Ablation Study

Train models with different edge subsets and compare performance:

```python
def ablation_experiment(data, edge_types_to_include):
    """Train model with subset of edge types."""
    
    # Filter to only specified edge types
    filtered_data = filter_edges(data, edge_types_to_include)
    
    # Train and evaluate
    model = train_model(filtered_data)
    return evaluate(model, filtered_data)

# Compare edge type importance
results = {}
results['all'] = ablation_experiment(data, ['ubicado_en', 'pertenece_a', 'compite_con'])
results['geo_only'] = ablation_experiment(data, ['ubicado_en'])
results['sector_only'] = ablation_experiment(data, ['pertenece_a'])
results['competition_only'] = ablation_experiment(data, ['compite_con'])
results['no_geo'] = ablation_experiment(data, ['pertenece_a', 'compite_con'])
results['no_sector'] = ablation_experiment(data, ['ubicado_en', 'compite_con'])
```

### Approach 2: Edge Attention Weights

Use GAT/GATv2 and analyze attention distributions per edge type:

```python
from torch_geometric.nn import GATConv

class AttentionAnalysisGAT(torch.nn.Module):
    def __init__(self, ...):
        self.conv = GATConv(..., return_attention_weights=True)
    
    def forward(self, x, edge_index):
        out, (edge_index, attention_weights) = self.conv(
            x, edge_index, return_attention_weights=True
        )
        return out, attention_weights

# After training, analyze attention by edge type
model.eval()
_, attention = model(data.x, data.edge_index)

# Group attention by edge type
geo_attention = attention[geo_edge_mask].mean()
sector_attention = attention[sector_edge_mask].mean()
competition_attention = attention[competition_edge_mask].mean()
```

## Link Prediction Task

For predicting edges (e.g., will empresa A compete with empresa B?):

### Data Preparation

```python
import torch_geometric.transforms as T

# Split edges into train/val/test
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=1.0,  # Equal negative samples
    add_negative_train_samples=True,
)

train_data, val_data, test_data = transform(data)
```

### LinkNeighborLoader

```python
from torch_geometric.loader import LinkNeighborLoader

loader = LinkNeighborLoader(
    data,
    num_neighbors=[20, 10],
    edge_label_index=train_data.edge_label_index,
    edge_label=train_data.edge_label,
    batch_size=256,
    shuffle=True,
)
```

### Link Prediction Model

```python
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = GCN(in_channels, hidden_channels, hidden_channels)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )
    
    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_label_index):
        src, dst = edge_label_index
        # Concatenate source and destination embeddings
        edge_feat = torch.cat([z[src], z[dst]], dim=-1)
        return self.decoder(edge_feat).squeeze()
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
```

### Training Loop

```python
def train_link_prediction(model, loader, optimizer):
    model.train()
    total_loss = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Encode using message passing edges
        z = model.encode(batch.x, batch.edge_index)
        
        # Predict on supervision edges
        pred = model.decode(z, batch.edge_label_index)
        
        # Binary cross entropy
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)
```

## Metrics for Link Prediction

```python
from sklearn.metrics import roc_auc_score, average_precision_score

@torch.no_grad()
def evaluate_link_prediction(model, data):
    model.eval()
    
    z = model.encode(data.x, data.edge_index)
    pred = model.decode(z, data.edge_label_index).sigmoid()
    
    y_true = data.edge_label.cpu().numpy()
    y_pred = pred.cpu().numpy()
    
    return {
        'auroc': roc_auc_score(y_true, y_pred),
        'ap': average_precision_score(y_true, y_pred),
    }
```

## Heterogeneous Link Prediction

For HeteroData with multiple edge types:

```python
from torch_geometric.transforms import RandomLinkSplit

# Split specific edge type
transform = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    edge_types=[('empresa', 'compite_con', 'empresa')],
    rev_edge_types=[('empresa', 'compite_con', 'empresa')],
)

train_data, val_data, test_data = transform(data)
```

## Edge Type Comparison Framework

```python
def compare_edge_types(data, model_class, edge_types):
    """
    Compare predictive power of different edge types.
    
    Returns dict mapping edge_type -> metrics
    """
    results = {}
    
    for edge_type in edge_types:
        # Create single-edge-type graph
        single_edge_data = create_single_edge_graph(data, edge_type)
        
        # Train model
        model = model_class(...)
        train(model, single_edge_data)
        
        # Evaluate
        metrics = evaluate(model, single_edge_data)
        results[edge_type] = metrics
        
        print(f"{edge_type}: AUROC={metrics['auroc']:.4f}")
    
    return results
```

## Decision Tree

```
What do you want to analyze?
├── Edge importance for node prediction → Ablation study
├── Which edges a model uses → Attention analysis
├── Predict missing edges → Link prediction task
└── Compare edge types → Train separate models per type
```

## References

- `19_torch_geometric_loader.md` - LinkNeighborLoader API
- `23_torch_geometric_transforms.md` - RandomLinkSplit
- `04b_heterogeneous_graph_learning.md` - Hetero link prediction
