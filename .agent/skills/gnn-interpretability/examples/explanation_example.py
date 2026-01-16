"""
GNN Explanation Example

This example demonstrates how to explain GNN predictions using 
PyG's Explainer framework. Shows which nodes, edges, and features
are most important for a prediction.

Use Case: Understanding why an MSME is predicted to survive or fail.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.explain import Explainer, GNNExplainer, AttentionExplainer


# ===========================================================
# Models
# ===========================================================

class GCNClassifier(torch.nn.Module):
    """Simple 2-layer GCN for node classification."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATClassifier(torch.nn.Module):
    """2-layer GAT for node classification (attention can be extracted)."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ===========================================================
# Training
# ===========================================================

def train_model(model, data, epochs: int = 200):
    """Train a GNN model for node classification."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    
    print(f"Test Accuracy: {test_acc:.4f}")
    return model


# ===========================================================
# Explanation
# ===========================================================

def explain_node_gnnexplainer(model, data, node_idx: int):
    """
    Explain a single node's prediction using GNNExplainer.
    
    Returns explanation with node_mask (feature importance) 
    and edge_mask (edge importance).
    """
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',  # Explain feature importance
        edge_mask_type='object',      # Explain edge importance
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',  # or 'log_probs', 'probs'
        ),
    )
    
    explanation = explainer(
        data.x, 
        data.edge_index, 
        index=node_idx
    )
    
    return explanation


def explain_node_attention(model, data, node_idx: int):
    """
    Explain a GAT model's prediction using attention weights.
    
    Faster than GNNExplainer since it just extracts existing attention.
    """
    
    explainer = Explainer(
        model=model,
        algorithm=AttentionExplainer(),
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    
    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_idx
    )
    
    return explanation


def analyze_explanation(explanation, data, node_idx: int, top_k: int = 5):
    """Analyze and print explanation results."""
    
    print(f"\n{'='*50}")
    print(f"Explanation for Node {node_idx}")
    print(f"{'='*50}")
    
    # Ground truth and prediction
    true_label = data.y[node_idx].item()
    print(f"True label: {true_label}")
    
    # Feature importance (if available)
    if explanation.node_mask is not None:
        feature_importance = explanation.node_mask
        
        # Get top-k important features
        top_features = feature_importance.topk(min(top_k, len(feature_importance)))
        
        print(f"\nTop {top_k} Important Features:")
        for i, (value, idx) in enumerate(zip(top_features.values, top_features.indices)):
            print(f"  {i+1}. Feature {idx.item()}: {value.item():.4f}")
    
    # Edge importance (if available)
    if explanation.edge_mask is not None:
        edge_mask = explanation.edge_mask
        edge_index = explanation.edge_index
        
        # Get top-k important edges
        top_edges = edge_mask.topk(min(top_k, len(edge_mask)))
        
        print(f"\nTop {top_k} Important Edges:")
        for i, (value, idx) in enumerate(zip(top_edges.values, top_edges.indices)):
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            print(f"  {i+1}. Edge ({src} -> {dst}): {value.item():.4f}")
    
    return explanation


def batch_explain(model, data, node_indices: list[int]):
    """Explain multiple nodes and aggregate results."""
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )
    
    explanations = []
    feature_importances = []
    
    for node_idx in node_indices:
        exp = explainer(data.x, data.edge_index, index=node_idx)
        explanations.append(exp)
        
        if exp.node_mask is not None:
            feature_importances.append(exp.node_mask)
    
    # Aggregate feature importance
    if feature_importances:
        avg_importance = torch.stack(feature_importances).mean(dim=0)
        top_features = avg_importance.topk(10)
        
        print("\nAggregated Top 10 Features Across All Explained Nodes:")
        for i, (value, idx) in enumerate(zip(top_features.values, top_features.indices)):
            print(f"  {i+1}. Feature {idx.item()}: {value.item():.4f}")
    
    return explanations


# ===========================================================
# Main
# ===========================================================

def main():
    # Load dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    print(f"Dataset: Cora")
    print(f"Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {dataset.num_classes}")
    
    # Train GCN model
    print("\n--- Training GCN Model ---")
    gcn_model = GCNClassifier(data.num_features, 64, dataset.num_classes)
    gcn_model = train_model(gcn_model, data, epochs=200)
    
    # Explain single node with GNNExplainer
    print("\n--- GNNExplainer on Node 42 ---")
    explanation = explain_node_gnnexplainer(gcn_model, data, node_idx=42)
    analyze_explanation(explanation, data, node_idx=42, top_k=5)
    
    # Train GAT model for attention-based explanation
    print("\n--- Training GAT Model ---")
    gat_model = GATClassifier(data.num_features, 8, dataset.num_classes, heads=4)
    gat_model = train_model(gat_model, data, epochs=200)
    
    # Explain with attention
    print("\n--- AttentionExplainer on Node 42 ---")
    attention_exp = explain_node_attention(gat_model, data, node_idx=42)
    analyze_explanation(attention_exp, data, node_idx=42, top_k=5)
    
    # Batch explanation
    print("\n--- Batch Explanation (10 nodes) ---")
    test_nodes = data.test_mask.nonzero().squeeze()[:10].tolist()
    batch_explain(gcn_model, data, test_nodes)


if __name__ == "__main__":
    main()
