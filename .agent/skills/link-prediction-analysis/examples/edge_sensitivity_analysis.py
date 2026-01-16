"""
Edge Sensitivity Analysis Example

Demonstrates how to analyze which edge types (geographic, sectoral, competition)
are most important for MSME survival prediction through:
1. Ablation studies - remove edge types and measure impact
2. Attention analysis - examine GAT attention weights by edge type

Based on GEMINI.md: "Edge Sensitivity Analysis"
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv
from sklearn.metrics import accuracy_score, roc_auc_score
from collections import defaultdict
import numpy as np


# ===========================================================
# Models
# ===========================================================

class FlexibleGNN(torch.nn.Module):
    """GNN that can work with different edge subsets."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class AttentionGNN(torch.nn.Module):
    """GAT model for attention weight analysis."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    
    def forward(self, x, edge_index, return_attention: bool = False):
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        
        if return_attention:
            return x, attn1, attn2
        return x


# ===========================================================
# Data Utilities
# ===========================================================

def create_synthetic_msme_graph(num_empresas: int = 1000):
    """
    Create synthetic MSME graph with multiple edge types.
    
    Edge types:
    - ubicado_en: empresa → distrito (geographic)
    - pertenece_a: empresa → sector (sectoral)  
    - compite_con: empresa ↔ empresa (competition)
    """
    
    # Node features (ventas, productividad, etc.)
    x = torch.randn(num_empresas, 8)
    
    # Survival labels (imbalanced: 70% survive)
    y = (torch.rand(num_empresas) > 0.3).long()
    
    # Create edge types
    num_distritos = 50
    num_sectores = 20
    
    # Geographic edges: each empresa in one distrito
    distrito_assignment = torch.randint(0, num_distritos, (num_empresas,))
    geo_edges = []
    for i in range(num_empresas):
        # Connect to empresas in same distrito
        same_distrito = (distrito_assignment == distrito_assignment[i]).nonzero().view(-1)
        for j in same_distrito[:5]:  # Max 5 neighbors
            if i != j:
                geo_edges.append([i, j.item()])
    
    # Sectoral edges: each empresa in one sector
    sector_assignment = torch.randint(0, num_sectores, (num_empresas,))
    sector_edges = []
    for i in range(num_empresas):
        same_sector = (sector_assignment == sector_assignment[i]).nonzero().view(-1)
        for j in same_sector[:5]:
            if i != j:
                sector_edges.append([i, j.item()])
    
    # Competition edges: random connections (same sector + distrito)
    competition_edges = []
    for i in range(num_empresas):
        same_both = ((distrito_assignment == distrito_assignment[i]) & 
                     (sector_assignment == sector_assignment[i])).nonzero().view(-1)
        for j in same_both[:3]:
            if i != j:
                competition_edges.append([i, j.item()])
    
    # Convert to tensors
    geo_edge_index = torch.tensor(geo_edges, dtype=torch.long).t().contiguous() if geo_edges else torch.zeros(2, 0, dtype=torch.long)
    sector_edge_index = torch.tensor(sector_edges, dtype=torch.long).t().contiguous() if sector_edges else torch.zeros(2, 0, dtype=torch.long)
    competition_edge_index = torch.tensor(competition_edges, dtype=torch.long).t().contiguous() if competition_edges else torch.zeros(2, 0, dtype=torch.long)
    
    # Combine all edges
    all_edge_index = torch.cat([geo_edge_index, sector_edge_index, competition_edge_index], dim=1)
    
    # Edge type labels
    edge_types = torch.cat([
        torch.zeros(geo_edge_index.size(1)),
        torch.ones(sector_edge_index.size(1)),
        torch.full((competition_edge_index.size(1),), 2),
    ]).long()
    
    # Train/test split
    perm = torch.randperm(num_empresas)
    train_mask = torch.zeros(num_empresas, dtype=torch.bool)
    test_mask = torch.zeros(num_empresas, dtype=torch.bool)
    train_mask[perm[:int(0.8 * num_empresas)]] = True
    test_mask[perm[int(0.8 * num_empresas):]] = True
    
    return Data(
        x=x,
        y=y,
        edge_index=all_edge_index,
        edge_types=edge_types,
        geo_edge_index=geo_edge_index,
        sector_edge_index=sector_edge_index,
        competition_edge_index=competition_edge_index,
        train_mask=train_mask,
        test_mask=test_mask,
    )


def filter_edges_by_type(data: Data, include_types: list[str]) -> Data:
    """Create new data with only specified edge types."""
    
    edge_indices = []
    if 'geo' in include_types:
        edge_indices.append(data.geo_edge_index)
    if 'sector' in include_types:
        edge_indices.append(data.sector_edge_index)
    if 'competition' in include_types:
        edge_indices.append(data.competition_edge_index)
    
    if not edge_indices:
        filtered_edge_index = torch.zeros(2, 0, dtype=torch.long)
    else:
        filtered_edge_index = torch.cat(edge_indices, dim=1)
    
    return Data(
        x=data.x,
        y=data.y,
        edge_index=filtered_edge_index,
        train_mask=data.train_mask,
        test_mask=data.test_mask,
    )


# ===========================================================
# Training & Evaluation
# ===========================================================

def train_model(model, data, epochs: int = 100, lr: float = 0.01):
    """Train a model on the given data."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    
    return model


@torch.no_grad()
def evaluate_model(model, data) -> dict:
    """Evaluate model performance."""
    
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    probs = F.softmax(out, dim=1)[:, 1]
    
    # Test set metrics
    test_pred = pred[data.test_mask].cpu().numpy()
    test_true = data.y[data.test_mask].cpu().numpy()
    test_probs = probs[data.test_mask].cpu().numpy()
    
    return {
        'accuracy': accuracy_score(test_true, test_pred),
        'auroc': roc_auc_score(test_true, test_probs) if len(np.unique(test_true)) > 1 else 0.5,
    }


# ===========================================================
# Ablation Study
# ===========================================================

def run_ablation_study(data: Data) -> dict:
    """
    Run ablation study to determine edge type importance.
    
    Compares performance when:
    - Using all edges
    - Using only geo edges
    - Using only sector edges
    - Using only competition edges
    - Removing each edge type
    """
    
    configurations = {
        'all': ['geo', 'sector', 'competition'],
        'geo_only': ['geo'],
        'sector_only': ['sector'],
        'competition_only': ['competition'],
        'no_geo': ['sector', 'competition'],
        'no_sector': ['geo', 'competition'],
        'no_competition': ['geo', 'sector'],
    }
    
    results = {}
    
    for name, include_types in configurations.items():
        print(f"\nConfiguration: {name}")
        
        # Filter data
        filtered_data = filter_edges_by_type(data, include_types)
        print(f"  Edges: {filtered_data.edge_index.size(1)}")
        
        # Train model
        model = FlexibleGNN(data.x.size(1), 64, 2)
        model = train_model(model, filtered_data, epochs=100)
        
        # Evaluate
        metrics = evaluate_model(model, filtered_data)
        results[name] = metrics
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
    
    return results


def analyze_ablation_results(results: dict):
    """Analyze ablation results to determine edge importance."""
    
    print("\n" + "="*50)
    print("EDGE SENSITIVITY ANALYSIS RESULTS")
    print("="*50)
    
    baseline = results['all']['auroc']
    
    # Impact of removing each edge type
    print("\nImpact of Removing Edge Types (AUROC drop):")
    for edge_type in ['geo', 'sector', 'competition']:
        key = f'no_{edge_type}'
        drop = baseline - results[key]['auroc']
        print(f"  Without {edge_type}: {drop:+.4f} ({'worse' if drop > 0 else 'better'})")
    
    # Standalone value of each edge type
    print("\nStandalone Predictive Power (AUROC):")
    for edge_type in ['geo', 'sector', 'competition']:
        key = f'{edge_type}_only'
        auroc = results[key]['auroc']
        print(f"  {edge_type} only: {auroc:.4f}")
    
    # Ranking
    standalone = {et: results[f'{et}_only']['auroc'] for et in ['geo', 'sector', 'competition']}
    ranked = sorted(standalone.items(), key=lambda x: x[1], reverse=True)
    
    print("\nRanking (most to least predictive):")
    for i, (et, auroc) in enumerate(ranked, 1):
        print(f"  {i}. {et}: {auroc:.4f}")


# ===========================================================
# Attention Analysis
# ===========================================================

def analyze_attention_by_edge_type(model: AttentionGNN, data: Data):
    """Analyze GAT attention weights grouped by edge type."""
    
    model.eval()
    with torch.no_grad():
        _, (edge_index, attn1), (_, attn2) = model(
            data.x, data.edge_index, return_attention=True
        )
    
    # Average attention across heads
    attn1_avg = attn1.mean(dim=1) if attn1.dim() > 1 else attn1
    
    # Group by edge type
    edge_type_names = ['geo', 'sector', 'competition']
    
    print("\nAttention Analysis by Edge Type:")
    for i, name in enumerate(edge_type_names):
        mask = data.edge_types == i
        if mask.sum() > 0:
            type_attention = attn1_avg[mask]
            print(f"  {name}:")
            print(f"    Mean attention: {type_attention.mean():.4f}")
            print(f"    Std attention: {type_attention.std():.4f}")
            print(f"    Max attention: {type_attention.max():.4f}")


# ===========================================================
# Main
# ===========================================================

def main():
    print("Edge Sensitivity Analysis for MSME Survival")
    print("="*50)
    
    # Create synthetic data
    print("\nCreating synthetic MSME graph...")
    data = create_synthetic_msme_graph(num_empresas=1000)
    
    print(f"Nodes: {data.x.size(0)}")
    print(f"Total edges: {data.edge_index.size(1)}")
    print(f"  Geographic: {data.geo_edge_index.size(1)}")
    print(f"  Sectoral: {data.sector_edge_index.size(1)}")
    print(f"  Competition: {data.competition_edge_index.size(1)}")
    
    # Run ablation study
    print("\n" + "="*50)
    print("ABLATION STUDY")
    print("="*50)
    
    results = run_ablation_study(data)
    analyze_ablation_results(results)
    
    # Attention analysis
    print("\n" + "="*50)
    print("ATTENTION ANALYSIS (GAT)")
    print("="*50)
    
    gat_model = AttentionGNN(data.x.size(1), 16, 2, heads=4)
    gat_model = train_model(gat_model, data, epochs=100)
    analyze_attention_by_edge_type(gat_model, data)


if __name__ == "__main__":
    main()
