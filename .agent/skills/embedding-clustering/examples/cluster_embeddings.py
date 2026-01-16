"""
GNN Embedding Clustering Example

Demonstrates "Methodological Clustering" on GNN latent embeddings:
1. Train a GNN for survival prediction
2. Extract node embeddings from hidden layers
3. Apply clustering (K-Means)
4. Analyze survival patterns per cluster
5. Compare to administrative regions

Based on GEMINI.md: "Methodological Clustering" goal
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, 
    adjusted_rand_score,
    normalized_mutual_info_score
)
import numpy as np
import matplotlib.pyplot as plt


# ===========================================================
# Model with Embedding Extraction
# ===========================================================

class GNNWithEmbeddings(torch.nn.Module):
    """GNN that exposes intermediate embeddings."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        """Get latent embeddings (before classifier)."""
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.classifier(z)


# ===========================================================
# Data Creation
# ===========================================================

def create_synthetic_data(num_nodes: int = 1000) -> Data:
    """Create synthetic MSME-like data with regions."""
    
    # Features: ventas, productividad, etc.
    x = torch.randn(num_nodes, 8)
    
    # Survival labels (imbalanced)
    y = (torch.rand(num_nodes) > 0.3).long()
    
    # Administrative regions (Coast=0, Sierra=1, Selva=2)
    regions = torch.randint(0, 3, (num_nodes,))
    
    # Create edges (same-region connections)
    edges = []
    for i in range(num_nodes):
        same_region = (regions == regions[i]).nonzero().view(-1)
        neighbors = same_region[torch.randperm(len(same_region))[:5]]
        for j in neighbors:
            if i != j:
                edges.append([i, j.item()])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Train/test masks
    perm = torch.randperm(num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:int(0.8 * num_nodes)]] = True
    test_mask[perm[int(0.8 * num_nodes):]] = True
    
    return Data(
        x=x,
        y=y,
        edge_index=edge_index,
        regions=regions,
        train_mask=train_mask,
        test_mask=test_mask,
    )


# ===========================================================
# Training
# ===========================================================

def train_model(model, data, epochs: int = 100):
    """Train GNN for survival prediction."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(epochs):
        model.train()
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
# Clustering Analysis
# ===========================================================

def find_optimal_k(embeddings: np.ndarray, k_range: range = range(2, 15)) -> int:
    """Find optimal number of clusters using silhouette score."""
    
    scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores.append((k, score))
        print(f"  K={k}: silhouette={score:.4f}")
    
    best_k = max(scores, key=lambda x: x[1])[0]
    return best_k


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Cluster embeddings using K-Means."""
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)


def analyze_clusters(data: Data, cluster_labels: np.ndarray):
    """Analyze survival rates and characteristics per cluster."""
    
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS")
    print("="*50)
    
    unique_clusters = sorted(set(cluster_labels))
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        count = mask.sum()
        survival_rate = data.y[mask].float().mean().item()
        
        # Region distribution
        region_names = ['Coast', 'Sierra', 'Selva']
        region_dist = [
            (data.regions[mask] == r).sum().item() / count * 100
            for r in range(3)
        ]
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {count} empresas ({count/len(data.y)*100:.1f}%)")
        print(f"  Survival Rate: {survival_rate*100:.1f}%")
        print(f"  Region Distribution:")
        for i, name in enumerate(region_names):
            print(f"    {name}: {region_dist[i]:.1f}%")
    
    return cluster_labels


def compare_to_regions(cluster_labels: np.ndarray, region_labels: np.ndarray):
    """Compare discovered clusters to administrative regions."""
    
    print("\n" + "="*50)
    print("COMPARISON TO ADMINISTRATIVE REGIONS")
    print("="*50)
    
    ari = adjusted_rand_score(region_labels, cluster_labels)
    nmi = normalized_mutual_info_score(region_labels, cluster_labels)
    
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    
    print("\nInterpretation:")
    if ari < 0.3:
        print("  → LOW ARI: Clusters reveal NEW patterns beyond regions!")
        print("     This suggests GNN discovered business environments")
        print("     that don't align with Coast/Sierra/Selva boundaries.")
    elif ari < 0.6:
        print("  → MODERATE ARI: Partial alignment with regions")
        print("     Some clusters cross regional boundaries.")
    else:
        print("  → HIGH ARI: Clusters align with regions")
        print("     GNN embeddings reflect administrative structure.")
    
    return ari, nmi


def visualize_embeddings(
    embeddings: np.ndarray, 
    cluster_labels: np.ndarray,
    survival_labels: np.ndarray,
    region_labels: np.ndarray,
    save_path: str = 'embedding_analysis.png'
):
    """Create visualization of embedding space."""
    
    print("\nGenerating t-SNE visualization...")
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(embeddings)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # By discovered cluster
    scatter1 = axes[0].scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=cluster_labels, cmap='tab10', s=5, alpha=0.7
    )
    axes[0].set_title('Discovered Clusters')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # By survival
    scatter2 = axes[1].scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=survival_labels, cmap='RdYlGn', s=5, alpha=0.7
    )
    axes[1].set_title('Survival Status')
    plt.colorbar(scatter2, ax=axes[1], label='Survived')
    
    # By administrative region
    scatter3 = axes[2].scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=region_labels, cmap='Set1', s=5, alpha=0.7
    )
    axes[2].set_title('Administrative Regions')
    plt.colorbar(scatter3, ax=axes[2], label='Region')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


# ===========================================================
# Main Pipeline
# ===========================================================

def main():
    print("="*60)
    print("GNN EMBEDDING CLUSTERING FOR MSME SURVIVAL ANALYSIS")
    print("="*60)
    
    # 1. Create data
    print("\n[1] Creating synthetic MSME data...")
    data = create_synthetic_data(num_nodes=1000)
    print(f"    Nodes: {data.x.size(0)}, Edges: {data.edge_index.size(1)}")
    
    # 2. Train GNN
    print("\n[2] Training GNN for survival prediction...")
    model = GNNWithEmbeddings(data.x.size(1), 64, 2)
    model = train_model(model, data, epochs=100)
    
    # 3. Extract embeddings
    print("\n[3] Extracting latent embeddings...")
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index).cpu().numpy()
    print(f"    Embedding shape: {embeddings.shape}")
    
    # 4. Find optimal K
    print("\n[4] Finding optimal number of clusters...")
    optimal_k = find_optimal_k(embeddings)
    print(f"    Optimal K: {optimal_k}")
    
    # 5. Cluster
    print(f"\n[5] Clustering with K={optimal_k}...")
    cluster_labels = cluster_embeddings(embeddings, optimal_k)
    
    # 6. Analyze
    analyze_clusters(data, cluster_labels)
    
    # 7. Compare to regions
    compare_to_regions(cluster_labels, data.regions.numpy())
    
    # 8. Visualize
    print("\n[8] Visualizing...")
    try:
        visualize_embeddings(
            embeddings,
            cluster_labels,
            data.y.numpy(),
            data.regions.numpy(),
        )
    except Exception as e:
        print(f"    Visualization skipped: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
