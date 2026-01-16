---
name: embedding-clustering
description: Clustering on GNN latent embeddings. Allows to cluster node embeddings to analyze business environments and graph dynamics. Integrates GNN with sklearn clustering algorithms.
---

# Embedding Clustering Skill

This skill provides expertise for clustering GNN node embeddings to discover hidden patterns and business environment groupings.

## When to use this skill

- Discovering endogenous business clusters beyond administrative regions
- Analyzing latent space structure of trained GNNs
- "Methodological Clustering" on embeddings (as per GEMINI.md)
- Comparing survival dynamics across discovered clusters
- Visualizing embedding space

## Core Workflow

```
1. Train GNN → 2. Extract Embeddings → 3. Cluster → 4. Analyze
```

## Step 1: Extract Node Embeddings

```python
class GNNWithEmbeddings(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        """Return node embeddings before final classifier."""
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index).relu()
        return x  # Latent embeddings
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return self.classifier(z)

# After training, extract embeddings
model.eval()
with torch.no_grad():
    embeddings = model.encode(data.x, data.edge_index)
```

## Step 2: Clustering Algorithms

### K-Means

```python
from sklearn.cluster import KMeans

# Cluster embeddings
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings.cpu().numpy())

# Add to data
data.cluster = torch.tensor(cluster_labels)
```

### Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='ward',
)
cluster_labels = clustering.fit_predict(embeddings.cpu().numpy())
```

### DBSCAN (Density-Based)

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=10)
cluster_labels = dbscan.fit_predict(embeddings.cpu().numpy())

# Note: -1 indicates noise points
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
```

## Step 3: Choosing Number of Clusters

### Elbow Method

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertias = []
K_range = range(2, 15)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings.cpu().numpy())
    inertias.append(kmeans.inertia_)

plt.plot(K_range, inertias, 'bx-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
```

### Silhouette Score

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings.cpu().numpy())
    score = silhouette_score(embeddings.cpu().numpy(), labels)
    scores.append((k, score))

best_k = max(scores, key=lambda x: x[1])[0]
print(f"Optimal clusters: {best_k}")
```

## Step 4: Cluster Analysis

### Survival Rate by Cluster

```python
def analyze_survival_by_cluster(data, cluster_labels):
    """Analyze survival rate per cluster."""
    
    results = {}
    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        survival_rate = data.y[mask].float().mean().item()
        count = mask.sum().item()
        
        results[cluster_id] = {
            'count': count,
            'survival_rate': survival_rate,
        }
        
    return results

# Example output:
# Cluster 0: 234 empresas, 78% survival
# Cluster 1: 189 empresas, 45% survival  <- High risk cluster
# Cluster 2: 312 empresas, 82% survival
```

### Cluster Characteristics

```python
def cluster_feature_profile(data, cluster_labels, feature_names):
    """Get mean feature values per cluster."""
    
    profiles = {}
    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        mean_features = data.x[mask].mean(dim=0)
        
        profiles[cluster_id] = {
            name: mean_features[i].item()
            for i, name in enumerate(feature_names)
        }
    
    return profiles

feature_names = ['ventas', 'productividad', 'tributos', 'empleados', ...]
profiles = cluster_feature_profile(data, cluster_labels, feature_names)
```

### Geographic Distribution

```python
def cluster_geographic_distribution(cluster_labels, region_labels):
    """Analyze how clusters map to administrative regions."""
    
    from collections import Counter
    
    for cluster_id in set(cluster_labels):
        mask = cluster_labels == cluster_id
        cluster_regions = region_labels[mask]
        
        distribution = Counter(cluster_regions.tolist())
        print(f"Cluster {cluster_id}:")
        for region, count in distribution.most_common(3):
            print(f"  {region}: {count}")
```

## Visualization

### t-SNE Embedding Plot

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_clusters(embeddings, cluster_labels, survival_labels=None):
    """Visualize embeddings colored by cluster and survival."""
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(embeddings.cpu().numpy())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # By cluster
    scatter1 = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', s=5, alpha=0.7)
    axes[0].set_title('Colored by Cluster')
    plt.colorbar(scatter1, ax=axes[0])
    
    # By survival
    if survival_labels is not None:
        scatter2 = axes[1].scatter(z_2d[:, 0], z_2d[:, 1],
                                    c=survival_labels.cpu().numpy(), 
                                    cmap='RdYlGn', s=5, alpha=0.7)
        axes[1].set_title('Colored by Survival')
        plt.colorbar(scatter2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('embedding_clusters.png', dpi=150)
    plt.show()
```

## Comparing Clusters to Administrative Regions

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compare_to_regions(cluster_labels, region_labels):
    """
    Compare discovered clusters to administrative regions.
    
    High ARI = clusters align with regions
    Low ARI = clusters reveal new patterns
    """
    
    ari = adjusted_rand_score(region_labels, cluster_labels)
    nmi = normalized_mutual_info_score(region_labels, cluster_labels)
    
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Info: {nmi:.4f}")
    
    if ari < 0.3:
        print("→ Clusters reveal patterns beyond administrative regions!")
    else:
        print("→ Clusters align with administrative structure")
    
    return ari, nmi
```

## Complete Pipeline

```python
def embedding_clustering_pipeline(model, data, n_clusters=5):
    """Complete pipeline for embedding-based clustering."""
    
    # 1. Extract embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data.x, data.edge_index)
    
    # 2. Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings.cpu().numpy())
    
    # 3. Analyze
    survival_by_cluster = analyze_survival_by_cluster(data, cluster_labels)
    
    # 4. Visualize
    visualize_clusters(embeddings, cluster_labels, data.y)
    
    return cluster_labels, survival_by_cluster
```

## Decision Tree

```
What do you want to discover?
├── Distinct business environments → K-Means clustering
├── Natural groupings without K → DBSCAN
├── Hierarchical structure → Agglomerative clustering
└── Compare to known regions → Compute ARI/NMI

How many clusters?
├── Unknown → Use elbow method or silhouette score
├── Match regions → Use number of regions
└── Exploratory → Try K=3,5,7,10 and compare
```

## References

- `06d_shallow_node_embeddings.md` - Unsupervised embeddings
- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
