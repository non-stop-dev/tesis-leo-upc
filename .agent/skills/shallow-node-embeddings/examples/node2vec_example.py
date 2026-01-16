"""
Node2Vec Shallow Embeddings Example

Demonstrates learning unsupervised node embeddings using Node2Vec.
Useful when:
- Graph lacks node features
- Pretaining embeddings before GNN training
- Need embeddings for link prediction

Based on: 06d_shallow_node_embeddings.md
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def train_node2vec(data, embedding_dim: int = 128, epochs: int = 100):
    """
    Train Node2Vec embeddings on a graph.
    
    Args:
        data: PyG Data object
        embedding_dim: Dimension of embeddings
        epochs: Training epochs
    
    Returns:
        Trained Node2Vec model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Node2Vec(
        data.edge_index,
        embedding_dim=embedding_dim,
        walks_per_node=10,
        walk_length=20,
        context_size=10,
        p=1.0,     # Return parameter
        q=1.0,     # In-out parameter (q<1: DFS, q>1: BFS)
        num_negative_samples=1,
    ).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("Training Node2Vec...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {total_loss / len(loader):.4f}")
    
    return model


@torch.no_grad()
def evaluate_embeddings(model, data, device):
    """
    Evaluate Node2Vec embeddings on node classification.
    
    Uses the embeddings as input to a simple linear classifier.
    """
    model.eval()
    
    # Get all node embeddings
    z = model().to(device)
    y = data.y.to(device)
    
    # Split data
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)
    
    # Train simple linear classifier
    classifier = torch.nn.Linear(z.size(1), data.y.max().item() + 1).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    
    classifier.train()
    for _ in range(200):
        optimizer.zero_grad()
        out = classifier(z[train_mask])
        loss = F.cross_entropy(out, y[train_mask])
        loss.backward()
        optimizer.step()
    
    # Evaluate
    classifier.eval()
    pred = classifier(z[test_mask]).argmax(dim=1)
    acc = (pred == y[test_mask]).float().mean().item()
    
    return acc


def visualize_embeddings(model, data, save_path: str = None):
    """Visualize Node2Vec embeddings using t-SNE."""
    
    model.eval()
    with torch.no_grad():
        z = model().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    z_2d = tsne.fit_transform(z)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        z_2d[:, 0], z_2d[:, 1],
        c=data.y.cpu().numpy(),
        cmap='tab10',
        s=10,
        alpha=0.7
    )
    plt.colorbar(scatter, label='Class')
    plt.title('Node2Vec Embeddings (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    # Load dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    print(f"Dataset: Cora")
    print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
    print(f"Classes: {dataset.num_classes}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Train Node2Vec
    model = train_node2vec(data, embedding_dim=128, epochs=100)
    
    # Evaluate on node classification
    acc = evaluate_embeddings(model, data, device)
    print(f"\nNode Classification Accuracy: {acc:.4f}")
    
    # Compare with random baseline
    print("(Random baseline would be ~14% for 7 classes)")
    
    # Visualize (optional)
    try:
        visualize_embeddings(model, data, save_path='node2vec_embeddings.png')
    except Exception as e:
        print(f"Visualization skipped: {e}")


if __name__ == "__main__":
    main()
