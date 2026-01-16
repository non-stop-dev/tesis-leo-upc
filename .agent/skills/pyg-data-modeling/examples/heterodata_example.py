"""
HeteroData Example for Multi-Type Graphs

This example shows how to create a heterogeneous graph with multiple
node types (MSMEs, Districts, Sectors) and edge types (relationships).

Use this when your graph has semantically different entity types.
"""

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


def create_msme_hetero_graph() -> HeteroData:
    """
    Create a heterogeneous graph for MSME survival analysis.
    
    Node Types:
        - 'empresa': MSME nodes with business features
        - 'distrito': District nodes with regional features
        - 'sector': Economic sector nodes (CIIU)
    
    Edge Types:
        - ('empresa', 'ubicado_en', 'distrito'): MSME located in district
        - ('empresa', 'pertenece_a', 'sector'): MSME belongs to sector
        - ('empresa', 'compite_con', 'empresa'): MSMEs competing (same sector+district)
    """
    data = HeteroData()
    
    # =========== Node Features ===========
    
    # MSME nodes: features like sales, productivity, formalization
    num_empresas = 1000  # Example: 1000 MSMEs
    data['empresa'].x = torch.randn(num_empresas, 8)  # 8 features
    data['empresa'].y = torch.randint(0, 2, (num_empresas,))  # Survival label
    
    # District nodes: regional characteristics
    num_distritos = 50
    data['distrito'].x = torch.randn(num_distritos, 4)  # 4 regional features
    
    # Sector nodes: sector-level aggregates
    num_sectores = 20
    data['sector'].x = torch.randn(num_sectores, 3)  # 3 sector features
    
    # =========== Edge Indices ===========
    
    # Each MSME is located in one district
    # edge_index: [2, num_edges] where row 0 = source, row 1 = target
    empresa_distrito_edges = torch.stack([
        torch.arange(num_empresas),  # source: empresa IDs
        torch.randint(0, num_distritos, (num_empresas,))  # target: distrito IDs
    ])
    data['empresa', 'ubicado_en', 'distrito'].edge_index = empresa_distrito_edges
    
    # Each MSME belongs to one sector
    empresa_sector_edges = torch.stack([
        torch.arange(num_empresas),
        torch.randint(0, num_sectores, (num_empresas,))
    ])
    data['empresa', 'pertenece_a', 'sector'].edge_index = empresa_sector_edges
    
    # Some MSMEs compete with each other (same local market)
    # Create random competition edges
    num_competition_edges = 5000
    competition_edges = torch.stack([
        torch.randint(0, num_empresas, (num_competition_edges,)),
        torch.randint(0, num_empresas, (num_competition_edges,))
    ])
    data['empresa', 'compite_con', 'empresa'].edge_index = competition_edges
    
    # =========== Train/Val/Test Masks ===========
    
    n = num_empresas
    perm = torch.randperm(n)
    
    data['empresa'].train_mask = torch.zeros(n, dtype=torch.bool)
    data['empresa'].val_mask = torch.zeros(n, dtype=torch.bool)
    data['empresa'].test_mask = torch.zeros(n, dtype=torch.bool)
    
    data['empresa'].train_mask[perm[:int(0.8 * n)]] = True
    data['empresa'].val_mask[perm[int(0.8 * n):int(0.9 * n)]] = True
    data['empresa'].test_mask[perm[int(0.9 * n):]] = True
    
    return data


def add_reverse_edges(data: HeteroData) -> HeteroData:
    """
    Add reverse edges for message passing in both directions.
    
    PyG's ToUndirected transform can do this automatically.
    """
    transform = T.ToUndirected()
    return transform(data)


def print_hetero_stats(data: HeteroData):
    """Print statistics about the heterogeneous graph."""
    print("=" * 50)
    print("Heterogeneous Graph Statistics")
    print("=" * 50)
    
    print("\nNode Types:")
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        num_features = data[node_type].x.shape[1] if hasattr(data[node_type], 'x') else 0
        print(f"  {node_type}: {num_nodes:,} nodes, {num_features} features")
    
    print("\nEdge Types:")
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.shape[1]
        print(f"  {edge_type}: {num_edges:,} edges")
    
    print("\nMetadata:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")


if __name__ == "__main__":
    # Create the heterogeneous graph
    data = create_msme_hetero_graph()
    
    # Add reverse edges for bidirectional message passing
    data = add_reverse_edges(data)
    
    # Print statistics
    print_hetero_stats(data)
    
    # Save for later use
    # torch.save(data, 'msme_hetero_graph.pt')
