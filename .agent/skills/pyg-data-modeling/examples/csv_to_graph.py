"""
CSV to PyG Graph Conversion Example

This example demonstrates how to convert tabular MSME data from CSV 
into a PyTorch Geometric Data object suitable for GNN training.

Adapted for: Peruvian MSME Survival Prediction
"""

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_csv(filepath: str) -> pd.DataFrame:
    """Load CSV and perform basic preprocessing."""
    df = pd.read_csv(filepath)
    
    # Example columns for MSME data:
    # - id_empresa: unique identifier
    # - distrito: district code
    # - ciiu_4dig: 4-digit economic activity code
    # - ventas_k: sales in thousands
    # - productividad_k: productivity measure
    # - regimen: tax regime (categorical)
    # - ruc: has RUC (0/1)
    # - op2021_original: survival label (0/1)
    
    return df


def create_node_features(df: pd.DataFrame, 
                         numeric_cols: list[str],
                         categorical_cols: list[str]) -> torch.Tensor:
    """
    Create node feature matrix from DataFrame columns.
    
    Args:
        df: DataFrame with MSME data
        numeric_cols: Columns to standardize (e.g., ['ventas_k', 'productividad_k'])
        categorical_cols: Columns to one-hot encode (e.g., ['regimen'])
    
    Returns:
        Tensor of shape [num_nodes, num_features]
    """
    features_list = []
    
    # Standardize numeric features
    if numeric_cols:
        scaler = StandardScaler()
        numeric_features = scaler.fit_transform(df[numeric_cols].fillna(0))
        features_list.append(numeric_features)
    
    # One-hot encode categorical features
    for col in categorical_cols:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].fillna('unknown'))
        one_hot = np.eye(len(le.classes_))[encoded]
        features_list.append(one_hot)
    
    # Concatenate all features
    all_features = np.concatenate(features_list, axis=1)
    
    return torch.tensor(all_features, dtype=torch.float)


def create_edges_by_district(df: pd.DataFrame, 
                              district_col: str = 'distrito') -> torch.Tensor:
    """
    Create edges between MSMEs in the same district (geographic proximity).
    
    This creates a graph where MSMEs are connected if they share the same district.
    For large datasets, consider sampling edges to avoid memory issues.
    
    Returns:
        edge_index tensor of shape [2, num_edges]
    """
    edge_list = []
    
    # Group by district
    for district, group in df.groupby(district_col):
        indices = group.index.tolist()
        n = len(indices)
        
        # For small groups, connect all pairs
        if n <= 100:
            for i in range(n):
                for j in range(i + 1, n):
                    edge_list.append([indices[i], indices[j]])
                    edge_list.append([indices[j], indices[i]])  # Undirected
        else:
            # For large groups, sample random connections
            np.random.seed(42)
            for i in range(n):
                neighbors = np.random.choice(indices, size=min(10, n-1), replace=False)
                for j in neighbors:
                    if i != j:
                        edge_list.append([indices[i], j])
    
    if not edge_list:
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


def create_edges_by_sector(df: pd.DataFrame,
                            sector_col: str = 'ciiu_4dig') -> torch.Tensor:
    """
    Create edges between MSMEs in the same economic sector (competition).
    
    Returns:
        edge_index tensor of shape [2, num_edges]
    """
    edge_list = []
    
    for sector, group in df.groupby(sector_col):
        indices = group.index.tolist()
        n = len(indices)
        
        # Sample connections for sectors with many MSMEs
        if n <= 50:
            for i in range(n):
                for j in range(i + 1, n):
                    edge_list.append([indices[i], indices[j]])
                    edge_list.append([indices[j], indices[i]])
        else:
            np.random.seed(42)
            for i in range(n):
                neighbors = np.random.choice(indices, size=min(5, n-1), replace=False)
                for j in neighbors:
                    if i != j:
                        edge_list.append([indices[i], j])
    
    if not edge_list:
        return torch.zeros((2, 0), dtype=torch.long)
    
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def build_msme_graph(csv_path: str) -> Data:
    """
    Build a complete PyG Data object from MSME CSV data.
    
    Returns:
        PyG Data object ready for GNN training
    """
    # Load data
    df = load_and_preprocess_csv(csv_path)
    
    # Reset index to ensure contiguous node IDs
    df = df.reset_index(drop=True)
    
    # Create node features
    numeric_cols = ['ventas_k', 'productividad_k', 'tributos_k']
    categorical_cols = ['regimen']
    x = create_node_features(df, numeric_cols, categorical_cols)
    
    # Create edges (combine district and sector edges)
    edge_district = create_edges_by_district(df)
    edge_sector = create_edges_by_sector(df)
    
    # Combine edge types
    edge_index = torch.cat([edge_district, edge_sector], dim=1)
    
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)
    
    # Create labels (survival: 0 or 1)
    y = torch.tensor(df['op2021_original'].values, dtype=torch.long)
    
    # Create train/val/test masks (example: 80/10/10 split)
    n = len(df)
    perm = torch.randperm(n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    
    train_mask[perm[:int(0.8 * n)]] = True
    val_mask[perm[int(0.8 * n):int(0.9 * n)]] = True
    test_mask[perm[int(0.9 * n):]] = True
    
    # Build Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    
    # Validate
    data.validate(raise_on_error=True)
    
    print(f"Graph Statistics:")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features: {data.num_node_features}")
    print(f"  Training nodes: {train_mask.sum().item():,}")
    print(f"  Validation nodes: {val_mask.sum().item():,}")
    print(f"  Test nodes: {test_mask.sum().item():,}")
    
    return data


if __name__ == "__main__":
    # Example usage
    # data = build_msme_graph("path/to/msme_data.csv")
    # torch.save(data, "msme_graph.pt")
    
    print("Run with: data = build_msme_graph('your_data.csv')")
