import torch
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os
import sys

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "database", "msme_gnn_preprocessed.dta")
OUTPUT_PATH = os.path.join(BASE_DIR, "database", "msme_graph.pt")

def load_and_process_data():
    print(f"🔄 Loading data from {DATA_PATH}...")
    try:
        df = pd.read_stata(DATA_PATH)
        print(f"✅ Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: File not found at {DATA_PATH}")
        sys.exit(1)

    # --- 1. Node Feature Engineering ---
    print("🛠️  Processing node features...")
    
    # Selecting Features
    # Numeric: Normalized
    numeric_cols = ['ventas_uit_2021', 'productividad_x_trabajador', 'tributos', 'digital_score']
    # Fill NA with median for numerics to avoid NaN in graph
    for col in numeric_cols:
        if col in df.columns:
            # Force numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
             print(f"⚠️ Warning: Missing column {col}, filling with 0")
             df[col] = 0

    scaler = StandardScaler()
    x_numeric = scaler.fit_transform(df[numeric_cols])
    
    # Categorical: One-Hot Encoded
    cat_cols = ['regimen', 'sector', 'tipo_local', 'sexo_gerente', 'tamano_empresa']
    # Fill NA
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = 0
            
    x_categorical = []
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        x_categorical.append(dummies.values)
    
    x_categorical = np.hstack(x_categorical)
    
    # Combine Features for MSME Nodes
    x_msme = np.hstack([x_numeric, x_categorical])
    x_msme = torch.from_numpy(x_msme).float()
    print(f"✅ MSME Features created. Shape: {x_msme.shape}")

    # Target Label
    print("🎯 Encoding target variable...")
    le = LabelEncoder()
    # Ensure it's treated as string first to avoid category issues
    y_encoded = le.fit_transform(df['op2021_ajustado'].astype(str))
    y = torch.from_numpy(y_encoded).long()
    print(f"✅ Target mappings: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --- 2. Entity Node Indexing ---
    print("📍 Indexing entity nodes (District, Sector)...")
    
    # Districts (Ubigeo)
    districts = df['ubigeo'].unique()
    district_map = {code: i for i, code in enumerate(districts)}
    num_districts = len(districts)
    
    # Sectors (CIIU 4-dig)
    sectors = df['ciiu_4dig'].unique()
    sector_map = {code: i for i, code in enumerate(sectors)}
    num_sectors = len(sectors)
    
    # Mapping MSMEs to Indices
    msme_to_district = df['ubigeo'].map(district_map).values
    msme_to_sector = df['ciiu_4dig'].map(sector_map).values
    
    # --- 3. Construct HeteroData ---
    print("🏗️  Constructing HeteroData object...")
    data = HeteroData()
    
    # Add Nodes
    data['msme'].x = x_msme
    data['msme'].y = y
    data['msme'].num_nodes = len(df)
    
    # Entity nodes don't have features initially, can use embeddings or 1-hot
    # Simple strategy: Identity identity matrix or empty features handled by embedding layer later
    # For now, we set num_nodes so PyG knows they exist
    data['district'].num_nodes = num_districts
    data['sector'].num_nodes = num_sectors
    
    # Add Edges (MSME -> Entity)
    # Edge: located_in (MSME -> District)
    src_msme = torch.arange(len(df))
    dst_district = torch.from_numpy(msme_to_district)
    edge_index_loc = torch.stack([src_msme, dst_district], dim=0)
    data['msme', 'located_in', 'district'].edge_index = edge_index_loc
    
    # Edge: competes_in (MSME -> Sector)
    dst_sector = torch.from_numpy(msme_to_sector)
    edge_index_comp = torch.stack([src_msme, dst_sector], dim=0)
    data['msme', 'competes_in', 'sector'].edge_index = edge_index_comp
    
    # --- 4. Transform to Undirected / Add Reverse Edges ---
    # Essential for Message Passing flow back to MSMEs
    # (District -> MSME) and (Sector -> MSME)
    print("🔄 Adding reverse edges (ToUndirected)...")
    data = T.ToUndirected()(data)
    
    # --- 5. Train/Val/Test Split (Inductive) ---
    print("✂️  Creating Train/Val/Test masks...")
    # 70% Train, 15% Val, 15% Test
    num_nodes = len(df)
    indices = torch.randperm(num_nodes)
    
    train_size = int(0.7 * num_nodes)
    val_size = int(0.15 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    data['msme'].train_mask = train_mask
    data['msme'].val_mask = val_mask
    data['msme'].test_mask = test_mask
    
    # --- 6. Save ---
    print(f"💾 Saving graph to {OUTPUT_PATH}...")
    torch.save(data, OUTPUT_PATH)
    print("🎉 Done! Graph construction complete.")
    print(data)

if __name__ == "__main__":
    load_and_process_data()
