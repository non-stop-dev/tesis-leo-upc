import pandas as pd
import sys

DATA_PATH = "/Users/leonardoleon/Library/Mobile Documents/com~apple~CloudDocs/Universidad/UPC/9no ciclo/Tesis 1/Entrega 6 (paper GNN)/database/msme_gnn_preprocessed.dta"

def check_data():
    print(f"Loading data from: {DATA_PATH}")
    try:
        # chunksize=1000 to just read a bit if possible, but read_stata might load all. 
        # reading full file to get stats
        df = pd.read_stata(DATA_PATH)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    required_cols = ['ubigeo', 'ciiu_4dig', 'op2021_ajustado', 'ciiu_2dig']
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
    else:
        print(f"✅ All required columns present: {required_cols}")

    # Inspect Ubigeo
    if 'ubigeo' in df.columns:
        print("\n--- UBIGEO Inspection ---")
        print(df['ubigeo'].head())
        print(f"Missing ubigeo: {df['ubigeo'].isna().sum()}")
        # Check length
        sample_ubigeo = df.dropna(subset=['ubigeo']).iloc[0]['ubigeo']
        print(f"Sample ubigeo: {sample_ubigeo} (Type: {type(sample_ubigeo)})")
    
    # Inspect CIIU
    if 'ciiu_4dig' in df.columns:
        print("\n--- CIIU 4-dig Inspection ---")
        print(df['ciiu_4dig'].head())
        print(f"Unique sectors: {df['ciiu_4dig'].nunique()}")
        print(f"Missing ciiu_4dig: {df['ciiu_4dig'].isna().sum()}")

    # Inspect Target
    if 'op2021_ajustado' in df.columns:
        print("\n--- Target Distribution (op2021_ajustado) ---")
        print(df['op2021_ajustado'].value_counts(normalize=True))
        print(df['op2021_ajustado'].value_counts())

    # Inspect Geo names for verification
    geo_cols = ['PROVINCIA', 'DISTRITO', 'DEPARTAMENTO']
    present_geo = [c for c in geo_cols if c in df.columns]
    if present_geo:
         print("\n--- Geographic Names Sample ---")
         print(df[present_geo + ['ubigeo']].head())

if __name__ == "__main__":
    check_data()
