#!/usr/bin/env python3
"""
Merge MSME counts with province GeoJSON using name matching.

Input:
    - database/msme_gnn_preprocessed.dta
    - visualization-astro/public/map-geojson/peru_provinces.geojson

Output:
    - visualization-astro/public/map-geojson/peru_provinces_msme.geojson
"""

import json
import os
import re
import sys
import unicodedata
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTA_PATH = os.path.join(BASE_DIR, "..", "..", "database", "msme_gnn_preprocessed.dta")
GEOJSON_INPUT = os.path.join(BASE_DIR, "..", "..", "visualization-astro/public/map-geojson/peru_provinces.geojson")
GEOJSON_OUTPUT = os.path.join(BASE_DIR, "..", "..", "visualization-astro/public/map-geojson/peru_provinces_msme.geojson")

def normalize_name(name: str) -> str:
    """Normalize name for matching."""
    if not name: return ""
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def load_msme_counts_by_province() -> dict[str, int]:
    print(f"[Data] Loading {DTA_PATH}...")
    if not os.path.exists(DTA_PATH):
        print(f"[Error] File not found: {DTA_PATH}")
        sys.exit(1)
    
    df = pd.read_stata(DTA_PATH)
    print(f"[Data] Loaded {len(df):,} MSMEs")
    
    # Aggregation by department + province to avoid collisions
    province_counts = df.groupby(['DEPARTAMENTO', 'PROVINCIA']).size().reset_index(name='count')
    
    counts_by_key: dict[str, int] = {}
    for _, row in province_counts.iterrows():
        dept_norm = normalize_name(row['DEPARTAMENTO'])
        prov_norm = normalize_name(row['PROVINCIA'])
        key = f"{dept_norm}|{prov_norm}"
        counts_by_key[key] = row['count']
        
        # Also store by just province name if not exists (for fallback)
        if prov_norm not in counts_by_key:
            counts_by_key[prov_norm] = row['count']
        else:
            # If collision, we prefer the composite key
            pass

    print(f"[Data] Found {len(province_counts)} unique provinces in DTA")
    return counts_by_key

def merge_with_geojson(msme_counts: dict[str, int]) -> dict:
    print(f"[GeoJSON] Loading {GEOJSON_INPUT}...")
    if not os.path.exists(GEOJSON_INPUT):
        print(f"[Error] GeoJSON not found: {GEOJSON_INPUT}")
        sys.exit(1)
    
    with open(GEOJSON_INPUT, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    
    # Manual mappings for known mismatches
    MANUAL_MAPPING = {
        "nasca": "nazca",
        "huancasancos": "huancasancos"
    }

    features = geojson.get("features", [])
    print(f"[GeoJSON] Found {len(features)} province features")
    
    matched = 0
    unmatched = []
    
    for feature in features:
        props = feature.get("properties", {})
        prov_name = props.get("name_es", props.get("name", ""))
        dept_name = props.get("department", "")
        
        norm_prov = normalize_name(prov_name)
        norm_prov = MANUAL_MAPPING.get(norm_prov, norm_prov)
        norm_dept = normalize_name(dept_name)
        
        composite_key = f"{norm_dept}|{norm_prov}"
        
        if composite_key in msme_counts:
            props["msme_count"] = msme_counts[composite_key]
            props["match_type"] = "exact_composite"
            matched += 1
        elif norm_prov in msme_counts:
            props["msme_count"] = msme_counts[norm_prov]
            props["match_type"] = "exact_name"
            matched += 1
        else:
            props["msme_count"] = 0
            props["match_type"] = "none"
            unmatched.append(f"{dept_name} > {prov_name}")
            
    print(f"[Merge] Matched: {matched}/{len(features)} provinces")
    if unmatched:
        print(f"  Unmatched: {unmatched[:5]}")
        
    return geojson

def main():
    msme_counts = load_msme_counts_by_province()
    merged_geojson = merge_with_geojson(msme_counts)
    
    os.makedirs(os.path.dirname(GEOJSON_OUTPUT), exist_ok=True)
    with open(GEOJSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(merged_geojson, f, ensure_ascii=False)
    print(f"[Saved] {GEOJSON_OUTPUT}")

if __name__ == "__main__":
    main()
