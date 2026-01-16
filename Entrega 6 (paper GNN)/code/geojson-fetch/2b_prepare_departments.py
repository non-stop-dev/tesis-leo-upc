#!/usr/bin/env python3
"""
Merge MSME counts with department GeoJSON using name matching.

Input:
    - database/msme_gnn_preprocessed.dta
    - visualization-astro/public/map-geojson/peru_departments.geojson

Output:
    - visualization-astro/public/map-geojson/peru_departments_msme.geojson
"""

import json
import os
import re
import sys
import unicodedata
import pandas as pd

# Paths - relative to this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTA_PATH = os.path.join(BASE_DIR, "..", "..", "database", "msme_gnn_preprocessed.dta")
GEOJSON_INPUT = os.path.join(BASE_DIR, "..", "..", "visualization-astro", "public", "map-geojson", "peru_departments.geojson")
GEOJSON_OUTPUT = os.path.join(BASE_DIR, "..", "..", "visualization-astro", "public", "map-geojson", "peru_departments_msme.geojson")

def normalize_name(name: str) -> str:
    """Normalize name for matching."""
    if not name: return ""
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def load_msme_counts_by_department() -> dict[str, int]:
    """Load MSME data and aggregate counts per department."""
    print(f"[Data] Loading {DTA_PATH}...")
    
    if not os.path.exists(DTA_PATH):
        print(f"[Error] File not found: {DTA_PATH}")
        sys.exit(1)
    
    df = pd.read_stata(DTA_PATH)
    print(f"[Data] Loaded {len(df):,} MSMEs")
    
    # Aggregate by department name
    counts = df.groupby("DEPARTAMENTO").size().to_dict()
    
    norm_counts = {normalize_name(k): v for k, v in counts.items()}
    print(f"[Data] Found {len(norm_counts):,} unique departments")
    
    return norm_counts

def merge_with_geojson(msme_counts: dict[str, int]) -> dict:
    """Merge MSME counts with department GeoJSON using name matching."""
    print(f"[GeoJSON] Loading {GEOJSON_INPUT}...")
    
    if not os.path.exists(GEOJSON_INPUT):
        print(f"[Error] GeoJSON not found: {GEOJSON_INPUT}")
        sys.exit(1)
    
    with open(GEOJSON_INPUT, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    
    # Manual mappings
    MANUAL_MAPPING = {
        "provinciahumanaconstitucionaldelcallao": "callao",
        "regioncallao": "callao",
        "limametropolitana": "lima"
    }

    features = geojson.get("features", [])
    print(f"[GeoJSON] Found {len(features)} department features")
    
    matched = 0
    for feature in features:
        props = feature.get("properties", {})
        name = props.get("name_es", props.get("name", ""))
        
        norm_name = normalize_name(name)
        norm_name = MANUAL_MAPPING.get(norm_name, norm_name)
        
        if norm_name in msme_counts:
            props["msme_count"] = msme_counts[norm_name]
            matched += 1
        else:
            props["msme_count"] = 0
    
    print(f"[Merge] Matched: {matched}/{len(features)} departments")
    return geojson

def save_geojson(geojson: dict):
    """Save merged GeoJSON."""
    os.makedirs(os.path.dirname(GEOJSON_OUTPUT), exist_ok=True)
    with open(GEOJSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"[Saved] {GEOJSON_OUTPUT}")

def main():
    msme_counts = load_msme_counts_by_department()
    merged_geojson = merge_with_geojson(msme_counts)
    save_geojson(merged_geojson)
    print("[Done] Department-level GeoJSON ready for visualization")

if __name__ == "__main__":
    main()
