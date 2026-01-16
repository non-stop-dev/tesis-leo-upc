#!/usr/bin/env python3
"""
Merge MSME counts with district GeoJSON.

This script:
1. Loads the preprocessed MSME data (DTA)
2. Aggregates MSME count per UBIGEO (district)
3. Merges counts with district GeoJSON boundaries
4. Outputs a visualization-ready GeoJSON

Input:
    - database/msme_gnn_preprocessed.dta
    - visualization/public/map-geojson/peru_districts.geojson

Output:
    - visualization/public/map-geojson/peru_districts_msme.geojson
"""

import json
import os
import sys
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTA_PATH = os.path.join(BASE_DIR, "..", "database", "msme_gnn_preprocessed.dta")
GEOJSON_INPUT = os.path.join(BASE_DIR, "visualization", "public", "map-geojson", "peru_districts.geojson")
GEOJSON_OUTPUT = os.path.join(BASE_DIR, "visualization", "public", "map-geojson", "peru_districts_msme.geojson")


def load_msme_counts() -> dict[str, int]:
    """Load MSME data and aggregate counts per district."""
    print(f"[Data] Loading {DTA_PATH}...")
    
    if not os.path.exists(DTA_PATH):
        print(f"[Error] File not found: {DTA_PATH}")
        sys.exit(1)
    
    df = pd.read_stata(DTA_PATH)
    print(f"[Data] Loaded {len(df):,} MSMEs")
    
    # Aggregate by UBIGEO
    counts = df.groupby("ubigeo").size().to_dict()
    print(f"[Data] Found {len(counts):,} unique districts")
    
    # Print summary stats
    values = list(counts.values())
    print(f"[Stats] Min: {min(values)}, Max: {max(values)}, Mean: {sum(values)/len(values):.1f}")
    
    return counts


def merge_with_geojson(msme_counts: dict[str, int]) -> dict:
    """Merge MSME counts with district GeoJSON."""
    print(f"[GeoJSON] Loading {GEOJSON_INPUT}...")
    
    if not os.path.exists(GEOJSON_INPUT):
        print(f"[Error] GeoJSON not found: {GEOJSON_INPUT}")
        print("[Tip] Run 1_fetch_boundaries.py first to download district boundaries")
        sys.exit(1)
    
    with open(GEOJSON_INPUT, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    
    features = geojson.get("features", [])
    print(f"[GeoJSON] Found {len(features)} district features")
    
    # Match and merge
    matched = 0
    unmatched_ubigeos: list[str] = []
    
    for feature in features:
        props = feature.get("properties", {})
        ubigeo = str(props.get("ubigeo", "")).zfill(6)  # Pad to 6 digits
        
        if ubigeo in msme_counts:
            props["msme_count"] = msme_counts[ubigeo]
            props["nombre_distrito"] = props.get("name_es", props.get("name", ""))
            matched += 1
        else:
            props["msme_count"] = 0
            unmatched_ubigeos.append(ubigeo)
    
    print(f"[Merge] Matched: {matched}/{len(features)} districts")
    
    if unmatched_ubigeos:
        print(f"[Warning] {len(unmatched_ubigeos)} districts without MSME data")
        # Show first 5 unmatched
        print(f"  Examples: {unmatched_ubigeos[:5]}")
    
    return geojson


def save_geojson(geojson: dict):
    """Save merged GeoJSON."""
    os.makedirs(os.path.dirname(GEOJSON_OUTPUT), exist_ok=True)
    
    with open(GEOJSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    
    print(f"[Saved] {GEOJSON_OUTPUT}")


def main():
    # Step 1: Load MSME counts
    msme_counts = load_msme_counts()
    
    # Step 2: Merge with GeoJSON
    merged_geojson = merge_with_geojson(msme_counts)
    
    # Step 3: Save
    save_geojson(merged_geojson)
    
    print("[Done] GeoJSON ready for visualization")


if __name__ == "__main__":
    main()
