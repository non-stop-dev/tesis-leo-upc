#!/usr/bin/env python3
"""
Merge MSME counts with district GeoJSON using name matching.

Since OSM districts lack INEI UBIGEO codes, we match by:
1. District name (normalized)
2. Fallback: Department + District name combination

Input:
    - database/msme_gnn_preprocessed.dta
    - visualization-astro/public/map-geojson/peru_districts.geojson

Output:
    - visualization-astro/public/map-geojson/peru_districts_msme.geojson
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
GEOJSON_INPUT = os.path.join(BASE_DIR, "..", "..", "visualization-astro", "public", "map-geojson", "peru_districts.geojson")
GEOJSON_OUTPUT = os.path.join(BASE_DIR, "..", "..", "visualization-astro", "public", "map-geojson", "peru_districts_msme.geojson")


def normalize_name(name: str) -> str:
    """Normalize district name for matching."""
    if not name:
        return ""
    # Remove accents
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    # Lowercase and strip
    name = name.lower().strip()
    # Remove common prefixes
    name = re.sub(r'^(distrito de |district of )', '', name)
    # Remove special chars except spaces
    name = re.sub(r'[^a-z0-9\s]', '', name)
    # Collapse whitespace
    name = re.sub(r'\s+', ' ', name)
    return name


def load_msme_counts_by_district() -> tuple[dict[str, int], dict[str, str]]:
    """
    Load MSME data and aggregate counts per district.
    Returns: (counts by normalized name, district lookups)
    """
    print(f"[Data] Loading {DTA_PATH}...")
    
    if not os.path.exists(DTA_PATH):
        print(f"[Error] File not found: {DTA_PATH}")
        sys.exit(1)
    
    df = pd.read_stata(DTA_PATH)
    print(f"[Data] Loaded {len(df):,} MSMEs")
    
    # Create aggregation by district name
    # Group by UBIGEO to get district-level counts
    district_counts = df.groupby(['ubigeo', 'DISTRITO', 'DEPARTAMENTO']).size().reset_index(name='count')
    
    # Create lookup by normalized name
    counts_by_name: dict[str, int] = {}
    ubigeo_lookup: dict[str, str] = {}
    
    for _, row in district_counts.iterrows():
        norm_name = normalize_name(row['DISTRITO'])
        dept_name = normalize_name(row['DEPARTAMENTO'])
        
        # Create composite key (dept + district) for unique matching
        composite_key = f"{dept_name}|{norm_name}"
        
        # Store count
        if composite_key in counts_by_name:
            counts_by_name[composite_key] += row['count']
        else:
            counts_by_name[composite_key] = row['count']
        
        # Also store by just district name (may have conflicts)
        if norm_name not in counts_by_name:
            counts_by_name[norm_name] = row['count']
        else:
            counts_by_name[norm_name] += row['count']
        
        ubigeo_lookup[composite_key] = row['ubigeo']
    
    print(f"[Data] Found {len(district_counts)} unique districts in DTA")
    
    return counts_by_name, ubigeo_lookup


def get_department_from_osm_hierarchy(feature: dict) -> str:
    """Try to extract department from feature properties or parent."""
    # OSM features may have is_in tag or we infer from batch processing
    props = feature.get("properties", {})
    return props.get("department", "")


def merge_with_geojson(msme_counts: dict[str, int]) -> dict:
    """Merge MSME counts with district GeoJSON using name matching."""
    print(f"[GeoJSON] Loading {GEOJSON_INPUT}...")
    
    if not os.path.exists(GEOJSON_INPUT):
        print(f"[Error] GeoJSON not found: {GEOJSON_INPUT}")
        sys.exit(1)
    
    with open(GEOJSON_INPUT, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    
    features = geojson.get("features", [])
    print(f"[GeoJSON] Found {len(features)} district features")
    
    # Match by name
    matched = 0
    partial_matched = 0
    unmatched: list[str] = []
    
    for feature in features:
        props = feature.get("properties", {})
        district_name = props.get("name_es", props.get("name", ""))
        norm_name = normalize_name(district_name)
        
        # Try exact name match
        if norm_name in msme_counts:
            props["msme_count"] = msme_counts[norm_name]
            props["nombre_distrito"] = district_name
            props["match_type"] = "name"
            matched += 1
        else:
            # Try fuzzy matching or partial match
            found = False
            for key, count in msme_counts.items():
                if '|' not in key and norm_name in key or key in norm_name:
                    props["msme_count"] = count
                    props["nombre_distrito"] = district_name
                    props["match_type"] = "partial"
                    partial_matched += 1
                    found = True
                    break
            
            if not found:
                props["msme_count"] = 0
                props["nombre_distrito"] = district_name
                props["match_type"] = "none"
                unmatched.append(district_name)
    
    print(f"[Merge] Matched: {matched} exact, {partial_matched} partial")
    print(f"[Merge] Unmatched: {len(unmatched)} districts")
    
    if unmatched and len(unmatched) <= 10:
        print(f"  Unmatched examples: {unmatched[:10]}")
    
    return geojson


def save_geojson(geojson: dict):
    """Save merged GeoJSON."""
    os.makedirs(os.path.dirname(GEOJSON_OUTPUT), exist_ok=True)
    
    with open(GEOJSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    
    print(f"[Saved] {GEOJSON_OUTPUT}")


def main():
    # Step 1: Load MSME counts
    msme_counts, _ = load_msme_counts_by_district()
    
    # Step 2: Merge with GeoJSON
    merged_geojson = merge_with_geojson(msme_counts)
    
    # Step 3: Save
    save_geojson(merged_geojson)
    
    print("[Done] GeoJSON ready for visualization")


if __name__ == "__main__":
    main()
