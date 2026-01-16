#!/usr/bin/env python3
import json
import os
import sys

def stitch_ways(ways):
    """Join coordinates into closed rings."""
    if not ways: return []
    rings = []
    # Convert to list of lists to handle mutability
    pending_ways = [list(w) for w in ways if len(w) > 0]
    while pending_ways:
        current_ring = pending_ways.pop(0)
        changed = True
        while changed:
            changed = False
            for i, way in enumerate(pending_ways):
                # Try all 4 connection possibilities
                if current_ring[-1] == way[0]:
                    current_ring.extend(way[1:])
                    pending_ways.pop(i)
                    changed = True
                    break
                elif current_ring[-1] == way[-1]:
                    current_ring.extend(way[:-1][::-1])
                    pending_ways.pop(i)
                    changed = True
                    break
                elif current_ring[0] == way[-1]:
                    current_ring = way[:-1] + current_ring
                    pending_ways.pop(i)
                    changed = True
                    break
                elif current_ring[0] == way[0]:
                    current_ring = way[1:][::-1] + current_ring
                    pending_ways.pop(i)
                    changed = True
                    break
            # If ring is closed, we can stop for this ring
            if current_ring[0] == current_ring[-1] and len(current_ring) > 2:
                break
        
        # Force closure if not closed
        if current_ring[0] != current_ring[-1]:
            current_ring.append(current_ring[0])
            
        if len(current_ring) > 3: # Valid polygon ring must have at least 4 points (first==last)
            rings.append(current_ring)
            
    return rings

def repair_feature(feature):
    """Repair geometry and ensure numeric top-level ID."""
    # Ensure numeric ID for feature-state
    props = feature.get("properties", {})
    osm_id = props.get("osm_id") or feature.get("id")
    if osm_id:
        try:
            feature["id"] = int(osm_id)
        except (ValueError, TypeError):
            # Fallback to hash if not numeric
            feature["id"] = hash(str(osm_id)) % (10**9)
    
    geom = feature.get("geometry", {})
    if not geom or "coordinates" not in geom:
        return feature
        
    g_type = geom["type"]
    coords = geom["coordinates"]
    
    # Extract all line segments
    ways = []
    if g_type == "Polygon":
        ways = coords
    elif g_type == "MultiPolygon":
        for poly in coords:
            ways.extend(poly)
    elif g_type in ["LineString", "MultiLineString"]:
        ways = coords if g_type == "MultiLineString" else [coords]
    else:
        return feature

    stitched = stitch_ways(ways)
    if not stitched:
        return feature
        
    # Rebuild as Polygon or MultiPolygon
    if len(stitched) == 1:
        feature["geometry"] = {"type": "Polygon", "coordinates": stitched}
    else:
        # MultiPolygon format is list of Polygons, each Polygon is list of Rings
        feature["geometry"] = {"type": "MultiPolygon", "coordinates": [[r] for r in stitched]}
        
    return feature

def process_file(input_path, output_path):
    print(f"Repairing {input_path} -> {output_path}")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    features = data.get("features", [])
    print(f"Found {len(features)} features.")
    
    repaired = [repair_feature(f) for f in features]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": repaired}, f)
    print("Done.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PUBLIC_DIR = os.path.join(BASE_DIR, "../../visualization-astro/public/map-geojson")
    
    # Files to repair
    files = [
        "peru_departments.geojson",
        "peru_departments_msme.geojson",
        "peru_districts.geojson",
        "peru_districts_msme.geojson",
        "peru_provinces.geojson",
        "peru_provinces_msme.geojson"
    ]
    
    for f in files:
        path = os.path.join(PUBLIC_DIR, f)
        if os.path.exists(path):
            process_file(path, path)
        else:
            print(f"Skipping {f} (not found)")
