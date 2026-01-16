#!/usr/bin/env python3
/**
 * @fileoverview Utility to repair GeoJSON geometries for MapLibre interactivity.
 * Stitches unclosed ways into rings and ensures numeric top-level IDs.
 */
import json
import os
import sys

def stitch_ways(ways):
    """Join coordinates into closed rings (First == Last)."""
    if not ways: return []
    rings = []
    pending_ways = [list(w) for w in ways if len(w) > 0]
    while pending_ways:
        current_ring = pending_ways.pop(0)
        changed = True
        while changed:
            changed = False
            for i, way in enumerate(pending_ways):
                if current_ring[-1] == way[0]:
                    current_ring.extend(way[1:]); pending_ways.pop(i); changed = True; break
                elif current_ring[-1] == way[-1]:
                    current_ring.extend(way[:-1][::-1]); pending_ways.pop(i); changed = True; break
                elif current_ring[0] == way[-1]:
                    current_ring = way[:-1] + current_ring; pending_ways.pop(i); changed = True; break
                elif current_ring[0] == way[0]:
                    current_ring = way[1:][::-1] + current_ring; pending_ways.pop(i); changed = True; break
        
        # Ensure mechanical closure
        if current_ring[0] != current_ring[-1]:
            current_ring.append(current_ring[0])
            
        if len(current_ring) >= 4:
            rings.append(current_ring)
    return rings

def repair_feature(feature):
    """Ensures numeric ID and closed geometry."""
    props = feature.get("properties", {})
    osm_id = props.get("osm_id") or feature.get("id")
    if osm_id:
        try:
            feature["id"] = int(osm_id)
        except:
            feature["id"] = hash(str(osm_id)) % (10**9)
    
    geom = feature.get("geometry", {})
    if not geom: return feature
    
    g_type = geom["type"]
    coords = geom["coordinates"]
    ways = []
    
    if g_type == "Polygon": ways = coords
    elif g_type == "MultiPolygon":
        for poly in coords: ways.extend(poly)
    elif g_type in ["LineString", "MultiLineString"]:
        ways = coords if g_type == "MultiLineString" else [coords]
    else: return feature

    stitched = stitch_ways(ways)
    if not stitched: return feature
    
    if len(stitched) == 1:
        feature["geometry"] = {"type": "Polygon", "coordinates": stitched}
    else:
        feature["geometry"] = {"type": "MultiPolygon", "coordinates": [[r] for r in stitched]}
    return feature

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["features"] = [repair_feature(f) for f in data.get("features", [])]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Repaired: {path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repair_geojson.py <file.geojson>")
    else:
        process_file(sys.argv[1])
