#!/usr/bin/env python3
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_GEOJSON = os.path.join(BASE_DIR, "../../visualization-astro/public/map-geojson/peru_districts.geojson")
LORETO_RAW = os.path.join(BASE_DIR, "../../visualization-astro/public/map-geojson/temp/dept_1994077.json")
FINAL_OUTPUT = os.path.join(BASE_DIR, "../../visualization-astro/public/map-geojson/peru_districts.geojson")

def stitch_ways(ways):
    if not ways: return []
    rings = []
    pending_ways = [list(w) for w in ways if len(w) > 0]
    while pending_ways:
        current_ring = pending_ways.pop(0)
        changed = True
        while changed:
            changed = False
            for i, way in enumerate(pending_ways):
                if current_ring[-1] == way[0]: current_ring.extend(way[1:]); pending_ways.pop(i); changed = True; break
                elif current_ring[-1] == way[-1]: current_ring.extend(way[:-1][::-1]); pending_ways.pop(i); changed = True; break
                elif current_ring[0] == way[-1]: current_ring = way[:-1] + current_ring; pending_ways.pop(i); changed = True; break
                elif current_ring[0] == way[0]: current_ring = way[1:][::-1] + current_ring; pending_ways.pop(i); changed = True; break
            if current_ring[0] == current_ring[-1] and len(current_ring) > 2: break
        if current_ring[0] != current_ring[-1]: current_ring.append(current_ring[0])
        rings.append(current_ring)
    return rings

def process_overpass_elements(elements):
    features = []
    for el in elements:
        if el["type"] != "relation": continue
        tags = el.get("tags", {})
        members = el.get("members", [])
        outer_ways = [[(n["lon"], n["lat"]) for n in m["geometry"]] for m in members if m["type"] == "way" and m.get("role") != "inner" and "geometry" in m]
        inner_ways = [[(n["lon"], n["lat"]) for n in m["geometry"]] for m in members if m["type"] == "way" and m.get("role") == "inner" and "geometry" in m]
        
        outer_rings = stitch_ways(outer_ways)
        if not outer_rings: continue
        rings = [outer_rings[0]]
        if inner_ways:
            i_rings = stitch_ways(inner_ways)
            if i_rings: rings.extend(i_rings)
        
        features.append({
            "type": "Feature",
            "id": el["id"],
            "properties": {
                "osm_id": el["id"],
                "name": tags.get("name", ""),
                "name_es": tags.get("name:es", tags.get("name", "")),
                "ubigeo": tags.get("ref:INEI", ""),
                "department": "Loreto"
            },
            "geometry": {"type": "Polygon", "coordinates": rings}
        })
    return features

def main():
    # 1. Load old features (if exists)
    all_features = []
    if os.path.exists(OLD_GEOJSON):
        print(f"Loading existing {OLD_GEOJSON}...")
        with open(OLD_GEOJSON, "r") as f:
            old_data = json.load(f)
            all_features = old_data.get("features", [])
            print(f"Loaded {len(all_features)} existing districts.")
    
    # 2. Process Loreto
    if os.path.exists(LORETO_RAW):
        print("Processing Loreto raw data...")
        with open(LORETO_RAW, "r") as f:
            loreto_data = json.load(f)
            loreto_features = process_overpass_elements(loreto_data.get("elements", []))
            print(f"Assembled {len(loreto_features)} Loreto districts.")
            
            # Remove existing Loreto duplicates if any
            all_features = [f for f in all_features if f.get("properties", {}).get("department") != "Loreto"]
            all_features.extend(loreto_features)

    # 3. Save final
    print(f"Saving {len(all_features)} districts to {FINAL_OUTPUT}...")
    with open(FINAL_OUTPUT, "w") as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f)
    print("Done!")

if __name__ == "__main__":
    main()
