#!/usr/bin/env python3
import json
import os
import time
import urllib.request
import urllib.parse
import sys

# Multiple mirrors for reliability
MIRRORS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://overpass.osm.ch/api/interpreter"
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "visualization-astro", "public", "map-geojson")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
FINAL_OUTPUT = os.path.join(OUTPUT_DIR, "peru_districts.geojson")

os.makedirs(TEMP_DIR, exist_ok=True)

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
                if current_ring[-1] == way[0]:
                    current_ring.extend(way[1:]); pending_ways.pop(i); changed = True; break
                elif current_ring[-1] == way[-1]:
                    current_ring.extend(way[:-1][::-1]); pending_ways.pop(i); changed = True; break
                elif current_ring[0] == way[-1]:
                    current_ring = way[:-1] + current_ring; pending_ways.pop(i); changed = True; break
                elif current_ring[0] == way[0]:
                    current_ring = way[1:][::-1] + current_ring; pending_ways.pop(i); changed = True; break
            if current_ring[0] == current_ring[-1] and len(current_ring) > 2: break
        if current_ring[0] != current_ring[-1]: current_ring.append(current_ring[0])
        rings.append(current_ring)
    return rings

def query_overpass(query, timeout=900):
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    for mirror in MIRRORS:
        print(f"  [Overpass] Trying {mirror}...")
        for attempt in range(1, 3):
            try:
                req = urllib.request.Request(mirror, data=data)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status == 200:
                        return json.loads(response.read().decode("utf-8"))
            except Exception as e:
                print(f"    [Attempt {attempt}] Error: {e}")
                time.sleep(5)
    return None

def fetch_departments():
    query = """
    [out:json][timeout:120];
    rel(288247);
    rel(r)["admin_level"="4"];
    out body;
    """
    print("[Overpass] Fetching departments...")
    data = query_overpass(query)
    if data and "elements" in data: return data["elements"]
    return []

def process_department(dept):
    tags = dept.get("tags", {})
    dept_name = tags.get("name", "Unknown")
    dept_id = dept.get("id")
    temp_file = os.path.join(TEMP_DIR, f"dept_{dept_id}.json")
    
    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 100:
        print(f"  [{dept_name}] Cached")
        with open(temp_file, "r") as f: return json.load(f)

    timeout = 1800 if "Loreto" in dept_name else 900
    area_id = 3600000000 + dept_id
    query = f'[out:json][timeout:{timeout}]; area({area_id})->.a; rel(area.a)["admin_level"="8"]; out body geom;'
    
    print(f"  [{dept_name}] Fetching districts...")
    data = query_overpass(query, timeout + 60)
    if not data or "elements" not in data: return []

    features = []
    for el in data.get("elements", []):
        if el["type"] != "relation": continue
        tags = el.get("tags", {})
        outer_ways = [ [(n["lon"], n["lat"]) for n in m["geometry"]] for m in el.get("members", []) if m["type"] == "way" and m.get("role") != "inner" and "geometry" in m]
        inner_ways = [ [(n["lon"], n["lat"]) for n in m["geometry"]] for m in el.get("members", []) if m["type"] == "way" and m.get("role") == "inner" and "geometry" in m]
        
        outer_rings = stitch_ways(outer_ways)
        if not outer_rings: continue
        parts = [[r] for r in outer_rings]
        if inner_ways:
            inner_rings = stitch_ways(inner_ways)
            if inner_rings: parts[0].extend(inner_rings)
        
        geom = {"type": "Polygon", "coordinates": parts[0]} if len(parts) == 1 else {"type": "MultiPolygon", "coordinates": parts}
        features.append({
            "type": "Feature",
            "id": el["id"],
            "properties": {
                "osm_id": el["id"],
                "name": tags.get("name:es", tags.get("name", "")),
                "department": dept_name,
                "msme_count": 0
            },
            "geometry": geom
        })
    
    with open(temp_file, "w") as f: json.dump(features, f)
    print(f"  [{dept_name}] Saved {len(features)} districts")
    return features

def main():
    depts = fetch_departments()
    if not depts: sys.exit(1)
    
    all_features = []
    for dept in depts:
        all_features.extend(process_department(dept))
        time.sleep(1)

    with open(FINAL_OUTPUT, "w") as f: 
        json.dump({"type": "FeatureCollection", "features": all_features}, f)
    print(f"\n[Done] {len(all_features)} districts saved.")

if __name__ == "__main__":
    main()
