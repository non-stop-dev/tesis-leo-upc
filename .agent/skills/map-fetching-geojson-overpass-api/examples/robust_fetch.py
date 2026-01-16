#!/usr/bin/env python3
"""
Robust GeoJSON Fetcher for Overpass API
This script demonstrates the 'area-based' query pattern for stable district fetching.
"""
import json
import time
import urllib.request
import urllib.parse

# Verified Mirrors
MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter"
]

def stitch_ways(ways):
    """Join coordinates into closed rings."""
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

def query_overpass(query, timeout=900):
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    for mirror in MIRRORS:
        print(f"[*] Trying {mirror}...")
        try:
            req = urllib.request.Request(mirror, data=data)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            print(f"[!] Error with mirror {mirror}: {e}")
            continue
    return None

def fetch_districts_by_area(dept_rel_id, dept_name):
    # AREA ID = 3,600,000,000 + RELATION ID
    area_id = 3600000000 + dept_rel_id
    query = f"""
    [out:json][timeout:1800];
    area({area_id})->.searchArea;
    rel(area.searchArea)["admin_level"="8"];
    out body geom;
    """
    
    data = query_overpass(query)
    if not data or "elements" not in data:
        print(f"[!] No data found for {dept_name}")
        return []

    features = []
    for el in data["elements"]:
        if el["type"] != "relation": continue
        tags = el.get("tags", {})
        
        # Geometry Assembly
        members = el.get("members", [])
        outer_ways = [[(n["lon"], n["lat"]) for n in m["geometry"]] for m in members if m["type"] == "way" and m.get("role") != "inner" and "geometry" in m]
        inner_ways = [[(n["lon"], n["lat"]) for n in m["geometry"]] for m in members if m["type"] == "way" and m.get("role") == "inner" and "geometry" in m]
        
        outer_rings = stitch_ways(outer_ways)
        if not outer_rings: continue
        
        inner_rings = stitch_ways(inner_ways)
        
        # MultiPolygon Support
        if len(outer_rings) > 1:
            # Simple heuristic: assuming first outer ring is the main part
            coords = [[r] for r in outer_rings]
            if inner_rings: coords[0].extend(inner_rings)
            geom_type = "MultiPolygon"
        else:
            coords = [outer_rings[0]] + inner_rings
            geom_type = "Polygon"

        features.append({
            "type": "Feature",
            "id": el["id"],
            "properties": {
                "osm_id": el["id"],
                "name": tags.get("name:es", tags.get("name", "")),
                "department": dept_name,
                "ubigeo": tags.get("ref:INEI", "")
            },
            "geometry": {"type": geom_type, "coordinates": coords}
        })
    
    return features

if __name__ == "__main__":
    # Example: Loreto (1994077)
    loreto_districts = fetch_districts_by_area(1994077, "Loreto")
    print(f"[+] Successfully fetched {len(loreto_districts)} districts for Loreto.")
    with open("loreto_districts.geojson", "w") as f:
        json.dump({{"type": "FeatureCollection", "features": loreto_districts}}, f)
