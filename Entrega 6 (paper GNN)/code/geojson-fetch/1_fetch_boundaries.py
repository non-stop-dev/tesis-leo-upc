#!/usr/bin/env python3
"""
Fetch Peru administrative boundaries from Overpass API.

This script downloads department, province, and district boundaries
and saves them as GeoJSON files.

Output:
    - public/map-geojson/peru_departments.geojson
    - public/map-geojson/peru_provinces.geojson  
    - public/map-geojson/peru_districts.geojson
"""

import json
import os
import sys
from typing import Any
import requests

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# Peru boundary relation ID
PERU_RELATION_ID = "288247"

def build_boundary_query(admin_level: int) -> str:
    """
    Build Overpass QL query for administrative boundaries.
    
    Admin levels in Peru:
    - 4: Departments (Regiones)
    - 6: Provinces (Provincias)
    - 8: Districts (Distritos)
    """
    return f"""
        [out:json][timeout:300];
        area["ISO3166-1"="PE"]->.peru;
        (
          relation["admin_level"="{admin_level}"]["boundary"="administrative"](area.peru);
        );
        out geom;
    """


def parse_relation_to_feature(element: dict[str, Any]) -> dict[str, Any] | None:
    """Convert Overpass relation element to GeoJSON Feature."""
    if element.get("type") != "relation":
        return None
    
    tags = element.get("tags", {})
    members = element.get("members", [])
    
    # Extract outer ways to build polygon
    outer_coords: list[list[list[float]]] = []
    
    for member in members:
        if member.get("role") == "outer" and member.get("type") == "way":
            geometry = member.get("geometry", [])
            if geometry:
                coords = [[pt["lon"], pt["lat"]] for pt in geometry]
                outer_coords.append(coords)
    
    if not outer_coords:
        return None
    
    # Build geometry
    if len(outer_coords) == 1:
        geometry = {
            "type": "Polygon",
            "coordinates": outer_coords
        }
    else:
        geometry = {
            "type": "MultiPolygon",
            "coordinates": [[ring] for ring in outer_coords]
        }
    
    # Extract UBIGEO from tags (if available)
    ubigeo = tags.get("ref:INEI", tags.get("ref", ""))
    
    properties = {
        "osm_id": element.get("id"),
        "name": tags.get("name", ""),
        "name_es": tags.get("name:es", tags.get("name", "")),
        "ubigeo": ubigeo,
        "admin_level": tags.get("admin_level", ""),
        "wikipedia": tags.get("wikipedia", ""),
    }
    
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": geometry
    }


def fetch_boundaries(admin_level: int, name: str) -> dict[str, Any]:
    """Fetch boundaries from Overpass API."""
    print(f"[Overpass] Fetching {name} (admin_level={admin_level})...")
    
    query = build_boundary_query(admin_level)
    
    response = requests.post(
        OVERPASS_API_URL,
        data={"data": query},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=600
    )
    
    if not response.ok:
        print(f"[Error] Overpass API returned {response.status_code}")
        sys.exit(1)
    
    data = response.json()
    elements = data.get("elements", [])
    print(f"[Overpass] Received {len(elements)} elements")
    
    features = []
    for element in elements:
        feature = parse_relation_to_feature(element)
        if feature:
            features.append(feature)
    
    print(f"[GeoJSON] Created {len(features)} features")
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


def main():
    # Output directory
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "visualization", "public", "map-geojson"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    boundaries = [
        (4, "departments", "peru_departments.geojson"),
        (6, "provinces", "peru_provinces.geojson"),
        (8, "districts", "peru_districts.geojson"),
    ]
    
    for admin_level, name, filename in boundaries:
        geojson = fetch_boundaries(admin_level, name)
        
        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)
        
        print(f"[Saved] {output_path}")
        print()


if __name__ == "__main__":
    main()
