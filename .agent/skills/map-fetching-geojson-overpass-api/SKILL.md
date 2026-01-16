---
name: fetching-geojson-overpass-api
description: Provides robust strategies and query patterns for fetching administrative boundaries (GeoJSON) from OpenStreetMap via the Overpass API, specifically optimized for Peru's nested hierarchy.
---

# Fetching GeoJSON from Overpass API

This skill provides verified patterns for retrieving high-quality administrative boundaries from OpenStreetMap (OSM) using the Overpass API. It addresses common issues like nested hierarchies, timeouts, and incorrect relation IDs.

## Core Strategy: Area-Based Queries

For large regions or departments, direct parent-child relation searches (like `rel(r)`) often fail or return incomplete results if the hierarchy is deep (e.g., Department -> Province -> District). 

**The most robust approach is using Area-Based Queries:**

1.  **Find the Relation ID**: Locate the OSM Relation ID for the parent entity (e.g., Peru or Loreto Department).
2.  **Convert to Area ID**: Add `3600000000` to the Relation ID to get its Overpass Area ID.
3.  **Query by Area**:
    ```overpass
    [out:json][timeout:900];
    area(3600000000 + rel_id)->.parentArea;
    rel(area.parentArea)["admin_level"="8"]; // Change level as needed
    out body geom;
    ```

## Peru-Specific Reference

| Entity | Relation ID | Admin Level | Notes |
|--------|-------------|-------------|-------|
| Peru | 288247 | 2 | National boundary |
| Loreto | 1994077 | 4 | Often has nested provinces |
| Lima | 1891287 | 4 | High density, use longer timeouts |

## Handling Overpass API Challenges

### 1. Timeouts (HTTP 504)
- **Increase Timeout**: Set `[timeout:1800]` inside the query and also in your HTTP client (e.g., `urllib` or `requests`).
- **Use Mirrors**: Cycle through multiple mirrors for reliability:
  - `https://overpass-api.de/api/interpreter`
  - `https://overpass.kumi.systems/api/interpreter`
  - `https://overpass.osm.ch/api/interpreter`

### 2. Geometry Assembly (MultiPolygons)
When processing the JSON output, ensure you stitch "outer" and "inner" ways correctly.
- Use a `stitch_ways` utility to join coordinate segments into closed rings.
- Ensure "inner" rings (holes) are correctly nested within their corresponding "outer" rings in the GeoJSON structure.

## Best Practices

- **Caching**: Always cache raw JSON responses to avoid redundant expensive API calls.
- **Incremental Fetching**: For large countries, fetch by Department (admin_level 4) and then aggregate, rather than fetching all districts in one massive query.
- **Normalization**: Normalize names for matching, as OSM names may include "Distrito de..." or differ from official INEI spellings.

## Resources and Examples

- **Robust Fetch Script**: [robust_fetch.py](file:///Users/leonardoleon/Library/Mobile%20Documents/com~apple~CloudDocs/Universidad/UPC/9no%20ciclo/Tesis%201/.agent/skills/fetching-geojson-overpass-api/examples/robust_fetch.py) - A production-ready Python script demonstrating area-based fetching and geometry stitching.
- **Common Peru IDs**: [common_peru_relations.json](file:///Users/leonardoleon/Library/Mobile%20Documents/com~apple~CloudDocs/Universidad/UPC/9no%20ciclo/Tesis%201/.agent/skills/fetching-geojson-overpass-api/resources/common_peru_relations.json) - Verified OSM relation IDs for Peru and all its departments.
