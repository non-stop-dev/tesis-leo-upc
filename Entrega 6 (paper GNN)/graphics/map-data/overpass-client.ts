/**
 * @fileoverview Overpass API client for fetching OSM data.
 * @module map-data/overpass-client
 */

import type { DistrictGraph, DistrictNode, DistrictWay } from './types.js';

const OVERPASS_API_URL = 'https://overpass-api.de/api/interpreter';

/**
 * Builds Overpass QL query for a district by relation ID.
 */
function buildOverpassQuery(relationId: string): string {
    return `
        [out:json][timeout:180];
        relation(${relationId})->._relDistrict;
        ._relDistrict map_to_area -> .district;
        ._relDistrict out geom;
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|living_street|pedestrian|footway|cycleway)$"](area.district);
          node["highway"~"^(traffic_signals|crossing)$"](area.district);
        );
        out body;
        >;
        out skel qt;
    `;
}

/**
 * Parses Overpass API response into DistrictGraph format.
 */
function parseOverpassResponse(data: any, relationId: string): DistrictGraph {
    const nodes: DistrictNode[] = [];
    const ways: DistrictWay[] = [];

    if (!data || !data.elements) {
        return { 
            nodes: [], 
            ways: [],
            metadata: {
                fetchedAt: new Date().toISOString(),
                relationId,
                source: 'overpass-api.de'
            }
        };
    }

    for (const element of data.elements) {
        if (element.type === 'node') {
            nodes.push({
                id: element.id,
                lat: element.lat,
                lon: element.lon,
                tags: element.tags,
            });
        } else if (element.type === 'way') {
            ways.push({
                id: element.id,
                nodes: element.nodes,
                tags: element.tags,
            });
        }
    }

    // Calculate bounds
    let minlat = 90, minlon = 180, maxlat = -90, maxlon = -180;

    if (nodes.length > 0) {
        nodes.forEach(n => {
            if (n.lat < minlat) minlat = n.lat;
            if (n.lon < minlon) minlon = n.lon;
            if (n.lat > maxlat) maxlat = n.lat;
            if (n.lon > maxlon) maxlon = n.lon;
        });
    } else {
        minlat = 0; minlon = 0; maxlat = 0; maxlon = 0;
    }

    return {
        nodes,
        ways,
        bounds: { minlat, minlon, maxlat, maxlon },
        metadata: {
            fetchedAt: new Date().toISOString(),
            relationId,
            source: 'overpass-api.de'
        }
    };
}

/**
 * Fetches district data from Overpass API.
 * 
 * @param relationId - OSM relation ID for the district
 * @returns Parsed DistrictGraph
 */
export async function fetchFromOverpass(relationId: string): Promise<DistrictGraph> {
    const query = buildOverpassQuery(relationId);
    
    console.log(`[Overpass] Fetching relation ${relationId}...`);
    
    const response = await fetch(OVERPASS_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: `data=${encodeURIComponent(query)}`
    });

    if (!response.ok) {
        throw new Error(`Overpass API error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    const graph = parseOverpassResponse(data, relationId);
    
    console.log(`[Overpass] Fetched ${graph.nodes.length} nodes, ${graph.ways.length} ways`);
    
    return graph;
}
