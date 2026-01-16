/**
 * @fileoverview Service layer for map data operations.
 * @module map-data/map-data.service
 */

import type { DistrictGraph, DistrictLocation, DistrictListItem } from './types.js';
import { fetchFromOverpass } from './overpass-client.js';
import { mapExists, readMap, writeMap, deleteMap, listDistricts } from './storage.js';

/**
 * Gets a district map, fetching from Overpass if not cached.
 */
export async function getDistrict(location: DistrictLocation): Promise<DistrictGraph> {
    // Check cache first
    const cached = await readMap(location);
    if (cached) {
        console.log(`[MapData] Cache hit for ${location.country}/${location.province}/${location.district}`);
        return cached;
    }
    
    // Fetch from Overpass
    console.log(`[MapData] Cache miss, fetching from Overpass...`);
    
    if (!location.relationId) {
        throw new Error('relationId required to fetch from Overpass');
    }
    
    const graph = await fetchFromOverpass(location.relationId);
    
    // Save to cache
    await writeMap(location, graph);
    
    return graph;
}

/**
 * Forces a refresh of district data from Overpass.
 */
export async function refreshDistrict(location: DistrictLocation): Promise<DistrictGraph> {
    console.log(`[MapData] Force refresh for ${location.country}/${location.province}/${location.district}`);
    
    const graph = await fetchFromOverpass(location.relationId);
    await writeMap(location, graph);
    
    return graph;
}

/**
 * Deletes a cached district.
 */
export async function removeDistrict(location: DistrictLocation): Promise<boolean> {
    return deleteMap(location);
}

/**
 * Lists all saved districts.
 */
export async function getAllDistricts(): Promise<DistrictListItem[]> {
    return listDistricts();
}

/**
 * Checks if a district is cached.
 */
export async function districtExists(location: DistrictLocation): Promise<boolean> {
    return mapExists(location);
}
