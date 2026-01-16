/**
 * @fileoverview File system storage for district map data.
 * @module map-data/storage
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import type { DistrictGraph, DistrictLocation, DistrictListItem } from './types.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Storage root: backend/src/map-data/osm-data/
const OSM_DATA_ROOT = path.resolve(__dirname, './osm-data');

/**
 * Gets the file path for a district's map data.
 */
function getMapPath(location: DistrictLocation): string {
    const targetPath = path.resolve(
        OSM_DATA_ROOT,
        location.country,
        location.province,
        location.district,
        'map.json'
    );

    // Security check: ensure path is within OSM_DATA_ROOT
    const relative = path.relative(OSM_DATA_ROOT, targetPath);
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
        throw new Error('Security Error: Path traversal detected');
    }

    return targetPath;
}

/**
 * Checks if a district map exists on disk.
 */
export async function mapExists(location: DistrictLocation): Promise<boolean> {
    try {
        await fs.access(getMapPath(location));
        return true;
    } catch {
        return false;
    }
}

/**
 * Reads a district map from disk.
 */
export async function readMap(location: DistrictLocation): Promise<DistrictGraph | null> {
    try {
        const mapPath = getMapPath(location);
        const data = await fs.readFile(mapPath, 'utf-8');
        return JSON.parse(data) as DistrictGraph;
    } catch {
        return null;
    }
}

/**
 * Writes a district map to disk.
 */
export async function writeMap(location: DistrictLocation, graph: DistrictGraph): Promise<void> {
    const mapPath = getMapPath(location);
    const dir = path.dirname(mapPath);
    
    // Create directory structure
    await fs.mkdir(dir, { recursive: true });
    
    // Write JSON file
    await fs.writeFile(mapPath, JSON.stringify(graph, null, 2), 'utf-8');
    
    console.log(`[Storage] Saved map to ${mapPath}`);
}

/**
 * Deletes a district map from disk.
 */
export async function deleteMap(location: DistrictLocation): Promise<boolean> {
    try {
        const mapPath = getMapPath(location);
        await fs.unlink(mapPath);
        
        // Try to clean up empty directories
        const districtDir = path.dirname(mapPath);
        const provinceDir = path.dirname(districtDir);
        const countryDir = path.dirname(provinceDir);
        
        try { await fs.rmdir(districtDir); } catch { /* not empty */ }
        try { await fs.rmdir(provinceDir); } catch { /* not empty */ }
        try { await fs.rmdir(countryDir); } catch { /* not empty */ }
        
        return true;
    } catch {
        return false;
    }
}

/**
 * Lists all saved districts as a tree structure.
 */
export async function listDistricts(): Promise<DistrictListItem[]> {
    const countries: DistrictListItem[] = [];
    
    try {
        // Ensure root exists
        await fs.mkdir(OSM_DATA_ROOT, { recursive: true });
        
        const countryDirs = await fs.readdir(OSM_DATA_ROOT, { withFileTypes: true });
        
        for (const countryDir of countryDirs) {
            if (!countryDir.isDirectory()) continue;
            
            const countryNode: DistrictListItem = {
                name: countryDir.name,
                type: 'country',
                children: []
            };
            
            const provincePath = path.join(OSM_DATA_ROOT, countryDir.name);
            const provinceDirs = await fs.readdir(provincePath, { withFileTypes: true });
            
            for (const provinceDir of provinceDirs) {
                if (!provinceDir.isDirectory()) continue;
                
                const provinceNode: DistrictListItem = {
                    name: provinceDir.name,
                    type: 'province',
                    children: []
                };
                
                const districtPath = path.join(provincePath, provinceDir.name);
                const districtDirs = await fs.readdir(districtPath, { withFileTypes: true });
                
                for (const districtDir of districtDirs) {
                    if (!districtDir.isDirectory()) continue;
                    
                    // Check if map.json exists
                    const mapPath = path.join(districtPath, districtDir.name, 'map.json');
                    try {
                        await fs.access(mapPath);
                        provinceNode.children!.push({
                            name: districtDir.name,
                            type: 'district',
                            path: `${countryDir.name}/${provinceDir.name}/${districtDir.name}`
                        });
                    } catch {
                        // No map.json, skip
                    }
                }
                
                if (provinceNode.children!.length > 0) {
                    countryNode.children!.push(provinceNode);
                }
            }
            
            if (countryNode.children!.length > 0) {
                countries.push(countryNode);
            }
        }
    } catch (err) {
        console.warn('[Storage] Error listing districts:', err);
    }
    
    return countries;
}
