/**
 * @fileoverview Type definitions for map data module.
 * @module map-data/types
 */

export interface DistrictNode {
    id: number;
    lat: number;
    lon: number;
    tags?: Record<string, string>;
}

export interface DistrictWay {
    id: number;
    nodes: number[];
    tags?: Record<string, string>;
}

export interface DistrictBounds {
    minlat: number;
    minlon: number;
    maxlat: number;
    maxlon: number;
}

export interface DistrictGraph {
    nodes: DistrictNode[];
    ways: DistrictWay[];
    bounds?: DistrictBounds;
    metadata?: {
        fetchedAt: string;
        relationId: string;
        source: string;
    };
}

export interface DistrictLocation {
    country: string;
    province: string;
    district: string;
    relationId: string;
}

export interface DistrictListItem {
    name: string;
    type: 'country' | 'province' | 'district';
    path?: string;
    children?: DistrictListItem[];
}

export interface DistrictListResponse {
    countries: DistrictListItem[];
}
