/**
 * @fileoverview Reusable MapLibre GL JS Map component
 * @module components/map/Map
 */

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export interface MapLayer {
    id: string;
    type: 'fill' | 'line' | 'circle' | 'heatmap';
    source: string;
    paint: Record<string, unknown>;
    filter?: unknown[];
}

export interface MapSource {
    id: string;
    type: 'geojson';
    data: string | GeoJSON.FeatureCollection;
}

export interface MapProps {
    /** Initial center coordinates [lng, lat] */
    center?: [number, number];
    /** Initial zoom level */
    zoom?: number;
    /** GeoJSON sources to add to the map */
    sources?: MapSource[];
    /** Layers to render */
    layers?: MapLayer[];
    /** Callback when a feature is clicked */
    onFeatureClick?: (feature: GeoJSON.Feature) => void;
    /** Callback when a feature is hovered. Returns feature and pixel coords. */
    onFeatureHover?: (feature: GeoJSON.Feature | null, x: number, y: number) => void;
    /** Source ID to apply high-performance feature-state hover highlight */
    hoverSourceId?: string;
    /** Callback when map becomes idle after rendering (use for loading states) */
    onIdle?: () => void;
    /** Additional CSS classes for the container */
    className?: string;
    /** Enable 3D terrain (future feature) */
    enable3D?: boolean;
}

const DEFAULT_CENTER: [number, number] = [-75.0152, -9.19];  // Peru center
const DEFAULT_ZOOM = 4.8;
const BASE_STYLE = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';

export default function Map({
    center = DEFAULT_CENTER,
    zoom = DEFAULT_ZOOM,
    sources = [],
    layers = [],
    onFeatureClick,
    onFeatureHover,
    hoverSourceId,
    onIdle,
    className = '',
    enable3D = false,
}: MapProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const mapRef = useRef<maplibregl.Map | null>(null);
    const hoveredIdRef = useRef<string | number | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);

    // Initialize map
    useEffect(() => {
        if (!mapContainer.current || mapRef.current) return;

        const map = new maplibregl.Map({
            container: mapContainer.current,
            style: BASE_STYLE,
            center,
            zoom,
            pitch: enable3D ? 45 : 0,
        });

        map.addControl(new maplibregl.NavigationControl(), 'top-right');

        map.on('load', () => {
            setIsLoaded(true);
        });

        mapRef.current = map;

        return () => {
            map.remove();
            mapRef.current = null;
        };
    }, [center, zoom, enable3D]);

    // Handle high-performance feature highlighting
    const setHoverState = (id: string | number | null, hovered: boolean) => {
        const map = mapRef.current;
        if (!map || !hoverSourceId || id === null) return;
        try {
            map.setFeatureState(
                { source: hoverSourceId, id },
                { hover: hovered }
            );
        } catch (e) {
            // Silently ignore if source or feature doesn't exist yet
        }
    };

    // Add sources and layers when map is loaded
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !isLoaded) return;

        // Add/Update sources
        sources.forEach((source) => {
            const existingSource = map.getSource(source.id) as maplibregl.GeoJSONSource;
            if (!existingSource) {
                map.addSource(source.id, {
                    type: 'geojson',
                    data: source.data,
                    // Critical for feature-state: promoteId ensures unique IDs are used
                    // We use osm_id which is present in both department and district maps
                    promoteId: 'osm_id' 
                });
            } else {
                existingSource.setData(source.data);
            }
        });

        // Add layers
        layers.forEach((layer) => {
            if (!map.getLayer(layer.id)) {
                map.addLayer({
                    id: layer.id,
                    type: layer.type,
                    source: layer.source,
                    paint: layer.paint,
                    ...(layer.filter && { filter: layer.filter }),
                } as maplibregl.LayerSpecification);
            } else {
                // Update properties for existing layers
                const existingLayer = map.getLayer(layer.id);
                if (existingLayer) {
                    // Update paint properties
                    Object.entries(layer.paint).forEach(([prop, value]) => {
                        map.setPaintProperty(layer.id, prop, value);
                    });
                    
                    // Update filters
                    if (layer.filter && !layer.id.includes('hover')) {
                        map.setFilter(layer.id, layer.filter as any);
                    }
                }
            }
        });
    }, [sources, layers, isLoaded]);

    // Expose onIdle callback for parent loading state management
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !isLoaded || !onIdle) return;

        const handleIdle = () => {
            onIdle();
        };

        map.on('idle', handleIdle);

        return () => {
            map.off('idle', handleIdle);
        };
    }, [isLoaded, onIdle]);

    // Event handlers
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !isLoaded) return;

        const handleClick = (e: maplibregl.MapMouseEvent) => {
            if (!onFeatureClick) return;
            const features = map.queryRenderedFeatures(e.point);
            if (features.length > 0) {
                onFeatureClick(features[0] as unknown as GeoJSON.Feature);
            }
        };

        const handleMouseMove = (e: maplibregl.MapMouseEvent) => {
            const features = map.queryRenderedFeatures(e.point, { layers: ['choropleth-fill'] });
            
            if (features.length > 0) {
                const feature = features[0];
                const newId = feature.id;

                map.getCanvas().style.cursor = 'pointer';

                // High performance highlighting (Direct MapLibre Call)
                if (newId !== hoveredIdRef.current) {
                    setHoverState(hoveredIdRef.current, false);
                    hoveredIdRef.current = newId;
                    setHoverState(newId, true);
                }

                // Notify parent ONLY of identity/data change
                if (onFeatureHover) {
                    onFeatureHover(feature as unknown as GeoJSON.Feature, e.point.x, e.point.y);
                }
            } else {
                map.getCanvas().style.cursor = '';
                
                if (hoveredIdRef.current !== null) {
                    setHoverState(hoveredIdRef.current, false);
                    hoveredIdRef.current = null;
                }

                if (onFeatureHover) {
                    onFeatureHover(null, 0, 0);
                }
            }
        };

        const handleMouseLeave = () => {
            if (hoveredIdRef.current !== null) {
                setHoverState(hoveredIdRef.current, false);
                hoveredIdRef.current = null;
            }
            if (onFeatureHover) {
                onFeatureHover(null, 0, 0);
            }
        };

        map.on('click', handleClick);
        map.on('mousemove', handleMouseMove);
        map.on('mouseleave', handleMouseLeave);

        return () => {
            map.off('click', handleClick);
            map.off('mousemove', handleMouseMove);
            map.off('mouseleave', handleMouseLeave);
        };
    }, [isLoaded, onFeatureClick, onFeatureHover, hoverSourceId]);


    return (
        <div
            ref={mapContainer}
            className={`w-full h-full min-h-96 ${className}`}
        />
    );
}
