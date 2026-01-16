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
    /** Callback when a feature is hovered */
    onFeatureHover?: (feature: GeoJSON.Feature | null) => void;
    /** Additional CSS classes for the container */
    className?: string;
    /** Enable 3D terrain (future feature) */
    enable3D?: boolean;
}

const DEFAULT_CENTER: [number, number] = [-75.0152, -9.19];  // Peru center
const DEFAULT_ZOOM = 5;
const BASE_STYLE = 'https://tiles.openfreemap.org/styles/bright';

export default function Map({
    center = DEFAULT_CENTER,
    zoom = DEFAULT_ZOOM,
    sources = [],
    layers = [],
    onFeatureClick,
    onFeatureHover,
    className = '',
    enable3D = false,
}: MapProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const mapRef = useRef<maplibregl.Map | null>(null);
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

    // Add sources and layers when map is loaded
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !isLoaded) return;

        // Add sources
        sources.forEach((source) => {
            if (!map.getSource(source.id)) {
                map.addSource(source.id, {
                    type: 'geojson',
                    data: source.data,
                });
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
            }
        });
    }, [sources, layers, isLoaded]);

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
            if (!onFeatureHover) return;
            const features = map.queryRenderedFeatures(e.point);
            if (features.length > 0) {
                map.getCanvas().style.cursor = 'pointer';
                onFeatureHover(features[0] as unknown as GeoJSON.Feature);
            } else {
                map.getCanvas().style.cursor = '';
                onFeatureHover(null);
            }
        };

        map.on('click', handleClick);
        map.on('mousemove', handleMouseMove);

        return () => {
            map.off('click', handleClick);
            map.off('mousemove', handleMouseMove);
        };
    }, [isLoaded, onFeatureClick, onFeatureHover]);

    return (
        <div
            ref={mapContainer}
            className={`w-full h-full min-h-96 ${className}`}
        />
    );
}
