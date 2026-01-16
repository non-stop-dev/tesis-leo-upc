/**
 * @fileoverview High-performance Map component using MapLibre feature-state.
 * This pattern avoids React re-renders and filter re-evaluations.
 */

import { useEffect, useRef, useState } from 'react';
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export interface MapHighPerformanceProps {
    sourceId: string;
    geojson: string | GeoJSON.FeatureCollection;
    promoteId: string;
    className?: string;
}

export default function MapHighPerformance({
    sourceId,
    geojson,
    promoteId,
    className = '',
}: MapHighPerformanceProps) {
    const mapContainer = useRef<HTMLDivElement>(null);
    const mapRef = useRef<maplibregl.Map | null>(null);
    const hoveredIdRef = useRef<string | number | null>(null);
    const [isLoaded, setIsLoaded] = useState(false);

    useEffect(() => {
        if (!mapContainer.current || mapRef.current) return;

        const map = new maplibregl.Map({
            container: mapContainer.current,
            style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
            center: [-75, -10],
            zoom: 5,
        });

        map.on('load', () => {
            // 1. Add Source with promoteId
            map.addSource(sourceId, {
                type: 'geojson',
                data: geojson,
                promoteId: promoteId // Critical: maps a property to top-level feature.id
            });

            // 2. Add Fill Layer
            map.addLayer({
                id: `${sourceId}-fill`,
                type: 'fill',
                source: sourceId,
                paint: {
                    'fill-color': '#3b82f6',
                    'fill-opacity': 0.5
                }
            });

            // 3. Add Hover Highlight Layer (Line)
            map.addLayer({
                id: `${sourceId}-hover`,
                type: 'line',
                source: sourceId,
                paint: {
                    'line-color': '#FFFFFF',
                    'line-width': [
                        'case',
                        ['boolean', ['feature-state', 'hover'], false],
                        3,
                        0
                    ]
                }
            });

            setIsLoaded(true);
        });

        // 4. Optimized Mouse Move Handler
        map.on('mousemove', `${sourceId}-fill`, (e) => {
            if (e.features && e.features.length > 0) {
                const newId = e.features[0].id;

                if (newId !== hoveredIdRef.current) {
                    // Reset previous hover state
                    if (hoveredIdRef.current !== null) {
                        map.setFeatureState(
                            { source: sourceId, id: hoveredIdRef.current },
                            { hover: false }
                        );
                    }

                    // Set new hover state
                    hoveredIdRef.current = newId;
                    map.setFeatureState(
                        { source: sourceId, id: newId },
                        { hover: true }
                    );
                    
                    map.getCanvas().style.cursor = 'pointer';
                }
            }
        });

        // 5. Clean Reset on Leave
        map.on('mouseleave', `${sourceId}-fill`, () => {
            if (hoveredIdRef.current !== null) {
                map.setFeatureState(
                    { source: sourceId, id: hoveredIdRef.current },
                    { hover: false }
                );
                hoveredIdRef.current = null;
            }
            map.getCanvas().style.cursor = '';
        });

        mapRef.current = map;
        return () => map.remove();
    }, [sourceId, geojson, promoteId]);

    return (
        <div ref={mapContainer} className={`w-full h-full min-h-[500px] ${className}`} />
    );
}
