/**
 * @fileoverview Choropleth map for MSME density visualization
 * @module components/map/ChoroplethMap
 */

import { useState } from 'react';
import Map, { type MapSource, type MapLayer } from './Map';

export interface ChoroplethData {
    /** GeoJSON URL or inline data */
    geojson: string | GeoJSON.FeatureCollection;
    /** Property name containing the value to visualize */
    valueProperty: string;
    /** Color stops for interpolation [value, color][] */
    colorStops: [number, string][];
    /** Property name for feature label/tooltip */
    labelProperty?: string;
}

export interface ChoroplethMapProps {
    /** Choropleth configuration */
    data: ChoroplethData;
    /** Map title */
    title?: string;
    /** Legend title */
    legendTitle?: string;
    /** Additional CSS classes */
    className?: string;
}

export default function ChoroplethMap({
    data,
    title,
    legendTitle = 'Density',
    className = '',
}: ChoroplethMapProps) {
    const [hoveredFeature, setHoveredFeature] = useState<GeoJSON.Feature | null>(null);

    const sources: MapSource[] = [
        {
            id: 'choropleth-data',
            type: 'geojson',
            data: data.geojson,
        },
    ];

    const layers: MapLayer[] = [
        {
            id: 'choropleth-fill',
            type: 'fill',
            source: 'choropleth-data',
            paint: {
                'fill-color': [
                    'interpolate',
                    ['linear'],
                    ['get', data.valueProperty],
                    ...data.colorStops.flat(),
                ],
                'fill-opacity': 0.7,
            },
        },
        {
            id: 'choropleth-outline',
            type: 'line',
            source: 'choropleth-data',
            paint: {
                'line-color': '#1e293b',
                'line-width': 0.5,
            },
        },
    ];

    const handleFeatureHover = (feature: GeoJSON.Feature | null) => {
        setHoveredFeature(feature);
    };

    const getTooltipValue = () => {
        if (!hoveredFeature?.properties) return null;
        const value = hoveredFeature.properties[data.valueProperty];
        const label = data.labelProperty
            ? hoveredFeature.properties[data.labelProperty]
            : 'Region';
        return { label, value };
    };

    const tooltipData = getTooltipValue();

    return (
        <div className={`relative ${className}`}>
            {title && (
                <h2 className="absolute top-4 left-4 z-10 bg-white/90 px-3 py-1.5 rounded-lg shadow text-lg font-semibold text-slate-800">
                    {title}
                </h2>
            )}

            {/* Tooltip */}
            {tooltipData && (
                <div className="absolute top-4 right-16 z-10 bg-white/95 px-4 py-2 rounded-lg shadow-lg text-sm">
                    <p className="font-medium text-slate-700">{tooltipData.label}</p>
                    <p className="text-slate-900">
                        {typeof tooltipData.value === 'number'
                            ? tooltipData.value.toLocaleString()
                            : tooltipData.value}
                    </p>
                </div>
            )}

            {/* Legend */}
            <div className="absolute bottom-8 left-4 z-10 bg-white/95 px-4 py-3 rounded-lg shadow-lg">
                <p className="text-xs font-medium text-slate-600 mb-2">{legendTitle}</p>
                <div className="flex items-center gap-1">
                    {data.colorStops.map(([value, color], i) => (
                        <div key={i} className="flex flex-col items-center">
                            <div
                                className="w-6 h-4"
                                style={{ backgroundColor: color }}
                            />
                            <span className="text-xs text-slate-500 mt-1">
                                {value >= 1000 ? `${(value / 1000).toFixed(0)}k` : value}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            <Map
                sources={sources}
                layers={layers}
                onFeatureHover={handleFeatureHover}
                className="w-full h-full"
            />
        </div>
    );
}
