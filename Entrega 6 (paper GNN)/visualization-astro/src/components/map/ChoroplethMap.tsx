/**
 * @fileoverview Choropleth map for MSME density visualization
 * @module components/map/ChoroplethMap
 */

import { useState, useRef, useEffect, useCallback } from 'react';
import Map, { type MapSource, type MapLayer } from './Map';

export interface ChoroplethData {
    geojson: string | GeoJSON.FeatureCollection;
    valueProperty: string;
    colorStops: [number, string][];
    labelProperty?: string;
    level?: 'DEPT' | 'PROV' | 'DIST';
}

export interface ChoroplethMapProps {
    data?: ChoroplethData; // Now optional if we use internal level state
    title?: string;
    legendTitle?: string;
    className?: string;
}

export default function ChoroplethMap({
    data: initialData,
    title,
    legendTitle = 'Density',
    className = '',
}: ChoroplethMapProps) {
    const [level, setLevel] = useState<'DEPT' | 'PROV' | 'DIST'>('DIST');
    const [hoveredFeature, setHoveredFeature] = useState<GeoJSON.Feature | null>(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const [isLoading, setIsLoading] = useState(false);
    const prevLevelRef = useRef<'DEPT' | 'PROV' | 'DIST'>(level);

    // Trigger loading state when level changes
    useEffect(() => {
        if (prevLevelRef.current !== level) {
            setIsLoading(true);
            prevLevelRef.current = level;
        }
    }, [level]);

    // Callback when map finishes rendering
    const handleMapIdle = useCallback(() => {
        if (isLoading) {
            setIsLoading(false);
        }
    }, [isLoading]);

    // Configuration for different levels
    const levelConfigs = {
        DEPT: {
            geojson: '/map-geojson/peru_departments_msme.geojson',
            valueProperty: 'msme_count',
            labelProperty: 'name',
            titleSuffix: 'Departamentos',
            colorStops: [
                [0, '#ffffcc'],
                [10000, '#a1dab4'],
                [50000, '#41b6c4'],
                [150000, '#2c7fb8'],
                [500000, '#253494'],
            ] as [number, string][],
        },
        PROV: {
            geojson: '/map-geojson/peru_provinces_msme.geojson',
            valueProperty: 'msme_count',
            labelProperty: 'name',
            titleSuffix: 'Provincias',
            colorStops: [
                [0, '#ffffcc'],
                [1000, '#a1dab4'],
                [5000, '#41b6c4'],
                [20000, '#2c7fb8'],
                [100000, '#253494'],
            ] as [number, string][],
        },
        DIST: {
            geojson: '/map-geojson/peru_districts_msme.geojson',
            valueProperty: 'msme_count',
            labelProperty: 'name',
            titleSuffix: 'Distritos',
            colorStops: [
                [0, '#ffffcc'],
                [100, '#a1dab4'],
                [500, '#41b6c4'],
                [2000, '#2c7fb8'],
                [10000, '#253494'],
            ] as [number, string][],
        }
    };

    const currentConfig = levelConfigs[level];
    const data = initialData || currentConfig;

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
        // Highlight layer for hover (using high-performance feature-state)
        {
            id: 'choropleth-hover',
            type: 'line',
            source: 'choropleth-data',
            paint: {
                'line-color': '#ffffff',
                'line-width': [
                    'case',
                    ['boolean', ['feature-state', 'hover'], false],
                    3,
                    0
                ],
            },
        }
    ];

    const handleFeatureHover = (feature: GeoJSON.Feature | null, x: number, y: number) => {
        setMousePos({ x, y });
        setHoveredFeature(feature);
    };

    const getTooltipData = () => {
        if (!hoveredFeature?.properties) return null;
        const value = hoveredFeature.properties[data.valueProperty];
        const label = hoveredFeature.properties[data.labelProperty || 'name'] || 
                     hoveredFeature.properties['name_es'] || 
                     hoveredFeature.properties['nombre_departamento'] ||
                     hoveredFeature.properties['nombre_distrito'] ||
                     'Unknown';
        return { label, value };
    };

    const tooltip = getTooltipData();

    return (
        <div className={`relative ${className}`}>
            <div className="absolute top-4 left-4 z-10 flex flex-col gap-2">
                {title && (
                    <h2 className="bg-white/95 px-4 py-2 rounded-xl shadow-lg text-lg font-bold text-slate-800 border border-slate-200">
                        {title}
                        <span className="ml-2 text-blue-500 text-sm font-medium">({currentConfig.titleSuffix})</span>
                    </h2>
                )}
                
                {/* Level Selector */}
                <div className="bg-white/95 p-1 rounded-xl shadow-lg border border-slate-200 flex gap-1 w-fit">
                    {(['DEPT', 'PROV', 'DIST'] as const).map((l) => (
                        <button
                            key={l}
                            onClick={() => setLevel(l)}
                            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                                level === l 
                                ? 'bg-blue-600 text-white shadow-md' 
                                : 'text-slate-500 hover:bg-slate-100 hover:text-slate-700'
                            }`}
                        >
                            {l === 'DEPT' ? 'Departamentos' : l === 'PROV' ? 'Provincias' : 'Distritos'}
                        </button>
                    ))}
                </div>
            </div>

            {/* Floating Tooltip */}
            {tooltip && (
                <div 
                    className="absolute z-50 pointer-events-none bg-slate-900/90 text-white px-3 py-2 rounded shadow-xl text-sm backdrop-blur-sm border border-slate-700 transition-all duration-75"
                    style={{ 
                        left: mousePos.x + 15, 
                        top: mousePos.y + 15,
                        transform: 'translate(0, 0)'
                    }}
                >
                    <p className="font-bold border-b border-slate-700 pb-1 mb-1">{tooltip.label}</p>
                    <div className="flex justify-between gap-4">
                      <span className="text-slate-400">Empresas:</span>
                      <span className="font-mono">
                          {typeof tooltip.value === 'number'
                              ? tooltip.value.toLocaleString()
                              : '0'}
                      </span>
                    </div>
                </div>
            )}

            {/* Loading Overlay - z-[5] keeps it behind controls (z-10) but covers map */}
            {isLoading && (
                <div className="absolute inset-0 z-[5] bg-white/60 backdrop-blur-sm flex items-end justify-center pb-20 transition-opacity duration-200">
                    <div className="flex items-center gap-3 bg-white/95 px-5 py-3 rounded-full shadow-xl border border-slate-200">
                        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                        <p className="text-sm font-medium text-slate-600">Cargando {currentConfig.titleSuffix}...</p>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="absolute bottom-6 left-6 z-10 bg-white/95 px-4 py-3 rounded-xl shadow-lg border border-slate-200 transition-opacity duration-300">
                <p className="text-[10px] uppercase tracking-wider font-bold text-slate-500 mb-2">{legendTitle}</p>
                <div className="flex items-center">
                    {data.colorStops.map(([value, color], i) => (
                        <div key={i} className="flex flex-col items-center">
                            <div
                                className="w-8 h-3 first:rounded-l-full last:rounded-r-full"
                                style={{ backgroundColor: color }}
                            />
                            <span className="text-[10px] font-medium text-slate-400 mt-1">
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
                onIdle={handleMapIdle}
                hoverSourceId="choropleth-data"
                className="w-full h-full"
            />
        </div>
    );
}
