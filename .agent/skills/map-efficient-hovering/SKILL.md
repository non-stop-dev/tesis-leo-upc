---
name: map-efficient-hovering
description: Implements ultra-efficient map hover interactions using MapLibre's feature-state. Use when building interactive maps that require smooth highlighting and minimal re-renders.
---

# Map Efficient Hovering Skill

This skill provides the technical patterns and best practices for implementing high-performance hover interactions in MapLibre GL JS, optimized for performance and visual fidelity.

## The Problem: React Overhead and Filter Latency

Traditional map hover implementations often rely on React state updates or `map.setFilter()`. These approaches have several drawbacks:
- **React Re-renders**: Updating React state on every mouse move causes costly component re-renders.
- **Filter Evaluation**: `setFilter` forces the map engine to re-evaluate the entire data source, leading to "stuttering" on dense GeoJSON maps.
- **Geometry Artifacts**: Features with unclosed rings or overlapping segments may only trigger hover events on their edges.

## The Solution: MapLibre `feature-state`

The `feature-state` mechanism allows for direct, targeted updates to individual features without triggering a full layer refresh or React cycle.

### 1. Source Configuration

To use `feature-state`, the data source MUST have a unique identifier for each feature. Use the `promoteId` property:

```typescript
map.addSource('my-source', {
    type: 'geojson',
    data: '/path/to/data.geojson',
    promoteId: 'osm_id' // Maps the property 'osm_id' to the top-level 'id'
});
```

### 2. High-Performance Event Handling

Avoid React state for the *visual* highlight. Handle it directly in the mouse events using `map.setFeatureState`. Use a `useRef` to track the currently hovered ID to avoid redundant calls.

```typescript
const hoveredIdRef = useRef<string | number | null>(null);

const handleMouseMove = (e) => {
    const features = map.queryRenderedFeatures(e.point, { layers: ['fill-layer'] });
    
    if (features.length > 0) {
        const newId = features[0].id; // Extracted via promoteId
        
        if (newId !== hoveredIdRef.current) {
            // Remove highlight from previous feature
            if (hoveredIdRef.current !== null) {
                map.setFeatureState(
                    { source: 'my-source', id: hoveredIdRef.current },
                    { hover: false }
                );
            }
            
            // Add highlight to new feature
            hoveredIdRef.current = newId;
            map.setFeatureState(
                { source: 'my-source', id: newId },
                { hover: true }
            );
        }
    }
};
```

### 3. Paint Expressions

Use expressions in your layer styling to react to the `feature-state`:

```javascript
map.addLayer({
    id: 'hover-highlight',
    type: 'line',
    source: 'my-source',
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
```

## Geometry Integrity (Way-Stitching)

If hover only works on edges, it usually means the GeoJSON polygons are "open" (nodes are connected as lines but not closed as rings). 

**Best Practices:**
- Use a **way-stitching** algorithm to ensure all rings are closed (first point == last point).
- Convert `LineString` segments into `Polygon` rings during preprocessing.
- Ensure every feature has a numeric `id` at the top level or via `promoteId`.

## How to use this skill

1. **Verify Geometry**: Use the `repair_geojson.py` script in the `scripts/` folder if interaction is broken in the interior.
2. **Setup Source**: Use `promoteId` in `addSource`.
3. **Implement Logic**: Use the pattern in [MapHighPerformance.tsx](file:///Users/leonardoleon/Library/Mobile%20Documents/com~apple~CloudDocs/Universidad/UPC/9no%20ciclo/Tesis%201/.agent/skills/map-efficient-hovering/examples/MapHighPerformance.tsx) to handle mouse events.
4. **Style Layer**: Use the `feature-state` expression for the highlight layer.
