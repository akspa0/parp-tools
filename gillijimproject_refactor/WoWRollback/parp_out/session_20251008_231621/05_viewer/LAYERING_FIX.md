# Z-Index Layering Fix

## Problem
Overlays (grid, objects, heatmaps) were not appearing on top of minimap tiles.

## Root Cause
Leaflet's default pane system has these z-index values:
- `tilePane`: 200
- `overlayPane`: 400 (default for SVG overlays and image overlays)
- `shadowPane`: 500
- `markerPane`: 600
- `tooltipPane`: 650
- `popupPane`: 700

Both minimap tiles AND plugin overlays were being added to the same `overlayPane` (z-index 400), causing them to stack in order of addition rather than by intended layer hierarchy.

## Solution
Created a custom pane called `tilesPane` with z-index 100 specifically for minimap background tiles.

### Implementation

**1. Create custom pane on map init:**
```javascript
// Create custom pane for minimap tiles (lower z-index so overlays appear on top)
map.createPane('tilesPane');
map.getPane('tilesPane').style.zIndex = 100; // Below overlays (400+)
```

**2. Assign tiles to custom pane:**
```javascript
// For SVG overlays (simulateTiles)
const overlay = L.svgOverlay(svg, bounds, {
    opacity: 1.0,
    interactive: true,
    pane: 'tilesPane' // Use custom pane with z-index 100
});

// For image overlays (loadRealTiles)
const overlay = L.imageOverlay(url, bounds, {
    opacity: 1.0,
    interactive: false,
    errorOverlayUrl: null,
    pane: 'tilesPane' // Use custom pane with z-index 100
});
```

## Result
Correct layer stacking (bottom to top):
1. **Z-index 100**: Minimap tiles (tilesPane)
2. **Z-index 400**: Grid overlay (GridPlugin)
3. **Z-index 420**: Density heatmap (DensityHeatmapPlugin)
4. **Z-index 450**: Chunk grid (ChunkGridPlugin)
5. **Z-index 600**: M2 objects (M2Plugin)
6. **Z-index 600**: WMO objects (WMOPlugin)

## Usage
1. Open `test-plugin-system.html`
2. Click **"Simulate Tile Images"** to load test tiles
3. Enable any overlay plugins (Grid, Chunk Grid, Density, M2, WMO)
4. Overlays now correctly appear on top of tiles

## Notes
- Tiles must be loaded manually (click button)
- Grid is enabled by default for testing
- All overlays use viewport-based lazy loading
- Tiles are in dedicated pane for proper z-ordering

---

**Fixed**: 2025-10-09 00:17
