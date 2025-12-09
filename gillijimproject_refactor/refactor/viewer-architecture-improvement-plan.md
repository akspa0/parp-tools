# Viewer Architecture Improvement Plan (2025-10-08)

## Executive Summary

**Current State**: Proof-of-concept viewer with performance issues and architectural problems
**Goal**: Streamlined, versatile viewer with proper overlay managers and clean separation of concerns
**Blockers**: Must fix OverlayGenerator compilation first (see overlay-generator-fix-plan.md)

---

## Critical Issues Discovered

### 1. Path & Filename Mismatches
**Backend (OverlayGenerator.cs writes)**:
```
viewer/overlays/<version>/<map>/objects_combined/tile_{tileX}_{tileY}.json
```

**Frontend (Viewer expects)**:
```
overlays/<version>/<map>/<variant>/tile_r{row}_c{col}.json
```

**Problems**:
- Filename format: `tile_{X}_{Y}.json` vs `tile_r{row}_c{col}.json`
- Directory structure: hardcoded `objects_combined` vs dynamic `<variant>`
- No r/c prefix in backend output

### 2. Data Format Mismatches
**Backend writes (new format)**:
```json
{
  "tileX": 30,
  "tileY": 30,
  "placements": [
    {
      "kind": "Wmo",
      "uniqueId": 1234,
      "assetPath": "path/to/asset.wmo",
      "world": [x, y, z],
      "rotation": [rx, ry, rz],
      "scale": 1.0,
      "flags": 0,
      "doodadSet": 0,
      "nameSet": 0
    }
  ]
}
```

**Frontend expects (legacy format)**:
```json
{
  "layers": [
    {
      "version": "0.5.3.3368",
      "kinds": [
        {
          "kind": "wmo",
          "points": [
            {
              "uniqueId": 1234,
              "fileName": "asset.wmo",
              "pixel": { "x": 256, "y": 256 },
              "world": { "x": ..., "y": ..., "z": ... }
            }
          ]
        }
      ]
    }
  ]
}
```

**Incompatibilities**:
- Different structure: flat `placements[]` vs nested `layers[].kinds[].points[]`
- Missing `pixel` coordinates (required for map rendering)
- Missing `fileName` extraction from path
- Array format: `world: [x,y,z]` vs `world: {x, y, z}`

### 3. Monolithic main.js (1033 lines)
**Problems**:
- `performObjectMarkerUpdate()`: 150 lines, mixed concerns
- Data loading, coordinate transforms, marker creation, popup HTML all intertwined
- No separation between data layer and presentation layer
- Direct Leaflet API calls throughout
- Hard to test, hard to extend

### 4. Missing Object Overlay Manager
**Current**:
- OverlayManager exists for terrain overlays (terrain_complete/)
- Objects loaded directly in main.js with manual debouncing and caching
- Duplicate logic for viewport calculations, tile loading, cleanup

**Needed**:
- ObjectOverlayManager parallel to OverlayManager
- Handles loading object overlays from objects_combined/
- Manages object markers lifecycle (create, update, remove)
- Proper caching and cleanup of markers outside viewport

---

## Architecture Problems

### Current Architecture (Problematic)
```
main.js (1033 lines)
├── Direct Leaflet API calls
├── Manual overlay loading
├── Inline marker creation
├── Mixed coordinate transforms
└── Embedded popup HTML

state.js
├── Path generation (getOverlayPath)
└── Basic state management

overlayLoader.js
└── Simple fetch + cache

overlays/overlayManager.js (Terrain only)
└── Terrain overlay rendering
```

### Target Architecture (Clean)
```
main.js (< 400 lines)
├── Initialization
├── UI event handlers
├── Manager coordination
└── High-level state management

managers/
├── ObjectOverlayManager.js
│   ├── Load object overlays
│   ├── Create/update/remove markers
│   ├── Viewport-based loading
│   └── Marker lifecycle management
│
├── TerrainOverlayManager.js (existing, refactored)
│   └── Terrain overlay rendering
│
└── CoordinateManager.js
    ├── Pixel ↔ LatLng transforms
    ├── WoW coords ↔ Leaflet coords
    └── Tile coordinate calculations

components/
├── ObjectMarker.js
│   ├── Marker creation logic
│   └── Popup generation
│
└── MinimapTile.js
    └── Minimap image overlay logic

utils/
├── overlayAdapter.js
│   └── Convert new JSON format → legacy format
│
└── pathResolver.js
    └── Resolve paths between backend and frontend
```

---

## Implementation Plan

### Phase 1: Fix Backend-Frontend Alignment (High Priority)

#### Step 1.1: Standardize Filenames
**Backend Change** (OverlayGenerator.cs):
```csharp
// Current:
var jsonPath = Path.Combine(objectsDir, $"tile_{tile.TileX}_{tile.TileY}.json");

// Change to:
var jsonPath = Path.Combine(objectsDir, $"tile_r{tile.TileX}_c{tile.TileY}.json");
```

**Why**: Match viewer's expected filename format

#### Step 1.2: Add Data Adapter Layer
**New File**: `assets/js/utils/overlayAdapter.js`
```javascript
/**
 * Adapts new backend JSON format to legacy viewer format
 */
export class OverlayAdapter {
    /**
     * Convert new TileOverlayJson to legacy format
     */
    static adaptObjectOverlay(newFormat, tileX, tileY, version) {
        // Group placements by kind
        const kindGroups = {};
        for (const placement of newFormat.placements || []) {
            const kind = placement.kind?.toLowerCase() || 'unknown';
            if (!kindGroups[kind]) {
                kindGroups[kind] = [];
            }
            
            // Convert placement to legacy point format
            const point = {
                uniqueId: placement.uniqueId,
                fileName: extractFileName(placement.assetPath),
                assetPath: placement.assetPath,
                world: {
                    x: placement.world[0],
                    y: placement.world[1],
                    z: placement.world[2]
                },
                pixel: worldToPixel(placement.world, tileX, tileY),
                rotation: {
                    x: placement.rotation[0],
                    y: placement.rotation[1],
                    z: placement.rotation[2]
                },
                scale: placement.scale,
                flags: placement.flags,
                doodadSet: placement.doodadSet,
                nameSet: placement.nameSet
            };
            
            kindGroups[kind].push(point);
        }
        
        // Convert to legacy layers format
        const kinds = Object.entries(kindGroups).map(([kind, points]) => ({
            kind: kind,
            label: kind,
            points: points
        }));
        
        return {
            layers: [{
                version: version,
                kinds: kinds
            }],
            minimap: { width: 512, height: 512 }
        };
    }
    
    static extractFileName(path) {
        if (!path) return 'Unknown';
        const parts = path.split(/[/\\]/);
        return parts[parts.length - 1] || 'Unknown';
    }
    
    static worldToPixel(world, tileX, tileY) {
        // TODO: Implement proper WoW world → pixel coordinate transform
        // For now, use center of tile as fallback
        return { x: 256, y: 256 };
    }
}
```

#### Step 1.3: Update Path Resolution
**Modify**: `assets/js/state.js`
```javascript
getOverlayPath(mapName, row, col, version, variant) {
    const safeVersion = version || this.selectedVersion;
    const safeVariant = variant || this.overlayVariant || 'combined';
    
    // New backend uses objects_combined for all object types
    const backendDir = safeVariant === 'combined' || safeVariant === 'm2' || safeVariant === 'wmo'
        ? 'objects_combined'
        : safeVariant;
    
    return `overlays/${safeVersion}/${mapName}/${backendDir}/tile_r${row}_c${col}.json`;
}
```

### Phase 2: Create ObjectOverlayManager

**New File**: `assets/js/overlays/objectOverlayManager.js`
```javascript
import { loadOverlay } from '../overlayLoader.js';
import { OverlayAdapter } from '../utils/overlayAdapter.js';

export class ObjectOverlayManager {
    constructor(map, state) {
        this.map = map;
        this.state = state;
        this.markerLayer = L.layerGroup();
        this.loadedTiles = new Map(); // key: "r{row}_c{col}" -> markers[]
        this.loadTimer = null;
        this.loadDelay = 500; // ms debounce
    }
    
    show() {
        this.markerLayer.addTo(this.map);
    }
    
    hide() {
        this.markerLayer.remove();
    }
    
    clear() {
        this.markerLayer.clearLayers();
        this.loadedTiles.clear();
    }
    
    // Load objects for visible tiles (debounced)
    loadVisibleObjects(mapName, version, variant = 'combined') {
        clearTimeout(this.loadTimer);
        this.loadTimer = setTimeout(() => {
            this._loadVisibleObjectsNow(mapName, version, variant);
        }, this.loadDelay);
    }
    
    async _loadVisibleObjectsNow(mapName, version, variant) {
        const visibleTiles = this.getVisibleTiles();
        
        // Load each visible tile
        for (const tile of visibleTiles) {
            await this.loadAndRenderTile(mapName, version, tile.row, tile.col, variant);
        }
        
        // Cleanup distant tiles
        this.cleanupDistantTiles(visibleTiles);
    }
    
    async loadAndRenderTile(mapName, version, tileRow, tileCol, variant) {
        const tileKey = `r${tileRow}_c${tileCol}`;
        
        // Check cache
        if (this.loadedTiles.has(tileKey)) {
            return;
        }
        
        try {
            const overlayPath = this.state.getOverlayPath(mapName, tileRow, tileCol, version, variant);
            const rawData = await loadOverlay(overlayPath);
            
            // Adapt new format to legacy format
            const data = OverlayAdapter.adaptObjectOverlay(rawData, tileRow, tileCol, version);
            
            // Create markers
            const markers = this.createMarkers(data, tileRow, tileCol, variant);
            
            // Cache markers
            this.loadedTiles.set(tileKey, markers);
            
            // Add to map
            markers.forEach(m => this.markerLayer.addLayer(m));
            
        } catch (error) {
            console.warn(`Failed to load objects for tile ${tileKey}:`, error);
        }
    }
    
    createMarkers(data, tileRow, tileCol, variant) {
        const markers = [];
        const versionData = data.layers?.[0];
        if (!versionData || !versionData.kinds) return markers;
        
        const objects = versionData.kinds.flatMap(kind => {
            const label = typeof kind.kind === 'string' ? kind.kind : (kind.label ?? 'unknown');
            return (kind.points || []).map(point => ({
                ...point,
                __kind: label
            }));
        });
        
        objects.forEach(obj => {
            const marker = this.createMarkerForObject(obj, tileRow, tileCol, variant);
            if (marker) markers.push(marker);
        });
        
        return markers;
    }
    
    createMarkerForObject(obj, tileRow, tileCol, variant) {
        if (!obj.pixel || !obj.world) return null;
        
        // Convert pixel to lat/lng
        const { lat, lng } = this.pixelToLatLng(tileRow, tileCol, obj.pixel.x, obj.pixel.y);
        
        const kindTag = (obj.__kind || '').toString().toLowerCase();
        const isWmo = kindTag.includes('wmo');
        
        // Create marker based on type
        if (isWmo) {
            return this.createWmoMarker(lat, lng, obj);
        } else {
            return this.createM2Marker(lat, lng, obj);
        }
    }
    
    createWmoMarker(lat, lng, obj) {
        const squareHalf = this.getScaledSquareSize(0.006);
        const bounds = [
            [lat - squareHalf, lng - squareHalf],
            [lat + squareHalf, lng + squareHalf]
        ];
        
        const square = L.rectangle(bounds, {
            color: '#000',
            weight: 1,
            fillColor: '#FF9800',
            fillOpacity: 0.85
        });
        
        square.bindPopup(this.createPopupHtml(obj), {
            maxWidth: 350,
            closeButton: true,
            autoClose: false,
            closeOnClick: false
        });
        
        return square;
    }
    
    createM2Marker(lat, lng, obj) {
        const circle = L.circleMarker([lat, lng], {
            radius: this.getScaledRadius(4),
            fillColor: '#2196F3',
            color: '#000',
            weight: 1,
            fillOpacity: 0.9
        });
        circle._baseRadius = 4;
        
        circle.bindPopup(this.createPopupHtml(obj), {
            maxWidth: 350,
            closeButton: true,
            autoClose: false,
            closeOnClick: false
        });
        
        return circle;
    }
    
    createPopupHtml(obj) {
        return `
            <div style="min-width: 280px; padding: 6px;">
                <strong style="font-size: 14px;">${obj.fileName || 'Unknown'}</strong><br>
                <div style="margin-top: 8px; font-size: 12px;">
                    <strong>UID:</strong> ${obj.uniqueId || 'N/A'}<br>
                    <strong>World X:</strong> ${obj.world.x?.toFixed(3) || 'N/A'}<br>
                    <strong>World Y:</strong> ${obj.world.y?.toFixed(3) || 'N/A'}<br>
                    <strong>World Z:</strong> ${obj.world.z?.toFixed(2) || 'N/A'}<br>
                    ${obj.assetPath ? `<strong>Path:</strong> ${obj.assetPath}<br>` : ''}
                </div>
            </div>
        `;
    }
    
    getVisibleTiles() {
        const bounds = this.map.getBounds();
        const tiles = [];
        
        const latS = bounds.getSouth();
        const latN = bounds.getNorth();
        const west = bounds.getWest();
        const east = bounds.getEast();
        
        const rowNorth = this.latToRow(latN);
        const rowSouth = this.latToRow(latS);
        const minRow = Math.floor(Math.min(rowNorth, rowSouth));
        const maxRow = Math.ceil(Math.max(rowNorth, rowSouth));
        const minCol = Math.floor(west);
        const maxCol = Math.ceil(east);
        
        for (let r = Math.max(0, minRow); r <= Math.min(63, maxRow); r++) {
            for (let c = Math.max(0, minCol); c <= Math.min(63, maxCol); c++) {
                tiles.push({ row: r, col: c });
            }
        }
        
        return tiles;
    }
    
    cleanupDistantTiles(visibleTiles) {
        const visibleSet = new Set(visibleTiles.map(t => `r${t.row}_c${t.col}`));
        
        for (const [tileKey, markers] of this.loadedTiles.entries()) {
            if (!visibleSet.has(tileKey)) {
                const match = tileKey.match(/r(\d+)_c(\d+)/);
                if (match) {
                    const row = parseInt(match[1]);
                    const col = parseInt(match[2]);
                    
                    const minDist = Math.min(...visibleTiles.map(t => 
                        Math.abs(t.row - row) + Math.abs(t.col - col)
                    ));
                    
                    if (minDist > 2) {
                        // Remove markers from map
                        markers.forEach(m => this.markerLayer.removeLayer(m));
                        this.loadedTiles.delete(tileKey);
                    }
                }
            }
        }
    }
    
    // Coordinate transforms (delegate to CoordinateManager in future)
    latToRow(lat) {
        return 63 - lat;
    }
    
    pixelToLatLng(tileRow, tileCol, pixelX, pixelY, tileWidth = 512, tileHeight = 512) {
        const lat = (63 - tileRow) - (pixelY / tileHeight);
        const lng = tileCol + (pixelX / tileWidth);
        return { lat, lng };
    }
    
    getScaledRadius(baseRadius) {
        const zoom = this.map.getZoom();
        const scale = 0.7 + (zoom / 12) * 1.8;
        return Math.max(2, baseRadius * scale);
    }
    
    getScaledSquareSize(baseSize) {
        const zoom = this.map.getZoom();
        const scale = 0.7 + (zoom / 12) * 1.8;
        return baseSize * scale;
    }
}
```

### Phase 3: Refactor main.js

**Target**: Reduce main.js from 1033 lines to < 400 lines

#### Changes:
1. **Remove** entire `performObjectMarkerUpdate()` function (150 lines)
2. **Replace** with ObjectOverlayManager calls:
```javascript
// Old (150+ lines):
async function performObjectMarkerUpdate() {
    objectMarkers.clearLayers();
    // ... 150 lines of inline logic ...
}

// New (3 lines):
function updateObjectMarkers() {
    if (objectOverlayManager) {
        objectOverlayManager.loadVisibleObjects(
            state.selectedMap,
            state.selectedVersion,
            state.overlayVariant
        );
    }
}
```

3. **Initialize** managers in `init()`:
```javascript
// Initialize overlay managers
overlayManager = new OverlayManager(map);
objectOverlayManager = new ObjectOverlayManager(map, state);
objectOverlayManager.show();
```

4. **Update** event handlers:
```javascript
map.on('moveend', () => {
    overlayManager.loadVisibleOverlays(state.selectedMap, state.selectedVersion);
    objectOverlayManager.loadVisibleObjects(state.selectedMap, state.selectedVersion, state.overlayVariant);
});

map.on('zoomend', () => {
    overlayManager.renderVisibleTiles();
    // ObjectOverlayManager handles zoom scaling internally
});
```

### Phase 4: Add World → Pixel Coordinate Transform

**Critical Missing Piece**: Backend writes world coordinates, but viewer needs pixel coordinates for map rendering.

**New File**: `assets/js/utils/coordinateTransforms.js`
```javascript
/**
 * WoW World Coordinate System:
 * - Origin (0, 0) is at the center of the map
 * - X increases to the east (positive = east)
 * - Y increases to the north (positive = north)
 * - Each tile is 533.33333 yards
 * - Total map is 64x64 tiles = 34133.33 yards
 * 
 * Tile Pixel Coordinates:
 * - Origin (0, 0) is at the NW corner of the tile
 * - X increases to the east
 * - Y increases to the south
 * - Tile resolution is 512x512 pixels
 */
export class CoordinateTransforms {
    static TILE_SIZE_YARDS = 533.33333;
    static MAP_SIZE_TILES = 64;
    static TILE_SIZE_PIXELS = 512;
    
    /**
     * Convert world coordinates to tile + pixel coordinates
     */
    static worldToTilePixel(worldX, worldY, worldZ) {
        // Map center is at (MAP_SIZE_TILES / 2, MAP_SIZE_TILES / 2) in tile coords
        const centerTile = this.MAP_SIZE_TILES / 2;
        const centerYards = centerTile * this.TILE_SIZE_YARDS;
        
        // Convert world yards to tile coordinates
        // WoW X (east) → tile col, WoW Y (north) → tile row (inverted)
        const tileXFloat = centerTile + (worldX / this.TILE_SIZE_YARDS);
        const tileYFloat = centerTile - (worldY / this.TILE_SIZE_YARDS);
        
        const tileX = Math.floor(tileXFloat);
        const tileY = Math.floor(tileYFloat);
        
        // Get fractional part and convert to pixels
        const tileFracX = tileXFloat - tileX;
        const tileFracY = tileYFloat - tileY;
        
        const pixelX = tileFracX * this.TILE_SIZE_PIXELS;
        const pixelY = tileFracY * this.TILE_SIZE_PIXELS;
        
        return {
            tileX: tileX,
            tileY: tileY,
            pixelX: pixelX,
            pixelY: pixelY,
            worldZ: worldZ
        };
    }
    
    /**
     * Convert tile + pixel coordinates to Leaflet lat/lng
     */
    static tilePixelToLatLng(tileX, tileY, pixelX, pixelY) {
        // Leaflet lat = inverted tile Y
        const lat = (63 - tileY) - (pixelY / this.TILE_SIZE_PIXELS);
        const lng = tileX + (pixelX / this.TILE_SIZE_PIXELS);
        return { lat, lng };
    }
    
    /**
     * One-shot: world coords → Leaflet lat/lng
     */
    static worldToLatLng(worldX, worldY, worldZ) {
        const { tileX, tileY, pixelX, pixelY } = this.worldToTilePixel(worldX, worldY, worldZ);
        return this.tilePixelToLatLng(tileX, tileY, pixelX, pixelY);
    }
}
```

**Update OverlayAdapter to use transforms**:
```javascript
static worldToPixel(world, tileX, tileY) {
    const { tileX: calcTileX, tileY: calcTileY, pixelX, pixelY } = 
        CoordinateTransforms.worldToTilePixel(world[0], world[1], world[2]);
    
    // Verify tile matches
    if (calcTileX !== tileX || calcTileY !== tileY) {
        console.warn(`World coords don't match tile: calc=${calcTileX},${calcTileY} vs expected=${tileX},${tileY}`);
    }
    
    return { x: pixelX, y: pixelY };
}
```

---

## Testing Strategy

### Phase 1 Tests
1. **Filename test**: Verify backend writes `tile_r30_c30.json` not `tile_30_30.json`
2. **Path resolution test**: Check `state.getOverlayPath()` returns correct path
3. **Adapter test**: Verify JSON conversion with sample data

### Phase 2 Tests
1. **Manager initialization**: Verify ObjectOverlayManager creates without errors
2. **Marker creation**: Test WMO square and M2 circle creation
3. **Caching**: Verify tiles are cached and reused
4. **Cleanup**: Verify distant tiles are removed

### Phase 3 Tests
1. **Integration**: Verify main.js initializes all managers
2. **Event handling**: Verify pan/zoom triggers correct manager methods
3. **Line count**: Verify main.js reduced to < 400 lines

### Phase 4 Tests
1. **Coordinate transform**: Verify world → pixel → lat/lng round-trip
2. **Visual alignment**: Verify objects appear at correct map positions
3. **Edge cases**: Test tiles at map edges (0,0), (63,63)

---

## Success Criteria

- [ ] Build succeeds with zero errors
- [ ] Backend and frontend use matching filenames (`tile_r{X}_c{Y}.json`)
- [ ] OverlayAdapter correctly converts new → legacy JSON format
- [ ] ObjectOverlayManager loads and renders objects
- [ ] main.js reduced from 1033 to < 400 lines
- [ ] Objects appear at correct map positions
- [ ] Performance: smooth panning with 100+ objects visible
- [ ] Memory: proper cleanup of off-screen objects
- [ ] No console errors during normal operation

---

## Estimated Timeline

- **Phase 1** (Backend-Frontend Alignment): 2 hours
- **Phase 2** (ObjectOverlayManager): 4 hours
- **Phase 3** (Refactor main.js): 2 hours
- **Phase 4** (Coordinate Transforms): 3 hours
- **Testing & Integration**: 3 hours

**Total**: ~14 hours (2 days)

---

## Dependencies

1. **Must complete first**: overlay-generator-fix-plan.md (30 minutes)
2. **Parallel work possible**: Can start Phase 4 (coordinate transforms) independently
3. **Sequential**: Phases 1 → 2 → 3 must be done in order

---

## Future Improvements (Post-MVP)

1. **CoordinateManager**: Extract all coordinate logic into dedicated manager
2. **MarkerFactory**: Separate marker creation from ObjectOverlayManager
3. **PopupGenerator**: Template-based popup HTML generation
4. **FilterManager**: Centralize UID range filtering, sedimentary layers
5. **PerformanceMonitor**: Track FPS, memory usage, loading times
6. **WebWorker**: Offload JSON parsing and coordinate transforms
7. **Progressive Loading**: Load high-priority tiles first, then fill in gaps
8. **Virtual Scrolling**: Only render markers in viewport + 1 tile buffer
