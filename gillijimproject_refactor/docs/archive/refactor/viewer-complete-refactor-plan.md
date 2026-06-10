# Viewer Complete Refactor Plan

**Date**: 2025-01-08 20:44  
**Status**: PLANNING - Hard refactor required

---

## 🎯 Core Problems

### 1. Coordinate System Chaos
- **Minimap PNGs**: Positioned correctly ✅
- **Everything else**: Wrong coordinate transforms ❌
- **Grid overlays**: Masking tiles, wrong orientation
- **Click coordinates**: Only show tile, not chunk
- **No standardization**: Each system uses different transforms

### 2. Performance Issues
- **Millions of data points** to plot
- **No proximity filtering**: Loads everything
- **No zoom-level optimization**: Same detail at all zooms
- **Monolithic overlays**: Can't control individual layers

### 3. Missing Core Features
- **No placement coordinates**: All objects at (0,0,0)
- **No flags tracking**: Can't differentiate object types
- **No overlay generation**: OverlayGenerator broken
- **No modular layers**: Everything mixed together

---

## 📐 Step 1: Fix Coordinate System Foundation

### Define Standard Coordinate Spaces

#### A. **World Coordinates** (WoW game coords)
```
Origin: Map center (0, 0)
Bounds: ±17066.66656 yards
X+: East
Y+: South (CRITICAL!)
```

#### B. **Tile Coordinates** (ADT grid)
```
Range: 0-63 (64x64 grid)
Tile (0,0): Northwest corner
Tile (63,63): Southeast corner
Size: 533.33333 yards per tile
```

#### C. **Chunk Coordinates** (MCNK grid)
```
Range: 0-15 per tile (16x16 chunks)
Total: 1024x1024 chunks across map
Size: 33.33333 yards per chunk
```

#### D. **Pixel Coordinates** (Minimap image)
```
Range: 0-511 (512x512 pixels per tile)
Origin: Top-left of tile image
```

#### E. **Leaflet Coordinates** (Map display)
```
CRS: L.CRS.Simple
Bounds: [[0, 0], [64, 64]]
lat: row (with Y-flip if coordMode: "wowtools")
lng: col
```

### Conversion Functions (CANONICAL)
```javascript
// World → Tile
function worldToTile(worldX, worldY) {
    const row = Math.floor(32 - (worldY / 533.33333));
    const col = Math.floor(32 - (worldX / 533.33333));
    return { row, col };
}

// Tile → World (center of tile)
function tileToWorld(row, col) {
    const worldX = (32 - col - 0.5) * 533.33333;
    const worldY = (32 - row - 0.5) * 533.33333;
    return { worldX, worldY };
}

// Tile + Pixel → World
function tilePixelToWorld(row, col, px, py) {
    const tileWorld = tileToWorld(row, col);
    const offsetX = (px / 512 - 0.5) * 533.33333;
    const offsetY = (py / 512 - 0.5) * 533.33333;
    return {
        worldX: tileWorld.worldX + offsetX,
        worldY: tileWorld.worldY + offsetY
    };
}

// World → Chunk
function worldToChunk(worldX, worldY) {
    const chunkX = Math.floor((32 * 16) - (worldX / 33.33333));
    const chunkY = Math.floor((32 * 16) - (worldY / 33.33333));
    return { chunkX, chunkY };
}

// Leaflet lat/lng → Tile row/col
function latLngToTile(lat, lng) {
    const row = isWowTools() ? (63 - lat) : lat;
    const col = lng;
    return { row: Math.floor(row), col: Math.floor(col) };
}
```

---

## 🏗️ Step 2: Modular Layer Architecture

### Layer Manager System
```javascript
class LayerController {
    constructor(map) {
        this.map = map;
        this.layers = new Map();
        this.coordSystem = new CoordinateSystem(); // Centralized transforms
    }
    
    registerLayer(id, manager) {
        this.layers.set(id, manager);
    }
    
    async loadVisibleData(bounds, zoom) {
        // Only load data within viewport + buffer
        // Adjust detail level based on zoom
        const visibleLayers = Array.from(this.layers.values())
            .filter(layer => layer.visible);
        
        await Promise.all(visibleLayers.map(layer => 
            layer.loadForBounds(bounds, zoom)
        ));
    }
}
```

### Proximity-Based Loading
```javascript
class BaseLayerManager {
    async loadForBounds(bounds, zoom) {
        // Calculate which tiles are visible
        const tiles = this.getTilesInBounds(bounds);
        
        // Determine detail level from zoom
        const detailLevel = this.getDetailLevel(zoom);
        
        // Only load if not already cached
        const toLoad = tiles.filter(t => !this.cache.has(t.key));
        
        if (toLoad.length > 0) {
            const data = await this.fetchData(toLoad, detailLevel);
            this.renderData(data);
        }
    }
    
    getDetailLevel(zoom) {
        if (zoom < 3) return 'low';    // Show only major objects
        if (zoom < 6) return 'medium'; // Show most objects
        return 'high';                  // Show everything
    }
}
```

---

## 🎨 Step 3: Visual Differentiation System

### Flag-Based Styling
```javascript
const FLAG_STYLES = {
    // M2 Flags
    0x0001: { color: '#FF0000', shape: 'circle', label: 'Hidden' },
    0x0002: { color: '#00FF00', shape: 'circle', label: 'Particle Emitter' },
    0x0004: { color: '#0000FF', shape: 'circle', label: 'Collision Only' },
    
    // WMO Flags
    0x0001: { color: '#FFA500', shape: 'square', label: 'Destroyable' },
    0x0002: { color: '#800080', shape: 'square', label: 'Use LOD' },
    
    // Default
    DEFAULT_M2: { color: '#2196F3', shape: 'circle', label: 'M2 Doodad' },
    DEFAULT_WMO: { color: '#FF9800', shape: 'square', label: 'WMO Object' }
};

function getMarkerStyle(placement) {
    const isM2 = placement.modelPath.endsWith('.m2');
    const flags = placement.flags || 0;
    
    // Check for specific flag matches
    for (const [flag, style] of Object.entries(FLAG_STYLES)) {
        if (flags & parseInt(flag)) {
            return style;
        }
    }
    
    // Default style
    return isM2 ? FLAG_STYLES.DEFAULT_M2 : FLAG_STYLES.DEFAULT_WMO;
}
```

---

## 🔧 Step 4: Fix Data Pipeline

### Extract Placement Coordinates
**Problem**: AlphaWDTAnalyzer doesn't read MODF/MDDF coordinates

**Fix**: Update `AdtScanner.cs`
```csharp
public class PlacementRecord
{
    public uint UniqueId { get; set; }
    public string ModelPath { get; set; }
    public Vector3 Position { get; set; }  // ADD: From MODF/MDDF
    public Vector3 Rotation { get; set; }  // ADD: From MODF/MDDF
    public ushort Scale { get; set; }      // ADD: From MODF/MDDF
    public ushort Flags { get; set; }      // ADD: From MODF/MDDF
    public int TileRow { get; set; }
    public int TileCol { get; set; }
}
```

### Generate Proper Overlays
**Problem**: OverlayGenerator not creating JSON files

**Debug**:
1. Check `[OverlayGen]` console output
2. Verify AnalysisIndex.Placements not null
3. Ensure coordinate transforms correct
4. Validate JSON structure

---

## 📊 Step 5: Grid System (Done Right)

### ADT Grid Only (For Now)
```javascript
function addAdtGrid() {
    // Simple rectangle borders for each tile
    const tiles = state.getTilesForMap(state.selectedMap);
    
    tiles.forEach(tile => {
        const bounds = tileBounds(tile.row, tile.col);
        const rect = L.rectangle(bounds, {
            color: '#555',
            weight: 1,
            fill: false,
            interactive: false
        });
        rect.addTo(map);
    });
}
```

### Chunk Grid (Zoom-Dependent)
```javascript
map.on('zoomend', () => {
    if (map.getZoom() >= 6) {
        showChunkGrid();
    } else {
        hideChunkGrid();
    }
});
```

---

## 🚀 Implementation Order

### Phase 1: Foundation (Week 1)
1. **Coordinate System**
   - Create `CoordinateSystem.js` with canonical transforms
   - Test all conversions
   - Document with examples

2. **Fix Data Pipeline**
   - Extract coordinates from MODF/MDDF
   - Add flags to PlacementRecord
   - Update CSV exports

3. **Fix Overlay Generation**
   - Debug OverlayGenerator
   - Ensure JSON files created
   - Validate structure

### Phase 2: Modular Layers (Week 2)
1. **LayerController**
   - Base architecture
   - Proximity loading
   - Zoom-level optimization

2. **Core Layers**
   - M2LayerManager
   - WmoLayerManager
   - Flag-based styling

3. **UI Controls**
   - Layer toggles
   - Legend system
   - Preset views

### Phase 3: Advanced Features (Week 3)
1. **Terrain Layers**
   - TerrainLayerManager
   - LiquidsLayerManager
   - AreaBoundariesLayerManager

2. **Grid System**
   - ADT grid (always visible)
   - Chunk grid (zoom >= 6)
   - Proper coordinate alignment

3. **Performance**
   - Caching strategy
   - Lazy loading
   - Memory management

---

## ✅ Success Criteria

### Coordinate System
- [ ] Click anywhere → Get correct world coords
- [ ] Click anywhere → Get correct tile coords
- [ ] Click anywhere → Get correct chunk coords
- [ ] All transforms documented and tested

### Performance
- [ ] Handles millions of data points
- [ ] Only loads visible data
- [ ] Zoom-level optimization works
- [ ] No lag when panning/zooming

### Visual System
- [ ] Flag-based colors work
- [ ] Shape differentiation (M2 circle, WMO square)
- [ ] Legend shows all styles
- [ ] Easy to understand at a glance

### Modular Layers
- [ ] Toggle layers individually
- [ ] Each layer loads independently
- [ ] Preset views work
- [ ] State persists (localStorage)

---

## 📝 Next Steps

1. **STOP** fighting with the current broken system
2. **CREATE** `CoordinateSystem.js` with canonical transforms
3. **FIX** data pipeline (coordinates + flags)
4. **BUILD** LayerController architecture
5. **TEST** each piece before moving on

**No more band-aids. Do it right this time.** 🎯
