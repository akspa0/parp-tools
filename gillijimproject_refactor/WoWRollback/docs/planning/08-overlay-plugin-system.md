# Overlay Plugin System - Modular Architecture

**Date**: 2025-01-08  
**Status**: 🎯 ACTIVE - New viewer foundation  
**Goal**: Build a plugin-based overlay system that's maintainable and extensible

---

## 📋 Quick Start Checklist

**Before you start:**
- [ ] Read this entire document
- [ ] Understand the coordinate system (most critical!)
- [ ] Review existing WdlGltfExporter and ADTPreFabTool
- [ ] Check current viewer state (committed as checkpoint)

**Implementation order:**
1. **Week 1**: CoordinateSystem.js + OverlayPlugin base + PluginManager
2. **Week 2**: GridPlugin + M2Plugin + WMOPlugin + UI controls
3. **Week 3**: Data pipeline (extract 3D coords + flags from MODF/MDDF)
4. **Week 4**: 2.5D viewer (optional, but recommended)

**Critical files to create:**
- `js/core/CoordinateSystem.js` - ALL coordinate transforms
- `js/core/OverlayPlugin.js` - Base plugin class
- `js/core/PluginManager.js` - Plugin lifecycle
- `js/plugins/GridPlugin.js` - First plugin (simple, no data)
- `js/plugins/M2Plugin.js` - M2 doodads with proximity loading
- `js/plugins/WMOPlugin.js` - WMO objects
- `js/main.js` - Entry point (rewrite from scratch)

---

## 🎯 Core Principles

### 1. **Coordinate System First**
- ONE canonical coordinate system
- ALL overlays use the same transforms
- NO more coordinate chaos

### 2. **Plugin Architecture**
- Each overlay is a self-contained plugin
- Register/unregister dynamically
- Independent lifecycle management

### 3. **Performance by Design**
- Proximity-based loading
- Zoom-level optimization
- Lazy rendering

### 4. **Asset Management**
- Clear file structure
- Module-based organization
- No more "flapping in the breeze"

---

## 📐 Coordinate System Foundation

### Data Model: 3D Positions

**All placement data includes 3D coordinates:**
```javascript
{
    uniqueId: 12345,
    modelPath: "World\\Azeroth\\Elwynn\\PassiveObjects\\Trees\\ElwynnTree01.m2",
    worldX: 1234.56,    // X coordinate (yards)
    worldY: -789.12,    // Y coordinate (yards)
    worldZ: 42.34,      // Z coordinate (elevation, yards)
    tileRow: 30,
    tileCol: 35,
    flags: 0x0000
}
```

**2D Viewer Strategy:**
- Store full 3D coordinates (X, Y, Z)
- Display using X/Y projection (top-down view)
- Use Z for visual cues:
  - Color intensity (higher = brighter)
  - Marker size (higher = larger)
  - Tooltip info (show elevation)
  - Future: Height-based filtering

**3D Viewer Future:**
- WDL terrain meshes (GLB format)
- Minimap textures baked into terrain
- Plot 3D positions on terrain surface
- Camera controls (orbit, pan, zoom)

### File: `js/core/CoordinateSystem.js`

```javascript
/**
 * Canonical coordinate system for WoW maps
 * Handles 3D world coordinates with 2D projection
 * ALL overlays MUST use these transforms
 */
export class CoordinateSystem {
    constructor(config) {
        this.coordMode = config.coordMode || 'wowtools';
        
        // Constants
        this.TILE_SIZE = 533.33333;        // yards
        this.MAP_HALF_SIZE = 17066.66656;  // yards
        this.CHUNK_SIZE = 33.33333;        // yards
        this.CHUNKS_PER_TILE = 16;
        this.TILE_PIXEL_SIZE = 512;
        
        // Z-coordinate ranges (for visualization)
        this.MIN_ELEVATION = -500;  // Typical min (deep water)
        this.MAX_ELEVATION = 2000;  // Typical max (mountains)
    }
    
    // World → Tile
    worldToTile(worldX, worldY) {
        const row = Math.floor(32 - (worldY / this.TILE_SIZE));
        const col = Math.floor(32 - (worldX / this.TILE_SIZE));
        return { row, col };
    }
    
    // Tile → World (center)
    tileToWorld(row, col) {
        const worldX = (32 - col - 0.5) * this.TILE_SIZE;
        const worldY = (32 - row - 0.5) * this.TILE_SIZE;
        return { worldX, worldY };
    }
    
    // World → Chunk
    worldToChunk(worldX, worldY) {
        const chunkX = Math.floor((32 * 16) - (worldX / this.CHUNK_SIZE));
        const chunkY = Math.floor((32 * 16) - (worldY / this.CHUNK_SIZE));
        return { chunkX, chunkY };
    }
    
    // Tile + Pixel → World
    tilePixelToWorld(row, col, px, py) {
        const tileWorld = this.tileToWorld(row, col);
        const offsetX = (px / this.TILE_PIXEL_SIZE - 0.5) * this.TILE_SIZE;
        const offsetY = (py / this.TILE_PIXEL_SIZE - 0.5) * this.TILE_SIZE;
        return {
            worldX: tileWorld.worldX + offsetX,
            worldY: tileWorld.worldY + offsetY
        };
    }
    
    // Leaflet lat/lng → Tile
    latLngToTile(lat, lng) {
        const row = this.coordMode === 'wowtools' ? (63 - lat) : lat;
        const col = lng;
        return { row: Math.floor(row), col: Math.floor(col) };
    }
    
    // Tile → Leaflet lat/lng
    tileToLatLng(row, col) {
        const lat = this.coordMode === 'wowtools' ? (63 - row) : row;
        const lng = col;
        return { lat, lng };
    }
    
    // Leaflet bounds for tile
    tileBounds(row, col) {
        const topLeft = this.tileToLatLng(row, col);
        const bottomRight = this.tileToLatLng(row + 1, col + 1);
        return [
            [Math.min(topLeft.lat, bottomRight.lat), topLeft.lng],
            [Math.max(topLeft.lat, bottomRight.lat), bottomRight.lng]
        ];
    }
    
    // Elevation visualization helpers
    normalizeElevation(z) {
        // Normalize Z to 0-1 range for visualization
        return (z - this.MIN_ELEVATION) / (this.MAX_ELEVATION - this.MIN_ELEVATION);
    }
    
    getElevationColor(z, baseColor) {
        // Adjust color brightness based on elevation
        const normalized = this.normalizeElevation(z);
        const brightness = 0.5 + (normalized * 0.5); // 50-100% brightness
        return this.adjustColorBrightness(baseColor, brightness);
    }
    
    getElevationRadius(z, baseRadius) {
        // Adjust marker size based on elevation
        const normalized = this.normalizeElevation(z);
        return baseRadius * (0.7 + normalized * 0.6); // 70-130% of base
    }
    
    adjustColorBrightness(hex, brightness) {
        const rgb = parseInt(hex.slice(1), 16);
        const r = Math.floor(((rgb >> 16) & 0xFF) * brightness);
        const g = Math.floor(((rgb >> 8) & 0xFF) * brightness);
        const b = Math.floor((rgb & 0xFF) * brightness);
        return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
    }
}
```

---

## 🔌 Plugin Base Class

### File: `js/core/OverlayPlugin.js`

```javascript
/**
 * Base class for all overlay plugins
 * Extend this to create new overlays
 */
export class OverlayPlugin {
    constructor(id, name, map, coordSystem) {
        this.id = id;
        this.name = name;
        this.map = map;
        this.coords = coordSystem;
        
        this.enabled = false;
        this.visible = false;
        this.opacity = 1.0;
        this.zIndex = 400;
        
        this.cache = new Map();
        this.layers = [];
    }
    
    // Lifecycle hooks (override these)
    async onLoad(version, mapName) {
        // Load plugin data
    }
    
    onEnable() {
        // Plugin enabled
        this.enabled = true;
    }
    
    onDisable() {
        // Plugin disabled
        this.enabled = false;
        this.clearLayers();
    }
    
    onShow() {
        // Show layers
        this.visible = true;
        this.layers.forEach(layer => layer.addTo(this.map));
    }
    
    onHide() {
        // Hide layers
        this.visible = false;
        this.layers.forEach(layer => layer.remove());
    }
    
    onViewportChange(bounds, zoom) {
        // Viewport changed - load visible data
        if (!this.enabled || !this.visible) return;
        this.loadVisibleData(bounds, zoom);
    }
    
    onDestroy() {
        // Cleanup
        this.clearLayers();
        this.cache.clear();
    }
    
    // Helper methods
    clearLayers() {
        this.layers.forEach(layer => layer.remove());
        this.layers = [];
    }
    
    setOpacity(value) {
        this.opacity = value;
        this.layers.forEach(layer => {
            if (layer.setOpacity) layer.setOpacity(value);
            if (layer.setStyle) layer.setStyle({ opacity: value, fillOpacity: value });
        });
    }
    
    setZIndex(value) {
        this.zIndex = value;
        this.layers.forEach(layer => {
            if (layer.setZIndex) layer.setZIndex(value);
        });
    }
    
    // Must implement
    async loadVisibleData(bounds, zoom) {
        throw new Error('Plugin must implement loadVisibleData()');
    }
    
    getConfig() {
        return {
            enabled: this.enabled,
            visible: this.visible,
            opacity: this.opacity,
            zIndex: this.zIndex
        };
    }
    
    setConfig(config) {
        this.opacity = config.opacity ?? this.opacity;
        this.zIndex = config.zIndex ?? this.zIndex;
        if (config.enabled !== undefined) {
            config.enabled ? this.onEnable() : this.onDisable();
        }
        if (config.visible !== undefined) {
            config.visible ? this.onShow() : this.onHide();
        }
    }
}
```

---

## 🎨 Example Plugin: M2 Doodads

### File: `js/plugins/M2Plugin.js`

```javascript
import { OverlayPlugin } from '../core/OverlayPlugin.js';

export class M2Plugin extends OverlayPlugin {
    constructor(map, coordSystem) {
        super('m2', 'M2 Doodads', map, coordSystem);
        
        this.color = '#2196F3';
        this.radius = 5;
        this.filterFlags = null;
    }
    
    async onLoad(version, mapName) {
        // Load M2 placement data
        const response = await fetch(`data/${version}/${mapName}/m2_placements.json`);
        this.data = await response.json();
    }
    
    async loadVisibleData(bounds, zoom) {
        // Get visible tiles
        const tiles = this.getVisibleTiles(bounds);
        
        // Filter by zoom level
        const detailLevel = this.getDetailLevel(zoom);
        
        // Load and render
        tiles.forEach(tile => {
            const key = `${tile.row}_${tile.col}`;
            if (!this.cache.has(key)) {
                this.renderTile(tile, detailLevel);
                this.cache.set(key, true);
            }
        });
    }
    
    renderTile(tile, detailLevel) {
        const placements = this.data.filter(p => 
            p.tileRow === tile.row && p.tileCol === tile.col
        );
        
        placements.forEach(p => {
            // Skip low-priority objects at low zoom
            if (detailLevel === 'low' && !this.isImportant(p)) return;
            
            const { lat, lng } = this.coords.tilePixelToLatLng(
                p.tileRow, p.tileCol, p.pixelX, p.pixelY
            );
            
            // Use elevation (Z) for visual cues
            const baseColor = this.getColor(p);
            const elevationColor = this.coords.getElevationColor(p.worldZ, baseColor);
            const elevationRadius = this.coords.getElevationRadius(p.worldZ, this.radius);
            
            const marker = L.circleMarker([lat, lng], {
                radius: elevationRadius,
                color: elevationColor,
                fillColor: elevationColor,
                fillOpacity: 0.6,
                weight: 1
            });
            
            marker.bindPopup(this.createPopup(p));
            marker.addTo(this.map);
            this.layers.push(marker);
        });
    }
    
    getVisibleTiles(bounds) {
        const sw = this.coords.latLngToTile(bounds.getSouth(), bounds.getWest());
        const ne = this.coords.latLngToTile(bounds.getNorth(), bounds.getEast());
        
        const tiles = [];
        for (let row = sw.row; row <= ne.row; row++) {
            for (let col = sw.col; col <= ne.col; col++) {
                tiles.push({ row, col });
            }
        }
        return tiles;
    }
    
    getDetailLevel(zoom) {
        if (zoom < 3) return 'low';
        if (zoom < 6) return 'medium';
        return 'high';
    }
    
    isImportant(placement) {
        // Flag-based importance (buildings, major objects)
        return (placement.flags & 0x0100) !== 0;
    }
    
    getColor(placement) {
        // Flag-based coloring
        if (placement.flags & 0x0001) return '#FF0000'; // Hidden
        if (placement.flags & 0x0002) return '#00FF00'; // Particle
        return this.color;
    }
    
    createPopup(placement) {
        return `
            <strong>${placement.modelPath}</strong><br>
            UniqueID: ${placement.uniqueId}<br>
            Position: (${placement.worldX.toFixed(2)}, ${placement.worldY.toFixed(2)}, ${placement.worldZ.toFixed(2)})<br>
            Elevation: ${placement.worldZ.toFixed(2)} yards<br>
            Tile: [${placement.tileRow}, ${placement.tileCol}]<br>
            Flags: 0x${placement.flags.toString(16).padStart(4, '0')}
        `;
    }
}
```

---

## 🎛️ Plugin Manager

### File: `js/core/PluginManager.js`

```javascript
export class PluginManager {
    constructor(map, coordSystem) {
        this.map = map;
        this.coords = coordSystem;
        this.plugins = new Map();
        
        // Setup viewport change listener
        this.map.on('moveend zoomend', () => {
            const bounds = this.map.getBounds();
            const zoom = this.map.getZoom();
            this.notifyViewportChange(bounds, zoom);
        });
    }
    
    register(plugin) {
        this.plugins.set(plugin.id, plugin);
        console.log(`[PluginManager] Registered: ${plugin.name}`);
    }
    
    unregister(pluginId) {
        const plugin = this.plugins.get(pluginId);
        if (plugin) {
            plugin.onDestroy();
            this.plugins.delete(pluginId);
            console.log(`[PluginManager] Unregistered: ${plugin.name}`);
        }
    }
    
    get(pluginId) {
        return this.plugins.get(pluginId);
    }
    
    async loadAll(version, mapName) {
        const promises = Array.from(this.plugins.values()).map(plugin =>
            plugin.onLoad(version, mapName)
        );
        await Promise.all(promises);
    }
    
    notifyViewportChange(bounds, zoom) {
        this.plugins.forEach(plugin => {
            plugin.onViewportChange(bounds, zoom);
        });
    }
    
    saveState() {
        const state = {};
        this.plugins.forEach((plugin, id) => {
            state[id] = plugin.getConfig();
        });
        localStorage.setItem('pluginState', JSON.stringify(state));
    }
    
    loadState() {
        const state = JSON.parse(localStorage.getItem('pluginState') || '{}');
        Object.entries(state).forEach(([id, config]) => {
            const plugin = this.plugins.get(id);
            if (plugin) {
                plugin.setConfig(config);
            }
        });
    }
}
```

---

## 📁 File Structure

```
WoWRollback.Viewer/
├── assets/
│   ├── js/
│   │   ├── core/
│   │   │   ├── CoordinateSystem.js      ← Canonical transforms
│   │   │   ├── OverlayPlugin.js         ← Base plugin class
│   │   │   └── PluginManager.js         ← Plugin lifecycle
│   │   │
│   │   ├── plugins/
│   │   │   ├── M2Plugin.js              ← M2 doodads
│   │   │   ├── WMOPlugin.js             ← WMO objects
│   │   │   ├── TerrainPlugin.js         ← Terrain properties
│   │   │   ├── LiquidsPlugin.js         ← Liquids
│   │   │   ├── AreaBoundariesPlugin.js  ← Area boundaries
│   │   │   ├── HolesPlugin.js           ← Terrain holes
│   │   │   └── GridPlugin.js            ← ADT/Chunk grid
│   │   │
│   │   ├── ui/
│   │   │   ├── PluginPanel.js           ← UI controls
│   │   │   └── PluginOptions.js         ← Settings modal
│   │   │
│   │   ├── state.js                     ← Global state
│   │   └── main.js                      ← Entry point
│   │
│   ├── styles.css
│   └── index.html
│
└── WoWRollback.Viewer.csproj
```

---

## 🚀 Main Entry Point

### File: `js/main.js`

```javascript
import { CoordinateSystem } from './core/CoordinateSystem.js';
import { PluginManager } from './core/PluginManager.js';
import { M2Plugin } from './plugins/M2Plugin.js';
import { WMOPlugin } from './plugins/WMOPlugin.js';
import { GridPlugin } from './plugins/GridPlugin.js';
import { state } from './state.js';

let map;
let coordSystem;
let pluginManager;

export async function init() {
    try {
        // Load config
        await state.loadConfig();
        await state.loadIndex();
        
        // Initialize coordinate system
        coordSystem = new CoordinateSystem(state.config);
        
        // Initialize map
        map = L.map('map', {
            crs: L.CRS.Simple,
            minZoom: 0,
            maxZoom: 12,
            zoom: 2
        });
        
        map.setView([32, 32], 2);
        
        // Initialize plugin manager
        pluginManager = new PluginManager(map, coordSystem);
        
        // Register plugins
        pluginManager.register(new GridPlugin(map, coordSystem));
        pluginManager.register(new M2Plugin(map, coordSystem));
        pluginManager.register(new WMOPlugin(map, coordSystem));
        
        // Load all plugins
        await pluginManager.loadAll(state.selectedVersion, state.selectedMap);
        
        // Restore state
        pluginManager.loadState();
        
        // Setup UI
        setupUI();
        
        console.log('[Viewer] Initialized successfully');
    } catch (error) {
        console.error('[Viewer] Initialization failed:', error);
    }
}

function setupUI() {
    // Plugin toggles
    document.querySelectorAll('.plugin-toggle').forEach(toggle => {
        toggle.addEventListener('change', (e) => {
            const plugin = pluginManager.get(e.target.dataset.plugin);
            if (e.target.checked) {
                plugin.onEnable();
                plugin.onShow();
            } else {
                plugin.onHide();
                plugin.onDisable();
            }
            pluginManager.saveState();
        });
    });
}

// Auto-init
document.addEventListener('DOMContentLoaded', init);
```

---

## 🎨 UI Controls

### File: `index.html` (excerpt)

```html
<div id="pluginPanel" class="sidebar-panel">
    <h3>Overlays</h3>
    
    <div class="plugin-control">
        <input type="checkbox" id="plugin-grid" class="plugin-toggle" data-plugin="grid" checked>
        <label for="plugin-grid">ADT Grid</label>
        <button class="plugin-options" data-plugin="grid">⚙️</button>
    </div>
    
    <div class="plugin-control">
        <input type="checkbox" id="plugin-m2" class="plugin-toggle" data-plugin="m2" checked>
        <label for="plugin-m2">M2 Doodads</label>
        <span class="plugin-icon" style="color: #2196F3;">●</span>
        <button class="plugin-options" data-plugin="m2">⚙️</button>
    </div>
    
    <div class="plugin-control">
        <input type="checkbox" id="plugin-wmo" class="plugin-toggle" data-plugin="wmo" checked>
        <label for="plugin-wmo">WMO Objects</label>
        <span class="plugin-icon" style="color: #FF9800;">■</span>
        <button class="plugin-options" data-plugin="wmo">⚙️</button>
    </div>
</div>
```

---

## ✅ Implementation Checklist

### Phase 1: Core (Week 1)
- [ ] Create `CoordinateSystem.js` with all transforms
- [ ] Create `OverlayPlugin.js` base class
- [ ] Create `PluginManager.js`
- [ ] Test coordinate conversions thoroughly

### Phase 2: First Plugin (Week 1)
- [ ] Create `GridPlugin.js` (simple, no data loading)
- [ ] Verify plugin lifecycle works
- [ ] Test viewport change notifications
- [ ] Ensure grid aligns with tiles

### Phase 3: Data Plugins (Week 2)
- [ ] Create `M2Plugin.js`
- [ ] Create `WMOPlugin.js`
- [ ] Implement proximity loading
- [ ] Implement zoom-level optimization

### Phase 4: UI & State (Week 2)
- [ ] Plugin toggle controls
- [ ] Options modals
- [ ] State persistence (localStorage)
- [ ] Preset views

### Phase 5: Advanced Plugins (Week 3)
- [ ] `TerrainPlugin.js`
- [ ] `LiquidsPlugin.js`
- [ ] `AreaBoundariesPlugin.js`
- [ ] Performance optimization

---

## 🎯 Success Criteria

### Architecture
- [ ] Adding new plugin requires ONE file
- [ ] No changes to core system for new plugins
- [ ] Plugins are truly independent

### Performance
- [ ] Handles millions of data points
- [ ] Only loads visible data
- [ ] Smooth panning/zooming

### Maintainability
- [ ] Clear file structure
- [ ] Well-documented APIs
- [ ] Easy to debug

### User Experience
- [ ] Simple toggle controls
- [ ] State persists across sessions
- [ ] Visual feedback for all actions

---

**Start with CoordinateSystem.js - get that right, everything else follows.** 🎯

---

## 🌐 Future: 3D Viewer Integration

### WDL Terrain Meshes

**You already have:**
- WDL → GLB converter
- Minimap textures baked into terrain
- Low-resolution terrain meshes

**3D Viewer Architecture:**
```javascript
// Future: js/viewers/Viewer3D.js
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

class Viewer3D {
    constructor(container, coordSystem) {
        this.coords = coordSystem;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, width/height, 0.1, 10000);
        this.renderer = new THREE.WebGLRenderer();
        
        // Load WDL terrain
        this.terrainMesh = null;
        this.placements = [];
    }
    
    async loadTerrain(mapName) {
        const loader = new GLTFLoader();
        const gltf = await loader.loadAsync(`terrain/${mapName}.glb`);
        this.terrainMesh = gltf.scene;
        this.scene.add(this.terrainMesh);
    }
    
    plotPlacement(placement) {
        // Convert world coords to 3D position
        const position = new THREE.Vector3(
            placement.worldX,
            placement.worldZ,  // Z becomes Y in Three.js
            placement.worldY
        );
        
        // Create marker at exact 3D position
        const geometry = new THREE.SphereGeometry(5);
        const material = new THREE.MeshBasicMaterial({ color: 0x2196F3 });
        const marker = new THREE.Mesh(geometry, material);
        marker.position.copy(position);
        
        this.scene.add(marker);
        this.placements.push(marker);
    }
}
```

### Migration Path: 2D → 2.5D → 3D

**Phase 1: 2D Viewer (Current) - Week 1-2**
- Top-down projection (Leaflet)
- Elevation as visual cues (color/size)
- Fast, lightweight
- Minimap PNG tiles as base layer

**Phase 2: 2.5D Isometric Viewer (Intermediate) - Week 3-4**
- **WDL terrain meshes** (low-res, perfect for overview)
- **Isometric camera** (45° angle, fixed rotation)
- **Three.js canvas** overlaid on Leaflet
- **Minimap textures** baked onto WDL terrain
- **Placement markers** rendered in 3D space
- **Still top-down navigation** (pan/zoom like 2D)
- **Better depth perception** without full 3D complexity

**Phase 3: 3D Viewer (Future) - Month 2+**
- Full 3D navigation
- ADT high-res terrain meshes
- Free camera controls (orbit, fly)
- Real model loading (M2/WMO)
- Lighting and shadows

**Shared Components:**
- Same `CoordinateSystem.js`
- Same plugin architecture
- Same data format (3D coords)
- Toggle between 2D/3D views

### Data Compatibility

**All plugins work in both viewers:**
```javascript
class M2Plugin extends OverlayPlugin {
    // Works in 2D viewer
    render2D(placement) {
        const { lat, lng } = this.coords.worldToLatLng(
            placement.worldX, placement.worldY
        );
        return L.circleMarker([lat, lng], { ... });
    }
    
    // Works in 3D viewer
    render3D(placement) {
        const position = new THREE.Vector3(
            placement.worldX,
            placement.worldZ,
            placement.worldY
        );
        return new THREE.Mesh(geometry, material);
    }
}
```

### Benefits of 3D Approach

**Better Analysis:**
- See elevation relationships
- Understand terrain context
- Visualize object stacking
- Identify placement errors

**Better Debugging:**
- Spot floating objects
- Find underground placements
- Verify terrain alignment
- Check LOD distances

**Better Presentation:**
- Impressive visualizations
- Client demos
- Documentation screenshots
- Video walkthroughs

---

## 📋 Data Pipeline Requirements

### C# Side: Extract Full 3D Coordinates

**Update `PlacementRecord`:**
```csharp
public class PlacementRecord
{
    public uint UniqueId { get; set; }
    public string ModelPath { get; set; }
    
    // 3D World Coordinates (from MODF/MDDF)
    public float WorldX { get; set; }
    public float WorldY { get; set; }
    public float WorldZ { get; set; }
    
    // Rotation (for 3D viewer)
    public float RotX { get; set; }
    public float RotY { get; set; }
    public float RotZ { get; set; }
    
    // Scale (for 3D viewer)
    public ushort Scale { get; set; }
    
    // Flags (for visualization)
    public ushort Flags { get; set; }
    
    // Tile reference
    public int TileRow { get; set; }
    public int TileCol { get; set; }
}
```

**JSON Output:**
```json
{
    "uniqueId": 12345,
    "modelPath": "World\\Azeroth\\Elwynn\\PassiveObjects\\Trees\\ElwynnTree01.m2",
    "worldX": 1234.56,
    "worldY": -789.12,
    "worldZ": 42.34,
    "rotX": 0.0,
    "rotY": 1.57,
    "rotZ": 0.0,
    "scale": 1024,
    "flags": 0,
    "tileRow": 30,
    "tileCol": 35
}
```

**This data works for:**
- ✅ 2D viewer (use X/Y, visualize Z)
- ✅ 3D viewer (use X/Y/Z + rotation + scale)
- ✅ Analysis tools (full spatial data)
- ✅ Export tools (complete placement info)

---

**Start with 2D, but design for 3D from day one.** 🎯

---

## 🎨 Phase 2: 2.5D Isometric Viewer (The Sweet Spot)

### Why 2.5D is Perfect for This

**You already have the tools:**
- ✅ `WdlGltfExporter.cs` - Exports WDL terrain as GLB
- ✅ `ADTPreFabTool` - Exports ADT with minimap textures baked in
- ✅ Low-res terrain meshes (17x17 vertices per tile)
- ✅ Minimap textures ready to apply

**Benefits over pure 2D:**
- **See elevation** - Mountains look like mountains
- **Better context** - Understand terrain relationships
- **Depth perception** - Objects at different heights are obvious
- **Still fast** - Low-poly WDL meshes are lightweight
- **Easy navigation** - Same pan/zoom as 2D

**Benefits over full 3D:**
- **Simpler** - Fixed isometric camera, no complex controls
- **Faster** - No need for high-res ADT meshes
- **Focused** - Analysis tool, not a game engine
- **Familiar** - Like strategy games (StarCraft, Age of Empires)

### Architecture: Hybrid Leaflet + Three.js

```javascript
// File: js/viewers/Viewer2_5D.js
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

class Viewer2_5D {
    constructor(container, coordSystem, leafletMap) {
        this.coords = coordSystem;
        this.leafletMap = leafletMap;
        
        // Three.js setup
        this.scene = new THREE.Scene();
        this.camera = new THREE.OrthographicCamera(-1000, 1000, 1000, -1000, 0.1, 10000);
        this.renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        
        // Isometric camera (45° angle, looking down)
        this.camera.position.set(1000, 1000, 1000);
        this.camera.lookAt(0, 0, 0);
        
        // Overlay Three.js canvas on Leaflet
        this.renderer.domElement.style.position = 'absolute';
        this.renderer.domElement.style.top = '0';
        this.renderer.domElement.style.left = '0';
        this.renderer.domElement.style.pointerEvents = 'none'; // Let Leaflet handle input
        container.appendChild(this.renderer.domElement);
        
        // Sync with Leaflet viewport
        this.leafletMap.on('move zoom', () => this.syncCamera());
        
        this.terrainMeshes = [];
        this.placementMarkers = [];
    }
    
    async loadWdlTerrain(mapName) {
        const loader = new GLTFLoader();
        
        // Load merged WDL GLB (all tiles in one file)
        const gltf = await loader.loadAsync(`terrain/${mapName}_wdl.glb`);
        
        // WDL mesh already has minimap textures baked in!
        this.terrainMeshes.push(gltf.scene);
        this.scene.add(gltf.scene);
        
        console.log('[2.5D] Loaded WDL terrain:', mapName);
    }
    
    plotPlacement(placement) {
        // Convert world coords to Three.js position
        // WoW: X=east, Y=south, Z=up
        // Three.js: X=east, Y=up, Z=south (need to swap)
        const position = new THREE.Vector3(
            placement.worldX,
            placement.worldZ,  // Z becomes Y (elevation)
            placement.worldY   // Y becomes Z (depth)
        );
        
        // Create marker (sphere or billboard)
        const geometry = new THREE.SphereGeometry(5);
        const material = new THREE.MeshBasicMaterial({ 
            color: this.getMarkerColor(placement),
            transparent: true,
            opacity: 0.8
        });
        const marker = new THREE.Mesh(geometry, material);
        marker.position.copy(position);
        
        // Add to scene
        this.scene.add(marker);
        this.placementMarkers.push(marker);
        
        // Store placement data for tooltips
        marker.userData = { placement };
    }
    
    syncCamera() {
        // Sync Three.js camera with Leaflet viewport
        const bounds = this.leafletMap.getBounds();
        const center = this.leafletMap.getCenter();
        const zoom = this.leafletMap.getZoom();
        
        // Convert Leaflet coords to world coords
        const { worldX, worldY } = this.coords.latLngToWorld(center.lat, center.lng);
        
        // Update camera position (maintain isometric angle)
        const distance = 1000 / Math.pow(2, zoom - 2);
        this.camera.position.set(worldX + distance, distance, worldY + distance);
        this.camera.lookAt(worldX, 0, worldY);
        
        // Update orthographic camera bounds based on zoom
        const size = 500 / Math.pow(2, zoom - 2);
        this.camera.left = -size;
        this.camera.right = size;
        this.camera.top = size;
        this.camera.bottom = -size;
        this.camera.updateProjectionMatrix();
        
        this.render();
    }
    
    render() {
        this.renderer.render(this.scene, this.camera);
    }
    
    getMarkerColor(placement) {
        // Same flag-based coloring as 2D viewer
        if (placement.flags & 0x0001) return 0xFF0000;
        if (placement.flags & 0x0002) return 0x00FF00;
        return 0x2196F3;
    }
}
```

### Integration with Existing Viewer

```javascript
// File: js/main.js
let viewer2D;   // Leaflet-based
let viewer2_5D; // Three.js overlay
let viewMode = '2d'; // '2d' or '2.5d'

async function init() {
    // ... existing init code ...
    
    // Initialize 2D viewer (always)
    viewer2D = initLeafletMap();
    
    // Initialize 2.5D viewer (optional)
    if (state.config.enable2_5D) {
        viewer2_5D = new Viewer2_5D(mapContainer, coordSystem, viewer2D);
        await viewer2_5D.loadWdlTerrain(state.selectedMap);
    }
    
    // Toggle button
    document.getElementById('toggle2_5D').addEventListener('click', () => {
        viewMode = viewMode === '2d' ? '2.5d' : '2d';
        toggleViewMode();
    });
}

function toggleViewMode() {
    if (viewMode === '2.5d') {
        // Show Three.js canvas
        viewer2_5D.renderer.domElement.style.display = 'block';
        viewer2_5D.syncCamera();
        
        // Hide Leaflet tile layer (but keep map for navigation)
        minimapLayer.remove();
    } else {
        // Hide Three.js canvas
        viewer2_5D.renderer.domElement.style.display = 'none';
        
        // Show Leaflet tiles
        minimapLayer.addTo(viewer2D);
    }
}
```

### Existing Tools Integration

**C# Side - Export WDL with Textures:**
```csharp
// Use existing WdlGltfExporter
var wdl = WdlReader.Read(wdlPath);
var options = new WdlGltfExporter.ExportOptions(
    Scale: 1.0,
    SkipHoles: true,
    NormalizeWorld: true,
    HeightScale: 1.0
);

// Export merged GLB for entire map
var stats = WdlGltfExporter.ExportMerged(
    wdl, 
    $"viewer/terrain/{mapName}_wdl.glb", 
    options
);

// TODO: Add minimap texture application
// (You have this in ADTPreFabTool - port the texture baking logic)
```

**Minimap Texture Baking (from ADTPreFabTool):**
```csharp
// Port this logic from ADTPreFabTool to apply minimap textures to WDL mesh
// 1. Load minimap PNG for each tile
// 2. Create UV coordinates for WDL mesh
// 3. Apply texture to material
// 4. Export GLB with embedded textures
```

### UI Toggle

```html
<div class="view-mode-toggle">
    <button id="toggle2_5D" class="btn">
        <span class="icon">🗺️</span>
        <span class="label">2.5D View</span>
    </button>
</div>

<style>
.view-mode-toggle {
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 1000;
}

.view-mode-toggle .btn {
    background: #2196F3;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
}

.view-mode-toggle .btn:hover {
    background: #1976D2;
}
</style>
```

### Performance Characteristics

**WDL Mesh Stats (per map):**
- **Vertices**: ~70,000 (64x64 tiles × 17×17 vertices)
- **Faces**: ~130,000 triangles
- **Texture**: 1 atlas (4096×4096) for all minimap tiles
- **File Size**: ~5-10 MB GLB
- **Load Time**: <1 second
- **Render**: 60 FPS easily

**Comparison:**
- **2D Leaflet**: 64×64 = 4,096 PNG tiles, lazy loaded
- **2.5D WDL**: 1 GLB file, all terrain at once
- **3D ADT**: 64×64 high-res meshes, would be huge

### Implementation Priority

**Week 3: Foundation**
- [ ] Port WDL export to viewer pipeline
- [ ] Add minimap texture baking
- [ ] Test GLB loading in Three.js

**Week 4: Integration**
- [ ] Create `Viewer2_5D.js` class
- [ ] Sync camera with Leaflet
- [ ] Render placement markers in 3D

**Week 5: Polish**
- [ ] Toggle button UI
- [ ] Smooth transitions
- [ ] Performance optimization

---

**2.5D is the perfect middle ground - all the benefits of 3D visualization without the complexity!** 🎯

---

## 🔧 Critical Implementation Details

### Data Pipeline: C# Side

**Current State:**
- `AlphaWdtAnalyzer` extracts placement lists from Alpha WDT
- **Problem**: Coordinates are (0,0,0) - Alpha didn't store them
- **Solution**: Extract from converted LK ADT files (MODF/MDDF chunks)

**Files to modify:**
```
WoWRollback.Core/
├── Models/
│   └── PlacementRecord.cs          ← Add WorldX/Y/Z, Rotation, Scale, Flags
├── Services/
│   ├── AlphaWdtAnalyzer.cs         ← Add coordinate extraction
│   └── LkAdtReader.cs              ← Read MODF/MDDF from LK ADTs
└── Services/Viewer/
    └── OverlayGenerator.cs         ← Generate JSON with 3D coords
```

**PlacementRecord.cs:**
```csharp
public class PlacementRecord
{
    public uint UniqueId { get; set; }
    public string ModelPath { get; set; }
    
    // ADD THESE:
    public float WorldX { get; set; }
    public float WorldY { get; set; }
    public float WorldZ { get; set; }
    public float RotX { get; set; }
    public float RotY { get; set; }
    public float RotZ { get; set; }
    public ushort Scale { get; set; }
    public ushort Flags { get; set; }
    
    public int TileRow { get; set; }
    public int TileCol { get; set; }
}
```

**MODF/MDDF Structure (from wowdev.wiki):**
```
MDDF Entry (36 bytes):
  [0-3]   nameIndex (uint32)
  [4-7]   uniqueId (uint32)
  [8-11]  worldX (float32)
  [12-15] worldZ (float32)  ← Note: Z at offset 12
  [16-19] worldY (float32)  ← Note: Y at offset 16
  [20-23] rotX (float32)
  [24-27] rotY (float32)
  [28-31] rotZ (float32)
  [32-33] scale (uint16)
  [34-35] flags (uint16)

MODF Entry (64 bytes):
  [0-3]   nameIndex (uint32)
  [4-7]   uniqueId (uint32)
  [8-11]  worldX (float32)
  [12-15] worldZ (float32)
  [16-19] worldY (float32)
  [20-23] rotX (float32)
  [24-27] rotY (float32)
  [28-31] rotZ (float32)
  [32-35] scale (uint16)
  [36-37] flags (uint16)
  [38-63] bounding box + other data
```

**LkAdtReader.cs (create this):**
```csharp
public class LkAdtReader
{
    public static List<PlacementRecord> ReadMddf(string adtPath)
    {
        var placements = new List<PlacementRecord>();
        using var fs = File.OpenRead(adtPath);
        using var br = new BinaryReader(fs);
        
        // Find MDDF chunk (stored as "FDDM" on disk - byte-reversed)
        while (fs.Position < fs.Length - 8)
        {
            var fourcc = new string(br.ReadChars(4));
            var size = br.ReadUInt32();
            
            if (fourcc == "FDDM") // MDDF reversed
            {
                var entryCount = size / 36;
                for (int i = 0; i < entryCount; i++)
                {
                    var p = new PlacementRecord
                    {
                        // Read at correct offsets
                        UniqueId = br.ReadUInt32(),     // [4-7]
                        WorldX = br.ReadSingle(),       // [8-11]
                        WorldZ = br.ReadSingle(),       // [12-15]
                        WorldY = br.ReadSingle(),       // [16-19]
                        RotX = br.ReadSingle(),         // [20-23]
                        RotY = br.ReadSingle(),         // [24-27]
                        RotZ = br.ReadSingle(),         // [28-31]
                        Scale = br.ReadUInt16(),        // [32-33]
                        Flags = br.ReadUInt16()         // [34-35]
                    };
                    placements.Add(p);
                }
                break;
            }
            else
            {
                fs.Seek(size, SeekOrigin.Current);
            }
        }
        
        return placements;
    }
    
    // Similar for ReadModf()
}
```

### Viewer Pipeline: C# Side

**OverlayGenerator.cs - Generate JSON:**
```csharp
public class OverlayGenerator
{
    public void GenerateOverlays(
        List<PlacementRecord> placements,
        string outputDir,
        string mapName,
        string version)
    {
        // Group by tile
        var byTile = placements
            .GroupBy(p => (p.TileRow, p.TileCol))
            .ToList();
        
        foreach (var group in byTile)
        {
            var tileData = new
            {
                tileRow = group.Key.TileRow,
                tileCol = group.Key.TileCol,
                placements = group.Select(p => new
                {
                    uniqueId = p.UniqueId,
                    modelPath = p.ModelPath,
                    worldX = p.WorldX,
                    worldY = p.WorldY,
                    worldZ = p.WorldZ,
                    rotX = p.RotX,
                    rotY = p.RotY,
                    rotZ = p.RotZ,
                    scale = p.Scale,
                    flags = p.Flags,
                    tileRow = p.TileRow,
                    tileCol = p.TileCol
                }).ToList()
            };
            
            var json = JsonSerializer.Serialize(tileData, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            
            var path = Path.Combine(
                outputDir,
                "overlays",
                version,
                mapName,
                $"tile_{group.Key.TileRow}_{group.Key.TileCol}.json"
            );
            
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, json);
        }
    }
}
```

### Viewer Assets: File Structure

```
WoWRollback.Viewer/
├── assets/
│   ├── js/
│   │   ├── core/
│   │   │   ├── CoordinateSystem.js      ← 200 lines
│   │   │   ├── OverlayPlugin.js         ← 150 lines
│   │   │   └── PluginManager.js         ← 100 lines
│   │   │
│   │   ├── plugins/
│   │   │   ├── GridPlugin.js            ← 100 lines
│   │   │   ├── M2Plugin.js              ← 250 lines
│   │   │   └── WMOPlugin.js             ← 200 lines
│   │   │
│   │   ├── viewers/
│   │   │   └── Viewer2_5D.js            ← 300 lines (optional)
│   │   │
│   │   ├── state.js                     ← Keep existing
│   │   └── main.js                      ← 150 lines (rewrite)
│   │
│   ├── styles.css                       ← Update
│   └── index.html                       ← Update
│
└── WoWRollback.Viewer.csproj
```

### Testing Strategy

**Phase 1: Coordinate System**
```javascript
// Test in browser console
const coords = new CoordinateSystem({ coordMode: 'wowtools' });

// Test world → tile
const tile = coords.worldToTile(0, 0);
console.assert(tile.row === 32 && tile.col === 32, 'Center should be 32,32');

// Test tile → world
const world = coords.tileToWorld(32, 32);
console.assert(Math.abs(world.worldX) < 1 && Math.abs(world.worldY) < 1, 'Should be near 0,0');

// Test elevation
const color = coords.getElevationColor(100, '#2196F3');
console.log('Elevation color:', color); // Should be brighter than base
```

**Phase 2: Plugin System**
```javascript
// Test plugin registration
const pluginManager = new PluginManager(map, coords);
const gridPlugin = new GridPlugin(map, coords);
pluginManager.register(gridPlugin);

// Test lifecycle
gridPlugin.onEnable();
gridPlugin.onShow();
console.assert(gridPlugin.enabled && gridPlugin.visible, 'Plugin should be active');

// Test viewport change
const bounds = map.getBounds();
const zoom = map.getZoom();
pluginManager.notifyViewportChange(bounds, zoom);
```

**Phase 3: Data Loading**
```javascript
// Test M2 plugin data loading
const m2Plugin = pluginManager.get('m2');
await m2Plugin.onLoad('0.5.3', 'Azeroth');
console.log('M2 data loaded:', m2Plugin.data.length);

// Test proximity filtering
const visibleTiles = m2Plugin.getVisibleTiles(map.getBounds());
console.log('Visible tiles:', visibleTiles);
```

### Common Pitfalls to Avoid

**❌ DON'T:**
1. Mix coordinate systems (use CoordinateSystem.js for ALL transforms)
2. Load all data at once (use proximity-based loading)
3. Create plugins without base class (extend OverlayPlugin)
4. Hardcode colors/sizes (use flag-based styling)
5. Forget to cache 404s (causes infinite spam)
6. Use CSS borders for grid (use SVG or Leaflet rectangles)
7. Flip coordinates twice (one transform only!)

**✅ DO:**
1. Test coordinate transforms thoroughly first
2. Start with GridPlugin (simplest, no data)
3. Add console logging for debugging
4. Use browser dev tools Network tab
5. Cache loaded data per tile
6. Clear cache on version/map change
7. Save/load plugin state to localStorage

### Dependencies

**JavaScript (add to index.html):**
```html
<!-- Leaflet (already have) -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<!-- Three.js (for 2.5D viewer) -->
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }
}
</script>
```

**C# (already have):**
- Warcraft.NET (for ADT reading)
- SixLabors.ImageSharp (for minimap processing)
- SharpGLTF (for WDL export)

### Success Metrics

**Week 1:**
- [ ] CoordinateSystem.js passes all tests
- [ ] OverlayPlugin base class works
- [ ] PluginManager registers/unregisters plugins
- [ ] GridPlugin displays correctly

**Week 2:**
- [ ] M2Plugin loads and displays markers
- [ ] Proximity loading works (only visible tiles)
- [ ] Flag-based coloring works
- [ ] Elevation visualization works
- [ ] UI controls toggle plugins

**Week 3:**
- [ ] PlacementRecord has 3D coords + flags
- [ ] LkAdtReader extracts from MODF/MDDF
- [ ] OverlayGenerator creates JSON files
- [ ] Viewer loads real placement data

**Week 4 (optional):**
- [ ] WDL GLB exports with textures
- [ ] Viewer2_5D.js loads terrain
- [ ] Isometric camera syncs with Leaflet
- [ ] Toggle between 2D/2.5D works

---

## 🎯 Final Checklist for One-Shot Implementation

**Before starting new chat:**
- [ ] This document is complete and saved
- [ ] Old viewer committed as checkpoint
- [ ] You understand the coordinate system
- [ ] You know which files to create
- [ ] You have a testing strategy

**What to tell the AI:**
1. "Implement the overlay plugin system from `docs/planning/08-overlay-plugin-system.md`"
2. "Start with CoordinateSystem.js and test it thoroughly"
3. "Follow the file structure exactly as documented"
4. "Use the code examples provided"
5. "Test each phase before moving to the next"

**Expected outcome:**
- ✅ Working 2D viewer with plugin system
- ✅ Coordinate system that actually works
- ✅ Extensible architecture for new overlays
- ✅ Foundation for 2.5D/3D viewers
- ✅ No more coordinate chaos!

---

**This document is now complete and ready for one-shot implementation!** 🚀
