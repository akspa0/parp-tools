# Modular Overlay Architecture - Individual Layer System

**Goal**: Replace monolithic "combined" overlays with individual, manageable layers for each data type.

**Status**: 🚨 **ACTIVE** - Implement alongside flags tracking

---

## 🎯 Vision

**Current Problem**: Overlays are mixed together (combined, m2, wmo) - hard to manage and extend.

**New Architecture**: Each data type gets its own:
- **Layer Manager** (handles loading, filtering, rendering)
- **Toggle Control** (individual on/off in UI)
- **Filter Options** (specific to data type)
- **Visual Style** (colors, shapes, opacity)

**Benefits**:
- ✅ Easy to add new data types
- ✅ Independent control per layer
- ✅ Better performance (only load what's visible)
- ✅ Cleaner codebase
- ✅ Scalable architecture

---

## 📊 Layer Types & Managers

### **Core Layers** (Implement First)

#### 1. **M2 Doodad Layer**
**Manager**: `M2LayerManager.js`
```javascript
class M2LayerManager {
    constructor(map) {
        this.map = map;
        this.markers = [];
        this.visible = true;
        this.filterByFlags = false;
        this.selectedFlags = [];
    }
    
    async load(version, mapName) {
        // Load M2 placement data
        const data = await fetchM2Data(version, mapName);
        this.createMarkers(data);
    }
    
    createMarkers(placements) {
        placements.forEach(p => {
            if (p.modelPath.endsWith('.m2')) {
                const marker = this.createM2Marker(p);
                this.markers.push(marker);
            }
        });
    }
    
    show() {
        this.visible = true;
        this.markers.forEach(m => m.addTo(this.map));
    }
    
    hide() {
        this.visible = false;
        this.markers.forEach(m => m.remove());
    }
    
    filterByFlags(flags) {
        // Filter markers by flag criteria
    }
}
```

**UI Control**:
```html
<div class="layer-control">
    <input type="checkbox" id="layer-m2" checked>
    <label>M2 Doodads (Circle)</label>
    <button class="layer-options">⚙️</button>
</div>
```

---

#### 2. **WMO Object Layer**
**Manager**: `WmoLayerManager.js`
```javascript
class WmoLayerManager {
    constructor(map) {
        this.map = map;
        this.markers = [];
        this.visible = true;
    }
    
    async load(version, mapName) {
        const data = await fetchWmoData(version, mapName);
        this.createMarkers(data);
    }
    
    createMarkers(placements) {
        placements.forEach(p => {
            if (p.modelPath.endsWith('.wmo')) {
                const marker = this.createWmoMarker(p);
                this.markers.push(marker);
            }
        });
    }
    
    // ... show/hide/filter methods
}
```

**UI Control**:
```html
<div class="layer-control">
    <input type="checkbox" id="layer-wmo" checked>
    <label>WMO Objects (Square)</label>
    <button class="layer-options">⚙️</button>
</div>
```

---

#### 3. **Terrain Properties Layer**
**Manager**: `TerrainLayerManager.js`
```javascript
class TerrainLayerManager {
    constructor(map) {
        this.map = map;
        this.overlays = [];
        this.visible = false;
        this.showImpassable = true;
        this.showVertexColored = true;
        this.showMultiLayer = true;
    }
    
    async load(version, mapName) {
        const data = await fetchTerrainData(version, mapName);
        this.createOverlays(data);
    }
    
    createOverlays(terrainData) {
        terrainData.forEach(tile => {
            if (tile.flags & TERRAIN_FLAGS.IMPASSABLE) {
                this.overlays.push(this.createImpassableOverlay(tile));
            }
            // ... more types
        });
    }
    
    setSubfilters(options) {
        this.showImpassable = options.impassable;
        this.showVertexColored = options.vertexColored;
        this.showMultiLayer = options.multiLayer;
        this.refresh();
    }
}
```

**UI Control**:
```html
<div class="layer-control">
    <input type="checkbox" id="layer-terrain">
    <label>Terrain Properties</label>
    <button class="layer-options">⚙️</button>
</div>
<div class="layer-suboptions" data-layer="terrain">
    <label><input type="checkbox" checked> Impassable Areas (Red)</label>
    <label><input type="checkbox" checked> Vertex Colored (Green)</label>
    <label><input type="checkbox" checked> Multi-Layer (Blue)</label>
</div>
```

---

#### 4. **Liquids Layer**
**Manager**: `LiquidsLayerManager.js`
```javascript
class LiquidsLayerManager {
    constructor(map) {
        this.map = map;
        this.overlays = [];
        this.visible = false;
        this.showRivers = true;
        this.showOceans = true;
        this.showMagma = true;
        this.showSlime = true;
    }
    
    async load(version, mapName) {
        const data = await fetchLiquidsData(version, mapName);
        this.createOverlays(data);
    }
    
    createOverlays(liquidsData) {
        liquidsData.forEach(liquid => {
            const overlay = this.createLiquidOverlay(liquid);
            this.overlays.push(overlay);
        });
    }
    
    setSubfilters(options) {
        this.showRivers = options.rivers;
        this.showOceans = options.oceans;
        this.showMagma = options.magma;
        this.showSlime = options.slime;
        this.refresh();
    }
}
```

---

#### 5. **Area Boundaries Layer**
**Manager**: `AreaBoundariesLayerManager.js`
```javascript
class AreaBoundariesLayerManager {
    constructor(map) {
        this.map = map;
        this.boundaries = [];
        this.visible = true;
        this.opacity = 0.6;
    }
    
    async load(version, mapName) {
        const data = await fetchAreaData(version, mapName);
        this.createBoundaries(data);
    }
    
    setOpacity(value) {
        this.opacity = value;
        this.boundaries.forEach(b => b.setStyle({ fillOpacity: value }));
    }
}
```

---

### **Advanced Layers** (Implement Later)

#### 6. **Holes Layer**
**Manager**: `HolesLayerManager.js`
- Show terrain holes (caves, tunnels)
- Toggle individual hole visualization

#### 7. **Shadows Layer**
**Manager**: `ShadowsLayerManager.js`
- MCNK shadow data
- Adjustable darkness/opacity

#### 8. **Height Map Layer**
**Manager**: `HeightMapLayerManager.js`
- Visualize terrain elevation
- Heatmap style (low→blue, high→red)

#### 9. **Texture Layer**
**Manager**: `TextureLayerManager.js`
- Show texture distribution
- Color-code by texture type

#### 10. **Sound Emitters Layer**
**Manager**: `SoundLayerManager.js`
- Sound emitter placements
- Diamond markers
- Color by sound type

---

## 🏗️ Master Layer Manager

**File**: `ViewerAssets/js/LayerController.js`

```javascript
class LayerController {
    constructor(map) {
        this.map = map;
        this.layers = {
            m2: new M2LayerManager(map),
            wmo: new WmoLayerManager(map),
            terrain: new TerrainLayerManager(map),
            liquids: new LiquidsLayerManager(map),
            areas: new AreaBoundariesLayerManager(map),
            holes: new HolesLayerManager(map),
            shadows: new ShadowsLayerManager(map),
            heightMap: new HeightMapLayerManager(map),
            textures: new TextureLayerManager(map),
            sounds: new SoundLayerManager(map)
        };
    }
    
    async loadAll(version, mapName) {
        const promises = Object.values(this.layers).map(layer => 
            layer.load(version, mapName)
        );
        await Promise.all(promises);
    }
    
    toggleLayer(layerId, visible) {
        const layer = this.layers[layerId];
        if (visible) {
            layer.show();
        } else {
            layer.hide();
        }
    }
    
    getLayer(layerId) {
        return this.layers[layerId];
    }
    
    // Save/load layer state to localStorage
    saveState() {
        const state = {};
        Object.entries(this.layers).forEach(([id, layer]) => {
            state[id] = {
                visible: layer.visible,
                options: layer.getOptions()
            };
        });
        localStorage.setItem('layerState', JSON.stringify(state));
    }
    
    loadState() {
        const state = JSON.parse(localStorage.getItem('layerState') || '{}');
        Object.entries(state).forEach(([id, config]) => {
            const layer = this.layers[id];
            if (layer) {
                layer.visible = config.visible;
                layer.setOptions(config.options);
            }
        });
    }
}
```

---

## 🎨 UI Architecture

### **Layer Control Panel**

**File**: `ViewerAssets/index.html`

```html
<div id="layerPanel" class="sidebar-panel">
    <h3>Map Layers</h3>
    
    <!-- Preset Views -->
    <div class="preset-views">
        <button onclick="loadPreset('objects')">Objects Only</button>
        <button onclick="loadPreset('terrain')">Terrain Only</button>
        <button onclick="loadPreset('all')">All Layers</button>
        <button onclick="loadPreset('custom')">Custom</button>
    </div>
    
    <hr>
    
    <!-- Object Layers -->
    <div class="layer-category">
        <h4>Objects</h4>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-m2" checked>
            <label>M2 Doodads</label>
            <span class="layer-icon">🔵</span>
            <button class="layer-options" onclick="openLayerOptions('m2')">⚙️</button>
        </div>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-wmo" checked>
            <label>WMO Objects</label>
            <span class="layer-icon">⬛</span>
            <button class="layer-options" onclick="openLayerOptions('wmo')">⚙️</button>
        </div>
    </div>
    
    <!-- Terrain Layers -->
    <div class="layer-category">
        <h4>Terrain</h4>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-terrain">
            <label>Terrain Properties</label>
            <button class="layer-options" onclick="openLayerOptions('terrain')">⚙️</button>
        </div>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-liquids">
            <label>Liquids</label>
            <button class="layer-options" onclick="openLayerOptions('liquids')">⚙️</button>
        </div>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-holes">
            <label>Holes</label>
            <button class="layer-options" onclick="openLayerOptions('holes')">⚙️</button>
        </div>
    </div>
    
    <!-- Administrative Layers -->
    <div class="layer-category">
        <h4>Administrative</h4>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-areas" checked>
            <label>Area Boundaries</label>
            <button class="layer-options" onclick="openLayerOptions('areas')">⚙️</button>
        </div>
    </div>
    
    <!-- Advanced Layers -->
    <div class="layer-category collapsed">
        <h4>Advanced <span class="toggle">▼</span></h4>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-shadows">
            <label>Shadows</label>
        </div>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-heightmap">
            <label>Height Map</label>
        </div>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-textures">
            <label>Textures</label>
        </div>
        
        <div class="layer-control">
            <input type="checkbox" id="layer-sounds">
            <label>Sound Emitters</label>
        </div>
    </div>
</div>
```

---

## 🔄 Migration Path (Phase Out Combined View)

### **Phase 1**: Implement Core Layers ✅
1. Create M2LayerManager
2. Create WmoLayerManager
3. Create TerrainLayerManager
4. Create LiquidsLayerManager
5. Create AreaBoundariesLayerManager

### **Phase 2**: Parallel Mode (Transition)
- Keep old "combined" overlay option
- Add new "individual layers" option
- Let users choose during testing
- Mark "combined" as **deprecated**

### **Phase 3**: Default to New System
- New layers become default
- "Combined" view still available but hidden
- Show migration notice

### **Phase 4**: Remove Combined View
- Delete old overlay code
- Pure modular system
- Clean up deprecated assets

---

## 📁 File Structure

```
ViewerAssets/
├── js/
│   ├── layers/
│   │   ├── LayerController.js         (master controller)
│   │   ├── M2LayerManager.js
│   │   ├── WmoLayerManager.js
│   │   ├── TerrainLayerManager.js
│   │   ├── LiquidsLayerManager.js
│   │   ├── AreaBoundariesLayerManager.js
│   │   ├── HolesLayerManager.js
│   │   ├── ShadowsLayerManager.js
│   │   ├── HeightMapLayerManager.js
│   │   ├── TextureLayerManager.js
│   │   └── SoundLayerManager.js
│   ├── ui/
│   │   ├── LayerPanel.js              (UI controls)
│   │   ├── LayerOptions.js            (options modals)
│   │   └── PresetManager.js           (view presets)
│   └── legacy/
│       └── CombinedOverlays.js        (deprecated)
```

---

## 🎯 Success Criteria

### Data Pipeline
- [ ] Each layer has its own data loader
- [ ] Data fetched independently per layer
- [ ] Caching per layer (don't reload if already loaded)

### UI
- [ ] Toggle each layer individually
- [ ] Options button opens layer-specific settings
- [ ] Preset views (Objects Only, Terrain Only, All, Custom)
- [ ] Collapsible category sections
- [ ] Icons showing layer style (circle, square, etc.)

### Performance
- [ ] Only load visible layers
- [ ] Lazy loading for advanced layers
- [ ] No performance regression vs old system

### User Experience
- [ ] Layer state persists (localStorage)
- [ ] Clear visual feedback (icons, colors)
- [ ] Easy to understand at a glance
- [ ] Migration path for existing users

---

## 🚀 Implementation Priority

1. **IMMEDIATE**: LayerController + M2/WMO managers
2. **HIGH**: Terrain and Liquids managers
3. **MEDIUM**: Area boundaries manager
4. **FUTURE**: Advanced layers (shadows, heightmap, etc.)

---

**Start with LayerController scaffold and M2LayerManager!** 🎨
