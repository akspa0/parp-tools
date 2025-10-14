# 3D Viewer Implementation Plan

## Overview

Implement a 3D visualization system for WoW map placements with terrain backdrop. The primary focus is **placement visualization** (M2/WMO objects in 3D space) with UniqueID layer filtering, reusing existing 2D viewer data. Terrain meshes serve as optional backdrop for spatial context.

**Phase 1 (Current):** Simple markers at placement coordinates
**Phase 2 (Future):** Actual M2/WMO model rendering (requires Alpha 0.5/0.6 format parsers)

## Goals

1. **Dual mesh export** - Export both GLB and OBJ formats for terrain
2. **3D placement viewer** - Visualize M2/WMO placements in 3D space
3. **UniqueID filtering** - Show/hide placement layers by ID ranges
4. **Data reuse** - Leverage existing 2D viewer placement JSONs
5. **Click interaction** - Show placement details on click (like 2D viewer)

## Current State

### What We Have
- ✅ GLB terrain mesh export per tile
- ✅ Mesh manifest JSON with tile metadata
- ✅ Placement data in `overlays/{version}/{mapName}/combined/` (per-tile JSONs)
- ✅ UniqueID range CSVs in `cached_maps/analysis/{version}/{mapName}/csv/`
- ✅ 2D viewer with full placement details

### What's Missing
- ❌ OBJ mesh export
- ❌ 3D viewer HTML/JS
- ❌ 3D placement marker rendering
- ❌ UniqueID layer filtering in 3D
- ❌ Click handlers for 3D markers

## Architecture

### Output Structure

```
{mapName}_mesh/
├── tile_30_41.glb          # GLB format (existing)
├── tile_30_41.obj          # OBJ format (NEW!)
├── tile_30_41.mtl          # Material file for OBJ (NEW!)
├── tile_30_42.glb
├── tile_30_42.obj
├── tile_30_42.mtl
└── mesh_manifest.json      # Tile metadata

viewer/
├── index.html              # 2D viewer (existing)
├── viewer3d.html           # 3D viewer (NEW!)
├── js/
│   ├── viewer3d.js         # Three.js setup (NEW!)
│   ├── placement-loader.js # Load placement data (NEW!)
│   ├── mesh-loader.js      # Load terrain GLB (NEW!)
│   ├── layer-filter.js     # UniqueID filtering (NEW!)
│   └── (existing 2D viewer JS files)
├── overlays/{version}/{mapName}/
│   ├── combined/           # Placement data (reuse existing)
│   └── mesh/               # Terrain meshes (existing)
└── cached_maps/analysis/{version}/{mapName}/csv/
    └── id_ranges_by_map.csv  # UniqueID ranges (reuse existing)
```

### Data Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Analysis Pipeline                                    │
│    ├─ Extract placements → combined/*.json              │
│    ├─ Extract terrain → {mapName}_terrain.csv           │
│    ├─ Extract meshes → {mapName}_mesh/*.glb + *.obj     │
│    └─ Generate ranges → id_ranges_by_map.csv            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 3D Viewer Loads Data                                 │
│    ├─ Load mesh_manifest.json (tile list)               │
│    ├─ Load combined/tile_*.json (placements)            │
│    ├─ Load id_ranges_by_map.csv (layer ranges)          │
│    └─ Optionally load mesh/*.glb (terrain backdrop)     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Render 3D Scene                                      │
│    ├─ Create 3D markers at placement coordinates        │
│    ├─ Apply UniqueID layer filters                      │
│    ├─ Optionally render terrain meshes                  │
│    └─ Add click handlers for details                    │
└─────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: OBJ Export (1-2 hours)

**Goal:** Add OBJ mesh export alongside GLB

**Tasks:**
1. Modify `AdtMeshExtractor.cs`:
   - Add `exportObj` parameter to `ExtractFromArchive()`
   - Implement `ExportOBJ()` method
   - Write vertices in OBJ format: `v x y z`
   - Write faces in OBJ format: `f v1 v2 v3`
   - Generate MTL file with basic material
   - Update manifest to include OBJ filenames

**Output Format:**
```obj
# tile_30_41.obj
mtllib tile_30_41.mtl
usemtl TerrainMaterial

v -8533.33 100.5 -13866.67
v -8500.00 105.2 -13866.67
...

f 1 2 3
f 2 4 3
...
```

```mtl
# tile_30_41.mtl
newmtl TerrainMaterial
Ka 0.6 0.6 0.6
Kd 0.6 0.6 0.6
Ks 0.0 0.0 0.0
```

**Acceptance Criteria:**
- ✅ OBJ files generated per tile
- ✅ MTL files generated per tile
- ✅ Mesh manifest includes OBJ filenames
- ✅ OBJ files load correctly in Blender/3D viewers

---

### Phase 2: Basic 3D Viewer (4-6 hours)

**Goal:** Create functional 3D viewer with placement markers

**Tasks:**

#### 2.1 HTML Structure (`viewer3d.html`)
```html
<!DOCTYPE html>
<html>
<head>
    <title>3D Viewer - {mapName}</title>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="viewer-container">
        <canvas id="viewer3d"></canvas>
        <div id="sidebar">
            <h3>Layers</h3>
            <div id="layer-list"></div>
            <div id="controls">
                <button id="select-all">Select All</button>
                <button id="deselect-all">Deselect All</button>
            </div>
            <div id="terrain-toggle">
                <label><input type="checkbox" id="show-terrain" checked> Show Terrain</label>
            </div>
        </div>
        <div id="details-popup" style="display:none;">
            <!-- Placement details -->
        </div>
    </div>
</body>
</html>
```

#### 2.2 Three.js Setup (`js/viewer3d.js`)
```javascript
class Viewer3D {
    constructor(mapName, version) {
        this.mapName = mapName;
        this.version = version;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50000);
        this.renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('viewer3d') });
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.markers = [];
        this.layers = new Map(); // UniqueID ranges
        
        this.init();
    }
    
    init() {
        // Setup scene
        this.scene.background = new THREE.Color(0x87CEEB); // Sky blue
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        this.scene.add(new THREE.DirectionalLight(0xffffff, 0.4));
        
        // Setup camera
        this.camera.position.set(0, 1000, 1000);
        this.camera.lookAt(0, 0, 0);
        
        // Setup controls
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        
        // Load data
        this.loadManifest();
        this.loadLayers();
        
        // Start render loop
        this.animate();
    }
    
    async loadManifest() {
        const manifest = await fetch(`overlays/${this.version}/${this.mapName}/mesh/mesh_manifest.json`).then(r => r.json());
        this.manifest = manifest;
        
        // Load placements for each tile
        for (const tile of manifest.tiles) {
            await this.loadTilePlacements(tile.x, tile.y);
        }
    }
    
    async loadTilePlacements(tileX, tileY) {
        const url = `overlays/${this.version}/${this.mapName}/combined/tile_r${tileY}_c${tileX}.json`;
        try {
            const data = await fetch(url).then(r => r.json());
            data.layers.forEach(layer => {
                layer.placements?.forEach(p => this.createMarker(p));
            });
        } catch (e) {
            console.log(`No placements for tile ${tileX}_${tileY}`);
        }
    }
    
    createMarker(placement) {
        // Create sphere marker
        const geometry = new THREE.SphereGeometry(10, 8, 8);
        const material = new THREE.MeshBasicMaterial({ 
            color: placement.kind === 'M2' ? 0xFF0000 : 0x0000FF 
        });
        const marker = new THREE.Mesh(geometry, material);
        
        // Position in world space
        marker.position.set(placement.worldX, placement.worldZ, -placement.worldY); // Note: Z up in Three.js
        
        // Store placement data
        marker.userData = {
            uniqueId: placement.uniqueId,
            assetPath: placement.assetPath,
            kind: placement.kind,
            worldX: placement.worldX,
            worldY: placement.worldY,
            worldZ: placement.worldZ
        };
        
        this.scene.add(marker);
        this.markers.push(marker);
    }
    
    async loadLayers() {
        const csv = await fetch(`cached_maps/analysis/${this.version}/${this.mapName}/csv/id_ranges_by_map.csv`).then(r => r.text());
        const lines = csv.split('\n').slice(1); // Skip header
        
        lines.forEach(line => {
            const [map, min, max, count] = line.split(',');
            if (map === this.mapName) {
                this.layers.set(`${min}-${max}`, {
                    min: parseInt(min),
                    max: parseInt(max),
                    count: parseInt(count),
                    enabled: true
                });
            }
        });
        
        this.renderLayerUI();
    }
    
    renderLayerUI() {
        const list = document.getElementById('layer-list');
        list.innerHTML = '';
        
        this.layers.forEach((layer, key) => {
            const div = document.createElement('div');
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = layer.enabled;
            checkbox.addEventListener('change', () => {
                layer.enabled = checkbox.checked;
                this.applyLayerFilters();
            });
            
            const label = document.createElement('label');
            label.textContent = ` Range ${key} (${layer.count} objects)`;
            
            div.appendChild(checkbox);
            div.appendChild(label);
            list.appendChild(div);
        });
    }
    
    applyLayerFilters() {
        this.markers.forEach(marker => {
            const uid = marker.userData.uniqueId;
            let visible = false;
            
            this.layers.forEach(layer => {
                if (layer.enabled && uid >= layer.min && uid <= layer.max) {
                    visible = true;
                }
            });
            
            marker.visible = visible;
        });
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}

// Initialize viewer
const viewer = new Viewer3D('Azeroth', '0.6.0.3592');
```

**Acceptance Criteria:**
- ✅ 3D scene renders with camera controls
- ✅ Placement markers appear at correct world coordinates
- ✅ Layer checkboxes show/hide markers by UniqueID range
- ✅ M2 and WMO markers have different colors

---

### Phase 3: Click Interaction (2-3 hours)

**Goal:** Add click handlers to show placement details

**Tasks:**

#### 3.1 Raycasting for Click Detection
```javascript
class Viewer3D {
    constructor() {
        // ... existing code ...
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.renderer.domElement.addEventListener('click', (e) => this.onClick(e));
    }
    
    onClick(event) {
        // Calculate mouse position in normalized device coordinates
        const rect = this.renderer.domElement.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        // Update raycaster
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        // Check for intersections
        const intersects = this.raycaster.intersectObjects(this.markers);
        
        if (intersects.length > 0) {
            const marker = intersects[0].object;
            this.showDetails(marker.userData);
        } else {
            this.hideDetails();
        }
    }
    
    showDetails(placement) {
        const popup = document.getElementById('details-popup');
        popup.innerHTML = `
            <h3>${placement.kind} Placement</h3>
            <p><strong>Asset:</strong> ${placement.assetPath}</p>
            <p><strong>UniqueID:</strong> ${placement.uniqueId}</p>
            <p><strong>World X:</strong> ${placement.worldX.toFixed(2)}</p>
            <p><strong>World Y:</strong> ${placement.worldY.toFixed(2)}</p>
            <p><strong>World Z:</strong> ${placement.worldZ.toFixed(2)}</p>
            <button onclick="viewer.hideDetails()">Close</button>
        `;
        popup.style.display = 'block';
    }
    
    hideDetails() {
        document.getElementById('details-popup').style.display = 'none';
    }
}
```

**Acceptance Criteria:**
- ✅ Click marker → popup shows placement details
- ✅ Popup displays: AssetPath, UniqueID, coordinates, kind
- ✅ Click empty space → popup closes
- ✅ Close button works

---

### Phase 4: Terrain Backdrop (Optional, 2-3 hours)

**Goal:** Load and render terrain meshes as backdrop

**Tasks:**

#### 4.1 GLB Loader
```javascript
class Viewer3D {
    constructor() {
        // ... existing code ...
        this.loader = new THREE.GLTFLoader();
        this.terrainMeshes = [];
    }
    
    async loadManifest() {
        // ... existing code ...
        
        // Optionally load terrain meshes
        if (document.getElementById('show-terrain').checked) {
            for (const tile of manifest.tiles) {
                await this.loadTerrainMesh(tile);
            }
        }
    }
    
    async loadTerrainMesh(tile) {
        const url = `overlays/${this.version}/${this.mapName}/mesh/${tile.glb}`;
        
        return new Promise((resolve, reject) => {
            this.loader.load(url, (gltf) => {
                const mesh = gltf.scene;
                mesh.userData.tile = tile;
                this.scene.add(mesh);
                this.terrainMeshes.push(mesh);
                resolve();
            }, undefined, reject);
        });
    }
}
```

**Acceptance Criteria:**
- ✅ Terrain meshes load from GLB files
- ✅ Meshes positioned correctly in world space
- ✅ Toggle terrain visibility with checkbox
- ✅ Terrain doesn't block marker clicks

---

## Technical Specifications

### Coordinate System

**WoW Coordinates:**
- X: East-West (increases eastward)
- Y: North-South (increases southward)
- Z: Height (increases upward)

**Three.js Coordinates:**
- X: Right-Left
- Y: Up-Down
- Z: Forward-Backward

**Conversion:**
```javascript
threeJS.x = wow.worldX;
threeJS.y = wow.worldZ;  // Height
threeJS.z = -wow.worldY; // Flip Y to Z
```

### Marker Styling

**M2 Markers:**
- Color: Red (`0xFF0000`)
- Size: 10 units radius

**WMO Markers:**
- Color: Blue (`0x0000FF`)
- Size: 10 units radius

**Hover State:**
- Increase size by 1.5x
- Add emissive glow

### Performance Considerations

**Marker Count:**
- Azeroth: ~28,000 placements
- Use instanced meshes if > 10,000 markers
- Frustum culling enabled by default

**Terrain Meshes:**
- Load on-demand per visible tile
- Unload off-screen tiles
- LOD system for distant tiles (future)

**Layer Filtering:**
- Filter markers in JavaScript (fast)
- No need to reload data

---

## Testing Plan

### Unit Tests
- ✅ OBJ export generates valid files
- ✅ Coordinate conversion is correct
- ✅ Layer filtering logic works

### Integration Tests
- ✅ 3D viewer loads placement data
- ✅ Markers appear at correct positions
- ✅ Click handlers work
- ✅ Layer filtering updates visibility

### Manual Tests
- ✅ Test with small map (development)
- ✅ Test with large map (Azeroth)
- ✅ Verify performance with 28K markers
- ✅ Test in Chrome, Firefox, Edge

---

## Acceptance Criteria

### Phase 1: OBJ Export
- ✅ OBJ files generated alongside GLB
- ✅ MTL files generated with basic material
- ✅ Files load correctly in Blender

### Phase 2: Basic 3D Viewer
- ✅ 3D scene renders with camera controls
- ✅ Placement markers visible at correct positions
- ✅ Layer checkboxes filter markers by UniqueID
- ✅ M2/WMO markers have different colors

### Phase 3: Click Interaction
- ✅ Click marker → show details popup
- ✅ Popup displays all placement info
- ✅ Click empty space → close popup

### Phase 4: Terrain Backdrop (Optional)
- ✅ Terrain meshes load from GLB
- ✅ Toggle terrain visibility
- ✅ Terrain doesn't interfere with marker clicks

---

### Future Enhancements

### v1.1 (Viewer Polish)
- Measure tool (distance between markers)
- Search/filter by AssetPath
- Export selected placements to CSV
- Screenshot/camera position save
- Bounding box visualization on hover

### v1.2 (Alpha Format Support)
- **Alpha 0.5/0.6 M2 parser** - Parse early M2 format
- **Alpha 0.5/0.6 WMO parser** - Parse early WMO format
- **Format converter** - Convert Alpha → Retail for wow.export compatibility
- **Texture extraction** - BLP from Alpha clients
- **Model cache** - Store converted models for reuse

### v1.3 (Actual Model Rendering)
- Load actual M2/WMO models (using converted formats)
- Integrate wow.export's M2Renderer/WMORenderer
- Texture terrain with minimap tiles
- LOD system for performance
- Progressive loading (markers first, models on-demand)

### v2.0 (Advanced Features)
- Multi-map comparison in 3D
- Time-travel animation (show/hide by version)
- Heatmap overlays (object density)
- Collision detection visualization
- Animated camera paths
- VR support

---

## Alpha Format Challenges

### Why We Can't Render Models Yet

**Alpha 0.5/0.6 formats are different:**
- M2 structure differs from retail (different chunk layout)
- WMO format has variations (different versions)
- Texture references use different paths
- Bone structures may differ
- Animation data incompatible with retail parsers

**What We Need First:**
1. Alpha M2 parser (C# or JS)
2. Alpha WMO parser (C# or JS)
3. Format converter (Alpha → Retail-compatible)
4. Texture path resolver (Alpha → Retail mapping)
5. Model cache system

**Current Solution:**
- Use **simple markers** (spheres/cubes) at placement coordinates
- Show **terrain meshes** as backdrop (already working)
- Click markers → show details (AssetPath, UniqueID, coords)
- **Later:** Add actual model rendering when parsers are ready

**Reference:**
- wow.export has retail M2/WMO renderers we can adapt
- But they expect retail format, not Alpha 0.5/0.6

---

## Dependencies

### C# Libraries
- ✅ WoWFormatLib - ADT parsing (already added)
- ✅ SharpGLTF - GLB export (already added)
- ⏳ Alpha M2 parser (future)
- ⏳ Alpha WMO parser (future)

### JavaScript Libraries
- Three.js v0.160.0 - 3D rendering
- OrbitControls - Camera controls
- GLTFLoader - Load terrain meshes
- ⏳ wow.export's M2Renderer (future, when Alpha parsers ready)
- ⏳ wow.export's WMORenderer (future, when Alpha parsers ready)

### Browser Requirements
- WebGL 2.0 support
- ES6 modules support
- Minimum: Chrome 90+, Firefox 88+, Edge 90+

---

## Timeline

### Current Implementation (Markers Only)
**Phase 1 (OBJ Export):** ✅ DONE! (1-2 hours)
**Phase 2 (Basic Viewer):** 4-6 hours (markers + camera)
**Phase 3 (Click Interaction):** 2-3 hours (raycasting + popups)
**Phase 4 (Terrain Backdrop):** 2-3 hours (optional GLB loading)

**Total:** 8-14 hours (6-11 hours without terrain)

### Future Implementation (Actual Models)
**v1.2 (Alpha Parsers):** 40-60 hours
- Alpha M2 format parser
- Alpha WMO format parser
- Format converter (Alpha → Retail)
- Texture path resolver
- Model cache system

**v1.3 (Model Rendering):** 20-30 hours
- Integrate wow.export renderers
- Adapt for Alpha formats
- LOD system
- Progressive loading
- Performance optimization

---

## Success Metrics

### Phase 1 (Current - Markers)
1. **Functionality:** All placement markers visible and clickable
2. **Performance:** 60 FPS with 28K markers (Azeroth)
3. **Usability:** Layer filtering works smoothly
4. **Data Reuse:** No duplicate data generation needed
5. **Compatibility:** Works in all major browsers

### Phase 2 (Future - Models)
1. **Accuracy:** Models render correctly from Alpha formats
2. **Performance:** 60 FPS with 1K+ models visible
3. **Compatibility:** Alpha 0.5/0.6 formats supported
4. **Quality:** Textures load and display correctly
5. **Fallback:** Graceful degradation to markers if model fails
