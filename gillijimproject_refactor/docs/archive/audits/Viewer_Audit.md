# WoWRollback Viewer Audit

**Date**: 2025-10-04  
**Purpose**: Document current viewer architecture before Phase 1 migration  
**Target**: `WoWRollback/ViewerAssets/` → `WoWRollback.Viewer` project

---

## Executive Summary

The viewer is a **Leaflet-based web application** that visualizes WoW map data with overlay layers. It works well but has **monolithic overlay management** without a plugin system.

### Current State
- **Location**: `WoWRollback/ViewerAssets/`
- **Technology**: Vanilla JS (ES6 modules), Leaflet.js, HTML5 Canvas
- **Architecture**: Monolithic - 6 hardcoded overlay types
- **Data Flow**: JSON overlays → OverlayManager → Renderers → Leaflet

### Migration Goal
Extract overlays into independent plugins with manifest-driven discovery.

---

## File Structure

### Root (10 files, 43KB)
```
ViewerAssets/
├── index.html (7.7KB)         # Main entry point
├── test.html (4.7KB)          # Debug page
├── debug-tiles.html (3KB)     # Tile coordinate tool
├── styles.css (6.1KB)         # Main styles
├── README.md (2.2KB)
└── CHANGELOG_QOL.md (3.2KB)
```

### JavaScript (7 files, 79KB)
```
js/
├── main.js (40KB)             # Entry point, map init
├── state.js (3.7KB)           # Global state
├── tile.js (13.9KB)           # Tile detail view
├── tileCanvas.js (3.8KB)      # Canvas utils
├── overlayLoader.js (0.9KB)   # JSON fetching
├── fit.js (3.5KB)             # Map utilities
└── overlays/
    ├── overlayManager.js (7.7KB)         # Coordinator
    ├── terrainPropertiesLayer.js (4KB)   # Height/flags
    ├── areaIdLayer.js (6.8KB)            # Zone boundaries
    ├── holesLayer.js (2.8KB)             # Terrain holes
    ├── liquidsLayer.js (3.1KB)           # Water/lava
    └── shadowMapLayer.js (4.2KB)         # Shadows
```

---

## Data Flow

```
1. User loads index.html
2. state.loadIndex() → fetch index.json
3. Initialize Leaflet map
4. User selects version/map
5. state.js fires change event
6. OverlayManager.loadVisibleOverlays()
7. For each visible tile:
   - terrainPropertiesLayer.loadTile()
   - areaIdLayer.loadTile()
   - etc.
8. Layers fetch JSON via overlayLoader
9. Render to Leaflet canvas
10. User toggles layers via checkboxes
```

---

## Overlay Layers

All layers follow an **implicit interface**:

```javascript
class SomeLayer {
    constructor(map) { }
    async loadTile(tileCoord, mapName, version) { }
    renderTileData(tileCoord, data) { }
    clear() { }
}
```

### Layer Complexity

| Layer | Data Source | Technology | Complexity |
|-------|-------------|------------|------------|
| Terrain Properties | terrain JSON | Canvas | Low |
| Area IDs | terrain JSON | Canvas | Medium |
| Holes | terrain JSON | Canvas | Low |
| Liquids | terrain JSON | Canvas | Low |
| Shadow Maps | JSON + PNG | ImageOverlay | Medium |
| Objects | objects JSON | CircleMarker | High |

**Note**: Objects live in `main.js`, NOT in overlayManager!

---

## Pain Points

### 1. Monolithic Architecture
- 6 layers hardcoded in overlayManager.js
- Adding overlay = edit 3+ files
- No runtime disable/enable
- Can't A/B test implementations

### 2. No Manifest
- Viewer doesn't know which overlays exist
- Relies on 404 handling
- No "sparse tile" awareness
- Retries missing tiles repeatedly

### 3. Inconsistent Object Handling
- Objects in main.js (not overlayManager)
- Different lifecycle
- Hard to extend

### 4. Tight Coupling
- Layers directly call loadOverlay()
- No dependency injection
- Hard to test in isolation

### 5. Performance
- Reloads tiles on every pan/zoom
- No tile unloading (memory leak)
- Fetches disabled layers

---

## Migration: Overlay → Plugin Mapping

### Target Structure
```
WoWRollback.Viewer/
├── js/
│   ├── runtime/
│   │   ├── runtime.js          # Plugin manager
│   │   └── plugin-interface.js # Lifecycle contract
│   └── plugins/
│       ├── terrain.js          # = terrainPropertiesLayer
│       ├── areaId.js           # = areaIdLayer
│       ├── holes.js            # = holesLayer
│       ├── liquids.js          # = liquidsLayer
│       ├── shadow.js           # = shadowMapLayer
│       ├── objects.js          # Extract from main.js
│       └── timeline.js         # NEW: Rollback feature
└── overlay_manifest.json
```

### Plugin Interface (target)
```javascript
class OverlayPlugin {
    async initialize(context) { }
    async loadTile(tileCoord) { }
    render(tileCoord, data) { }
    teardown() { }
}
```

---

## Migration Process (Per-Plugin)

### 1. Create Plugin (Isolated)
Copy layer code to `js/plugins/{name}.js`, adapt to interface.

### 2. Side-by-Side Testing
Add URL flag `?plugin_{name}=1`, run both systems.

### 3. Validation
- SHA256 comparison of rendered canvases
- Performance benchmarks
- Memory leak detection

### 4. Promotion
- Week 1: Opt-in
- Week 2: Default on
- Week 3: Remove old code

---

## Feature Flags

### URL Parameters
```javascript
const params = new URLSearchParams(window.location.search);
const useRuntime = params.get('use_runtime') === '1';

if (useRuntime) {
    const runtime = new Runtime();
    await runtime.loadManifest('./overlay_manifest.json');
    overlayManager = runtime.createManager(map);
} else {
    overlayManager = new OverlayManager(map);
}
```

### Per-Plugin Flags
```javascript
const useTerrainPlugin = params.get('plugin_terrain') !== '0'; // Default ON
```

---

## Manifest Schema (Phase 2)

```json
{
  "version": "0.5.3.3368",
  "map": "Azeroth",
  "overlays": [
    {
      "id": "terrain.properties",
      "plugin": "terrain",
      "title": "Terrain Properties",
      "tiles": "sparse",
      "resources": {
        "tilePattern": "overlays/{version}/{map}/terrain_complete/tile_{col}_{row}.json"
      }
    }
  ]
}
```

---

## Rollback Plan

### Immediate (<1 minute)
```bash
# Remove URL flag:
http://localhost:8080/index.html

# Old system works instantly
```

### Code Rollback
```powershell
git restore WoWRollback/ViewerAssets/
.\rebuild-and-regenerate.ps1 -Maps DeadminesInstance
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] WoWRollback.Viewer project builds
- [ ] Assets copied to output
- [ ] Old viewer works (flag OFF)
- [ ] New viewer works (flag ON)
- [ ] SHA256: old == new outputs

### All Phases Complete When:
- [ ] 6 overlays → plugins
- [ ] Manifest-driven loading
- [ ] A/B testing framework
- [ ] Old code removed
- [ ] Zero regressions

---

## Next Steps

1. **User Approval**: Review this audit
2. **Phase 1**: Create `WoWRollback.Viewer` project
3. **Incremental**: One plugin at a time
4. **Continuous**: Viewer works at every step

---

**Status**: ✅ Complete, awaiting approval  
**Confidence**: High  
**Estimated Phase 1**: 1 week
