# WebWowViewerCpp Integration - Future 3D Viewer

**Date**: 2025-10-08  
**Status**: 📋 PLANNED - Future Enhancement  
**Priority**: Low (after 2D plugin system is complete)

---

## Overview

**WebWowViewerCpp** is a full-featured 3D viewer for World of Warcraft maps and models, available at `lib/WebWowViewerCpp`. This could serve as the foundation for our 2.5D/3D viewer instead of building from scratch.

### What It Is
- **Full 3D WoW viewer** (C++ with WebGL frontend)
- **Production-ready** - Used by wow.tools for their model viewer (https://wow.tools/mv/)
- **Complete rendering pipeline** - Handles maps, models (M2/WMO), terrain, liquids
- **CASC integration** - Can read directly from game files
- **Both standalone and WebGL** versions available

### Repository
- **Location**: `lib/WebWowViewerCpp`
- **GitHub**: https://github.com/Deamon87/WebWowViewerCpp
- **License**: Check repository for licensing details
- **Active Development**: Yes (GitHub Actions CI/CD)

---

## Current Architecture (2D Plugin System)

Our current implementation:
```
2D Leaflet Map
├── CoordinateSystem.js (WoW coordinates)
├── PluginManager (overlay management)
├── GridPlugin (ADT grid)
├── M2Plugin (planned - doodad markers)
├── WMOPlugin (planned - WMO markers)
└── Minimap tiles (WebP images)
```

**Strengths:**
- ✅ Lightweight and fast
- ✅ Works in any browser
- ✅ Easy to implement overlays
- ✅ Good for analysis and comparison

**Limitations:**
- ❌ No elevation visualization
- ❌ No 3D model rendering
- ❌ Limited spatial understanding
- ❌ Can't see terrain context

---

## Proposed Integration Architecture

### Phase 1: 2D Plugin System (Current) ✅
**Timeline**: Week 1-2  
**Status**: In Progress

Continue building the 2D Leaflet-based viewer with plugin system.

### Phase 2: WebWowViewerCpp Investigation (Future)
**Timeline**: Week 3-4  
**Status**: Planned

Investigate WebWowViewerCpp integration:
1. Build the WebGL version
2. Understand their coordinate system
3. Test embedding in our viewer
4. Evaluate performance and compatibility

### Phase 3: Hybrid 2D/3D Viewer (Future)
**Timeline**: Month 2+  
**Status**: Concept

Integrate both systems:

```
┌─────────────────────────────────────────┐
│  WoW Rollback Viewer                    │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────┐  ┌──────────────────┐ │
│  │   2D View   │  │    3D View       │ │
│  │  (Leaflet)  │  │ (WebWowViewerCpp)│ │
│  │             │  │                  │ │
│  │ • Minimap   │  │ • Terrain mesh   │ │
│  │ • Overlays  │  │ • M2/WMO models  │ │
│  │ • Analysis  │  │ • Camera control │ │
│  └─────────────┘  └──────────────────┘ │
│         ▲                  ▲            │
│         └──────┬───────────┘            │
│                │                        │
│      ┌─────────▼────────┐               │
│      │ Unified Controls │               │
│      │  • Sync position │               │
│      │  • Sync overlays │               │
│      │  • Toggle view   │               │
│      └──────────────────┘               │
└─────────────────────────────────────────┘
```

---

## Integration Benefits

### For Users
- **Better spatial understanding** - See elevation and terrain context
- **3D model preview** - View actual M2/WMO models, not just markers
- **Immersive exploration** - Fly through maps like in-game
- **Side-by-side comparison** - 2D analysis + 3D visualization

### For Development
- **Proven technology** - Don't reinvent 3D rendering
- **Active maintenance** - Used by wow.tools community
- **Complete feature set** - Terrain, models, liquids, shadows
- **WebGL optimized** - Performance-tested on real data

---

## Technical Considerations

### Coordinate System Compatibility

**Our System** (`CoordinateSystem.js`):
```javascript
TILE_SIZE = 533.33333 yards
MAP_HALF_SIZE = 17066.66656 yards
Origin: (0,0) at map center
```

**WebWowViewerCpp**:
- Need to investigate their coordinate system
- Likely uses WoW's native coordinate space
- May need coordinate transform layer

### Data Pipeline

**Current (2D)**:
```
WoW Files → C# Exporter → JSON/WebP → Leaflet
```

**With WebWowViewerCpp (3D)**:
```
WoW Files → CASC → WebWowViewerCpp → WebGL
```

**Hybrid**:
```
WoW Files ─┬→ C# Exporter → JSON/WebP → Leaflet (2D)
           └→ CASC → WebWowViewerCpp → WebGL (3D)
```

### Performance Considerations

**2D Viewer**:
- Lightweight (WebP tiles + SVG overlays)
- Fast loading
- Low memory usage

**3D Viewer**:
- Heavy (terrain meshes + textures + models)
- Slower initial load
- Higher memory usage
- GPU-dependent

**Solution**: Lazy loading
- Start with 2D view (fast)
- Load 3D on demand (when user switches view)
- Unload 3D when not in use

---

## Implementation Plan (Future)

### Step 1: Build WebWowViewerCpp
```bash
cd lib/WebWowViewerCpp
git submodule update --init --recursive
cmake -B build -G Ninja
cmake --build build
```

### Step 2: Test WebGL Version
- Build emscripten port
- Test in browser
- Verify CASC loading works
- Test with our data

### Step 3: Create Integration Layer
```javascript
// File: js/viewers/Viewer3D.js
import { WebWowViewer } from '../../lib/WebWowViewerCpp/emscripten_port/wowviewer.js';

export class Viewer3D {
    constructor(container, coordSystem) {
        this.coords = coordSystem;
        this.viewer = new WebWowViewer(container);
    }
    
    // Sync with 2D viewer
    syncPosition(worldX, worldY, worldZ) {
        // Convert our coordinates to WebWowViewerCpp space
        const viewerCoords = this.coords.toViewerSpace(worldX, worldY, worldZ);
        this.viewer.setCamera(viewerCoords);
    }
    
    // Load map
    async loadMap(mapName, cascPath) {
        await this.viewer.openCasc(cascPath);
        await this.viewer.loadMap(mapName);
    }
}
```

### Step 4: Add View Toggle
```html
<div class="view-controls">
    <button id="view2D" class="active">2D Map</button>
    <button id="view3D">3D View</button>
    <button id="viewSplit">Split View</button>
</div>
```

### Step 5: Sync State
- Position sync (2D ↔ 3D)
- Overlay visibility (show same objects in both views)
- Selection sync (click in 2D, highlight in 3D)

---

## Alternative: 2.5D Isometric View

If full 3D integration is too complex, consider a **2.5D isometric view** as a middle ground:

### Using WDL Terrain Meshes
We already have `WdlGltfExporter.cs` that exports low-res terrain:
- Load WDL terrain as GLB
- Render with Three.js (simpler than WebWowViewerCpp)
- Fixed isometric camera (no free flight)
- Overlay minimap textures on terrain
- Plot M2/WMO positions as markers

**Benefits:**
- Simpler than full 3D
- Still shows elevation
- Lighter weight
- Easier to integrate

**See**: `08-overlay-plugin-system.md` (lines 910-1040) for 2.5D architecture

---

## Decision Matrix

| Feature | 2D Only | 2.5D Isometric | Full 3D (WebWowViewerCpp) |
|---------|---------|----------------|---------------------------|
| **Complexity** | Low | Medium | High |
| **Performance** | Excellent | Good | Heavy |
| **Elevation** | No | Yes | Yes |
| **Models** | Markers only | Markers only | Full rendering |
| **Development Time** | 1-2 weeks | 3-4 weeks | 6-8 weeks |
| **Browser Support** | 100% | 95% (WebGL) | 95% (WebGL) |
| **Memory Usage** | Low | Medium | High |

---

## Recommendation

**Current Phase**: Focus on 2D plugin system
- Complete GridPlugin, M2Plugin, WMOPlugin
- Get full 2D functionality working
- Build solid foundation

**Next Phase**: Evaluate 2.5D vs Full 3D
- Test WebWowViewerCpp integration feasibility
- Compare with simpler 2.5D approach
- Make decision based on:
  - User needs
  - Development resources
  - Performance requirements

**Timeline**:
- **Now - Week 2**: 2D plugin system
- **Week 3-4**: Evaluation and prototyping
- **Month 2+**: 3D integration (if chosen)

---

## Resources

### WebWowViewerCpp
- **Repo**: https://github.com/Deamon87/WebWowViewerCpp
- **Live Demo**: https://wow.tools/mv/
- **Local Path**: `lib/WebWowViewerCpp`

### Our Exporters
- **WdlGltfExporter**: Exports WDL terrain as GLB
- **ADTPreFabTool**: Exports ADT with minimap textures
- **MinimapComposer**: Generates WebP minimap tiles

### Related Docs
- `08-overlay-plugin-system.md` - Current plugin architecture
- `PLUGIN_SYSTEM_STATUS.md` - Implementation status
- `WEBP_MIGRATION.md` - Performance optimizations

---

## Next Steps

1. ✅ **Document this discovery** (this file)
2. ✅ **Continue with 2D plugin system** (current priority)
3. ⏳ **Complete M2Plugin and WMOPlugin**
4. ⏳ **Test 2D viewer thoroughly**
5. ⏳ **Evaluate 3D integration** (after 2D is complete)

---

**Status**: Documented for future reference. Continuing with 2D plugin system as planned.
