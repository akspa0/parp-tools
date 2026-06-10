# WoWRollback 3D Viewer Vision - "Google Earth for WoW Alpha"

## Executive Summary

Transform WoWRollback from a 2D tile-based comparison tool into a comprehensive 3D digital archaeology platform for World of Warcraft Alpha data.

**Vision**: Users can fly through Alpha Azeroth in 3D, toggle overlays showing terrain properties, see actual object placements with real models, compare versions side-by-side, and export data for further analysis.

---

## Current State (2D Foundation)

### What Works ✅
- **2D Leaflet-based viewer** with tile overlays
- **Object placement data** (M2/WMO positions, rotations, scales)
- **Terrain property extraction** (flags, liquids, holes, AreaIDs)
- **Version comparison** (diff visualization, timeline tracking)
- **CSV/JSON export** for all data
- **Minimap rendering** from BLP tiles
- **Alpha→LK ADT conversion** (coordinate system working)

### What's Broken ❌
- **AreaTable mapping** (shows "Unknown Area" instead of names)
- **Shadow maps** (data extracted but no UI layer)
- **Area boundaries** (recently fixed but needs validation)
- **Performance** (large maps slow to generate)

### What's Missing for 3D 🚧
- **3D terrain mesh generation** from heightmap data
- **3D object models** (MDX/M2, WMO loading)
- **Texture support** (BLP with alpha channels)
- **3D camera/navigation** (fly-through, zoom, pan)
- **LOD system** (level-of-detail for performance)
- **Frustum culling** (only render visible tiles)
- **Lazy loading** (load tiles as user navigates)
- **3D overlay rendering** (terrain properties in 3D)

---

## Integration Opportunities

### Existing Projects to Leverage

#### 1. **ADTPrefabTool** (`lib/ADTPrefabTool.poc/`)
**Purpose**: Original C# terrain parsing attempt  
**Assets**:
- ADT chunk parsing logic
- Heightmap extraction
- Terrain mesh generation ideas
- Coordinate system conversions

**Integration Plan**:
- Extract reusable terrain parsing code
- Merge into WoWRollback.Core as `Services/Terrain/`
- Use for 3D heightmap → mesh conversion

#### 2. **wow-mdx-viewer** (`lib/wow-mdx-viewer/`)
**Purpose**: Alpha MDX model rendering  
**Assets**:
- MDX file format parsing
- Bone animation support
- Vertex/texture handling
- WebGL rendering pipeline

**Integration Plan**:
- Port MDX parser to C# for backend processing
- Keep WebGL renderer for frontend
- Generate model placement manifests (JSON)
- Load models on-demand in 3D viewer

#### 3. **wow.export** (`lib/wow.export/src/`)
**Purpose**: Texture extraction and conversion  
**Assets**:
- BLP format parsing (all versions)
- Alpha channel handling
- Mipmapping support
- Texture atlas generation

**Integration Plan**:
- Extract BLP decoder as standalone library
- Use for texture extraction in backend
- Generate WebGL-compatible textures (PNG with alpha)
- Build texture cache/CDN for viewer

#### 4. **gillijimproject (C++ original)**
**Purpose**: Original Alpha data parser  
**Assets**:
- Proven parsing logic
- Edge case handling
- Performance optimizations

**Integration Plan**:
- Reference implementation for validation
- Port critical algorithms to C#
- Use as testing baseline

---

## Architecture Vision

### Three-Layer System

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (Browser)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  2D Viewer   │  │  3D Viewer   │  │  Comparison  │  │
│  │  (Leaflet)   │  │ (Three.js)   │  │   (Split)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ JSON/GLB/Textures
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Middleware (Asset Server)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Tile Cache  │  │  Model CDN   │  │  Texture CDN │  │
│  │   (JSON)     │  │   (GLB)      │  │   (PNG)      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ▲
                          │ File System
                          ▼
┌─────────────────────────────────────────────────────────┐
│             Backend (WoWRollback.Cli)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  ADT Parser  │  │  MDX Parser  │  │  BLP Parser  │  │
│  │              │  │              │  │              │  │
│  │ • Heightmap  │  │ • Models     │  │ • Textures   │  │
│  │ • Chunks     │  │ • Animations │  │ • Mipmaps    │  │
│  │ • Liquids    │  │ • Bones      │  │ • Atlases    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  WMO Parser  │  │  DBC Parser  │  │  Exporter    │  │
│  │              │  │              │  │              │  │
│  │ • Groups     │  │ • AreaTable  │  │ • GLB        │  │
│  │ • Portals    │  │ • Maps       │  │ • OBJ        │  │
│  │ • Doodads    │  │ • Lookup     │  │ • JSON       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Project Structure

```
WoWRollback/
├── WoWRollback.Core/
│   ├── Parsers/                 # File format parsers
│   │   ├── Adt/                 # ADT terrain (Alpha + LK)
│   │   ├── Wdt/                 # WDT map index
│   │   ├── Mdx/                 # M2/MDX models (NEW)
│   │   ├── Wmo/                 # WMO objects (NEW)
│   │   ├── Blp/                 # BLP textures (NEW)
│   │   └── Dbc/                 # DBC databases
│   ├── Services/
│   │   ├── Terrain/             # Terrain processing
│   │   │   ├── HeightmapExtractor
│   │   │   ├── ChunkProcessor
│   │   │   └── MeshGenerator    # (NEW - 3D mesh from heightmap)
│   │   ├── Models/              # (NEW - Model processing)
│   │   │   ├── MdxLoader
│   │   │   ├── WmoLoader
│   │   │   └── PlacementResolver
│   │   ├── Textures/            # (NEW - Texture processing)
│   │   │   ├── BlpDecoder
│   │   │   ├── AlphaExtractor
│   │   │   └── AtlasBuilder
│   │   ├── Export/              # Data export
│   │   │   ├── GlbExporter      # (NEW - glTF binary)
│   │   │   ├── ObjExporter
│   │   │   └── JsonExporter
│   │   └── Viewer/              # Viewer asset generation
│   │       ├── TileBuilder2D    # Existing 2D overlays
│   │       └── TileBuilder3D    # (NEW - 3D tile manifests)
│   └── Models/                  # Data models
│       ├── Adt/
│       ├── Mdx/                 # (NEW)
│       ├── Wmo/                 # (NEW)
│       └── Viewer/
│           ├── Tile2DOverlay
│           └── Tile3DManifest   # (NEW)
├── WoWRollback.Cli/
│   └── Commands/
│       ├── AnalyzeAlphaWdt      # Existing
│       ├── CompareVersions      # Existing
│       ├── GenerateViewer3D     # (NEW - 3D viewer assets)
│       └── ExportScene          # (NEW - GLB/OBJ export)
├── ViewerAssets/
│   ├── 2d/                      # Existing Leaflet viewer
│   │   ├── index.html
│   │   ├── js/
│   │   └── css/
│   └── 3d/                      # (NEW - Three.js viewer)
│       ├── index.html
│       ├── js/
│       │   ├── core/            # Three.js setup
│       │   ├── loaders/         # GLTF, texture loading
│       │   ├── terrain/         # Terrain mesh rendering
│       │   ├── models/          # MDX/WMO rendering
│       │   ├── overlays/        # 3D overlay rendering
│       │   └── controls/        # Camera, navigation
│       └── shaders/             # Custom GLSL shaders
│           ├── terrain.vert
│           ├── terrain.frag
│           ├── liquid.vert
│           └── liquid.frag
└── docs/
    └── planning/
        ├── 03-3d-viewer-vision.md      # This document
        ├── 04-3d-implementation.md     # (Next - detailed plan)
        └── 05-integration-strategy.md  # (Next - merging projects)
```

---

## Phased Implementation Plan

### Phase 0: Foundation Stabilization (CURRENT)
**Goal**: Fix all 2D issues, establish solid base  
**Duration**: 1-2 weeks

#### Tasks
- [x] Fix terrain overlay path mismatch
- [x] Fix area boundary hide() bug
- [ ] Fix AreaTable mapping (use LK AreaIDs)
- [ ] Implement shadow map overlay
- [ ] Performance optimization (caching, parallel processing)
- [ ] Complete all 2D overlay types
- [ ] Comprehensive testing
- [ ] Documentation cleanup

**Deliverable**: Rock-solid 2D viewer with all features working

---

### Phase 1: 3D Terrain Foundation (2-3 weeks)
**Goal**: Render basic terrain mesh in 3D

#### Backend Tasks
1. **Heightmap Extraction**
   - Port heightmap logic from ADTPrefabTool
   - Extract MCVT (vertex heights) from MCNK
   - Generate per-chunk heightmaps
   - Export as binary arrays or images

2. **Mesh Generation**
   - Create triangle mesh from 9×9 + 8×8 vertices
   - Handle inner/outer vertices correctly
   - Apply textures coordinates
   - Export as GLB (glTF binary)

3. **Tile Manifest**
   - Generate JSON manifest per tile
   - Include chunk meshes, heights, normals
   - Link to texture files
   - Add metadata (bounds, LOD levels)

#### Frontend Tasks
1. **Three.js Setup**
   - Initialize WebGL renderer
   - Camera controls (orbit, fly-through)
   - Basic lighting
   - Skybox/background

2. **Terrain Loader**
   - Load GLB terrain meshes
   - Frustum culling
   - LOD switching
   - Tile streaming (load visible tiles only)

3. **Basic Texturing**
   - Load minimap as placeholder texture
   - Apply to terrain mesh
   - Basic UV mapping

**Deliverable**: Flyable 3D terrain with minimap textures

---

### Phase 2: Texture System (2-3 weeks)
**Goal**: Real BLP textures with alpha blending

#### Backend Tasks
1. **BLP Parser Integration**
   - Port wow.export BLP decoder
   - Support all BLP formats (BLP0, BLP1, BLP2)
   - Extract mipmaps
   - Handle compressed formats (DXT1, DXT3, DXT5)

2. **Alpha Layer Support**
   - Extract MCAL (alpha maps) from MCNK
   - Multiple texture layers per chunk
   - Alpha blending weights
   - Export as PNG with alpha channel

3. **Texture Atlas**
   - Pack multiple textures into atlases
   - Generate UV coordinates
   - Optimize for WebGL (power-of-2 sizes)
   - Mipmap generation

#### Frontend Tasks
1. **Multi-Texture Shader**
   - Custom GLSL shader for terrain
   - Support 4+ texture layers per chunk
   - Alpha blending
   - Normal mapping (if available)

2. **Texture Streaming**
   - Lazy load textures
   - Texture cache/LRU eviction
   - Low-res → high-res progressive loading
   - Compressed texture formats (WebGL extensions)

**Deliverable**: Terrain with real game textures

---

### Phase 3: Object Placement (3-4 weeks)
**Goal**: Show M2/WMO placements in 3D

#### Backend Tasks
1. **MDX/M2 Parser**
   - Port wow-mdx-viewer parser to C#
   - Extract vertices, normals, UVs
   - Parse bone structure (basic)
   - Skip animations initially
   - Export as GLB

2. **WMO Parser**
   - Parse WMO root + groups
   - Extract geometry
   - Handle portals (basic)
   - Doodad sets
   - Export as GLB

3. **Placement Manifest**
   - Per-tile object placement JSON
   - Include position, rotation, scale
   - Link to model GLB files
   - UniqueID for diff tracking

#### Frontend Tasks
1. **Model Loader**
   - Load GLB models on-demand
   - Instance rendering for duplicates
   - Transform to world space
   - Frustum culling

2. **Object Picking**
   - Raycasting for object selection
   - Show object metadata on click
   - Highlight selected objects
   - Filter by type (M2/WMO)

3. **Model Texturing**
   - Load BLP textures for models
   - Apply to GLB materials
   - Handle transparency
   - Team colors (if applicable)

**Deliverable**: Full scene with placed objects

---

### Phase 4: Advanced Features (4-6 weeks)
**Goal**: Polish, overlays, animations, comparison

#### 3D Overlays
- **Terrain Properties** - Color code chunks by flags
- **Liquids** - Animated water/lava planes
- **Area Boundaries** - 3D walls or colored zones
- **Shadow Maps** - Baked shadow textures
- **Heightmap Visualization** - False color by elevation

#### Animations
- **Model Animations** - Basic M2 bone animation
- **Liquid Animation** - Wave simulation
- **Camera Paths** - Recorded fly-throughs
- **Timeline Scrubbing** - Animate version changes

#### Version Comparison
- **Split View** - Side-by-side 3D viewers
- **Overlay Mode** - Ghost objects from other version
- **Diff Highlighting** - Red = removed, Green = added
- **Timeline Slider** - Scrub through versions

#### Performance
- **Worker Threads** - Background loading
- **WebAssembly** - Critical path optimizations
- **Compressed Transfers** - gzip, brotli
- **IndexedDB Cache** - Persistent client-side cache

**Deliverable**: Full-featured 3D viewer

---

### Phase 5: Export & Tooling (2-3 weeks)
**Goal**: Export capabilities, modding support

#### Export Formats
- **Full Scene GLB** - Entire map as single file
- **Per-Tile GLB** - Modular exports
- **OBJ + MTL** - For Blender/3DS Max
- **FBX** - For game engines
- **Collision Data** - For physics simulation

#### Modding Tools
- **Object Editor** - Move, rotate, scale objects
- **Terrain Editor** - Modify heightmaps (basic)
- **Texture Swapper** - Replace textures
- **Export to ADT** - Write modified ADTs (advanced)

**Deliverable**: Export pipeline for modders

---

## Technical Challenges & Solutions

### Challenge 1: Performance - Large Datasets
**Problem**: Azeroth has thousands of tiles, millions of triangles

**Solutions**:
- **Frustum Culling** - Only render visible tiles
- **LOD System** - Multiple detail levels per tile
- **Instancing** - Reuse models (trees, rocks)
- **Occlusion Culling** - Skip hidden geometry
- **Tile Streaming** - Load/unload dynamically
- **WebWorkers** - Parse/process in background
- **WebAssembly** - Critical decoders in WASM

### Challenge 2: Memory - Browser Limits
**Problem**: Browsers have ~2GB memory limit

**Solutions**:
- **Lazy Loading** - Load only visible area
- **LRU Cache** - Evict old tiles
- **Compressed Textures** - DXT formats via WebGL
- **Shared Geometries** - Instance duplicate models
- **Streaming Buffers** - Progressive mesh loading

### Challenge 3: Coordinate Systems
**Problem**: Alpha vs LK, game vs world, tile vs chunk

**Solutions**:
- **Single Source of Truth** - Use LK coordinates everywhere
- **Conversion Layer** - Centralized coordinate transforms
- **Validation** - Test known landmarks (Goldshire, Orgrimmar)
- **Visualization** - Debug overlays for coordinate grids

### Challenge 4: Texture Quality
**Problem**: Alpha BLPs may be low-res or missing

**Solutions**:
- **Fallback Textures** - Checkerboard for missing
- **Upscaling** - AI upscaling for low-res (optional)
- **Placeholder Diffuse** - Vertex colors as backup
- **Minimap Projection** - Use minimap as low-res texture

### Challenge 5: Model Complexity
**Problem**: WMO models are complex (groups, portals, doodads)

**Solutions**:
- **Simplified Models** - Low-poly proxies for distance
- **Progressive Loading** - Load detail as camera approaches
- **Portal Culling** - Respect WMO portal visibility
- **Doodad Sets** - Toggle sets on/off

---

## Technology Stack

### Backend (C# .NET 9)
- **WoWRollback.Core** - Core parsing/processing
- **System.Numerics** - Vector math
- **SixLabors.ImageSharp** - Image processing
- **SharpGLTF** - GLB export
- **Warcraft.NET** - WoW file formats (extended)

### Frontend (Web)
- **Three.js** - 3D rendering engine
- **Leaflet** - 2D map (keep existing)
- **dat.GUI** - Control panel
- **stats.js** - Performance monitoring
- **Vanilla JS/ES6** - No framework overhead

### Shaders (GLSL)
- **Custom terrain shader** - Multi-texture blending
- **Liquid shader** - Animated water/lava
- **Overlay shader** - Color-coded properties

### Build/Deploy
- **Vite** - Fast dev server, bundler
- **TypeScript** - Type safety for JS
- **WebAssembly** - Critical path optimizations

---

## Data Flow Example

### 3D Tile Loading Sequence

1. **User navigates camera** in 3D viewer
2. **Frustum culling** determines visible tiles
3. **Frontend requests** tile manifest: `/tiles/3d/0.5.3.3368/Azeroth/tile_r30_c30.json`
4. **Middleware serves** cached JSON or generates on-demand
5. **Manifest contains**:
   ```json
   {
     "map": "Azeroth",
     "version": "0.5.3.3368",
     "row": 30,
     "col": 30,
     "terrain": "/tiles/3d/.../terrain.glb",
     "textures": ["/textures/.../grass.png", ...],
     "objects": [
       {
         "model": "/models/tree01.glb",
         "position": [1234.5, 567.8, 90.1],
         "rotation": [0, 45, 0],
         "scale": [1.2, 1.2, 1.2],
         "uniqueId": 12345
       },
       ...
     ],
     "overlays": {
       "terrainProperties": "/overlays/.../terrain_props.json",
       "liquids": "/overlays/.../liquids.json"
     }
   }
   ```
6. **Frontend loads** terrain GLB via Three.js GLTFLoader
7. **Frontend loads** textures and applies to mesh
8. **Frontend loads** object models and instances them
9. **Render** complete tile
10. **Repeat** for all visible tiles

---

## Milestones & Timeline

### Immediate (Next 2 Weeks)
- ✅ Fix all 2D bugs (AreaTable, shadow maps)
- ✅ Complete 2D feature set
- ✅ Stabilize existing codebase
- ✅ Write comprehensive tests

### Short Term (Months 1-2)
- 🎯 Phase 1: 3D Terrain Foundation
- 🎯 Phase 2: Texture System
- 📋 Integration with ADTPrefabTool code

### Medium Term (Months 3-4)
- 🎯 Phase 3: Object Placement
- 📋 Integration with wow-mdx-viewer
- 📋 Integration with wow.export

### Long Term (Months 5-6)
- 🎯 Phase 4: Advanced Features
- 🎯 Phase 5: Export & Tooling
- 🚀 Public Release

---

## Success Criteria

### Phase 1 Success
- [ ] Load and render Elwynn Forest in 3D
- [ ] Smooth camera navigation (60 FPS)
- [ ] Visible terrain features (hills, valleys, water)
- [ ] Performance: 10+ tiles loaded simultaneously

### Phase 2 Success
- [ ] Real textures applied to terrain
- [ ] Grass, dirt, stone textures visible
- [ ] Alpha blending between textures
- [ ] No texture popping on load

### Phase 3 Success
- [ ] Goldshire visible with buildings
- [ ] Trees, rocks, fences in correct positions
- [ ] Click object to see metadata
- [ ] No Z-fighting or culling issues

### Phase 4 Success
- [ ] Toggle terrain overlays in 3D
- [ ] Compare 0.5.3 vs 0.5.5 side-by-side
- [ ] Smooth 60 FPS with full scene
- [ ] Works on mid-range hardware

### Phase 5 Success
- [ ] Export Elwynn to GLB (opens in Blender)
- [ ] Export maintains textures and UVs
- [ ] File size reasonable (<500MB per zone)
- [ ] Round-trip: export → modify → re-import

---

## Risk Assessment

### High Risk
- **Performance in browser** - May need native app fallback
- **Memory limits** - Aggressive streaming required
- **Model complexity** - Simplified proxies may be needed

### Medium Risk
- **Coordinate system bugs** - Extensive testing required
- **Texture quality** - Alpha assets may be incomplete
- **Animation support** - Complex, may defer to Phase 6

### Low Risk
- **Basic terrain rendering** - Well-understood problem
- **Object placement** - Already have position data
- **Export formats** - Standard libraries available

---

## Next Steps

### For This Session
1. ✅ Create this vision document
2. ⏳ Review and refine scope
3. ⏳ Prioritize Phase 0 tasks
4. ⏳ Create detailed Phase 1 implementation plan

### For Next Session
1. 📋 Create `04-3d-implementation.md` - Technical deep-dive
2. 📋 Create `05-integration-strategy.md` - Merging existing projects
3. 📋 Start Phase 0 bug fixes (AreaTable, shadow maps)
4. 📋 Set up project structure for 3D modules

### For Future Sessions
- Begin Phase 1 terrain implementation
- Test integration with ADTPrefabTool
- Prototype Three.js viewer
- Benchmark performance targets

---

## Questions to Answer

1. **Viewer Mode**: Should 2D and 3D be separate apps or unified with mode toggle?
2. **Hosting**: Client-side only or need server for tile generation?
3. **Target Hardware**: What's minimum spec? (affects LOD strategy)
4. **Export Priority**: Which formats are most important?
5. **Animation Scope**: Skip entirely or basic support?
6. **Texture Resolution**: What quality targets? (affects file size)
7. **Object Detail**: Render all props or filter by size?

---

## Conclusion

This is a **major expansion** from 2D comparison tool to full 3D platform. The vision is achievable by:

1. ✅ **Stabilizing 2D first** (1-2 weeks)
2. 🎯 **Incremental 3D rollout** (Phases 1-5)
3. 🔧 **Leveraging existing projects** (ADTPrefabTool, wow-mdx-viewer, wow.export)
4. 📐 **Modular architecture** (can deploy 2D without 3D)
5. 🎨 **Progressive enhancement** (works at each phase)

**Recommendation**: Complete Phase 0 (2D stabilization) first, then reassess scope and resources for 3D phases.

Ready to build the ultimate WoW Alpha digital archaeology platform! 🌍✨
