# WoW Alpha 0.5.3 Renderer — Itemized Implementation Plan

## Goal

Reimplement the WoW Alpha 0.5.3 rendering pipeline in C# so we can render **all** asset types (MDX models, WMO buildings, ADT terrain, particles, liquids, detail doodads) with faithful visual parity to the original client, using the existing MdxViewer as the host application.

## Reference Material

All Ghidra-verified addresses and pseudocode live in two documentation sets:

| Set | Path | Covers |
|-----|------|--------|
| Deep Dive | `documentation/WoW_Alpha_Deep_Dive/` | MDX animation, particles, BLP textures, shader/blend system, model pipeline |
| World Rendering | `documentation/World_Rendering_Analysis/` | Terrain, collision, movement, mesh traversal, liquids, detail doodads, frustum culling, lighting, texture layering, chunk rendering |

## Existing Code Inventory (what we already have)

| Component | File(s) | Status |
|-----------|---------|--------|
| MDX parser | `MdxLTool.Formats.Mdx.MdxFile` | ✅ Complete — loads all chunks |
| WMO v14 parser | `WoWMapConverter.Core.Converters.WmoV14ToV17Converter` | ✅ Complete — groups, materials, doodads, portals |
| BLP2 loader | `SereniaBLPLib` | ✅ Complete — DXT1/3/5, palette, JPEG |
| MDX renderer | `Rendering/ModelRenderer.cs` | ✅ Per-geoset, multi-layer materials, textured |
| WMO renderer | `Rendering/WmoRenderer.cs` | ✅ Groups, doodad sets, textured |
| Camera | `Rendering/Camera.cs` | ✅ Free-fly, WASD, mouse look |
| ISceneRenderer | `Rendering/ISceneRenderer.cs` | ✅ Interface with sub-object visibility |
| MPQ data source | `DataSources/MpqDataSource.cs` | ✅ Listfile, nested WMO MPQs |
| DBC integration | `DataSources/MpqDBCProvider.cs` | ✅ DBCD, CreatureModelData, CreatureDisplayInfo |
| Replaceable textures | `Rendering/ReplaceableTextureResolver.cs` | ✅ Per-model DBC lookup |
| GLB export | `Export/GlbExporter.cs` | ✅ MDX + WMO |
| ImGui UI | `ViewerApp.cs` | ✅ File browser, model info, doodad sets, visibility |

---

## Phase 0 — Foundation & Shared Systems

Refactor shared infrastructure out of the two existing renderers before adding new systems.

### 0.1 WoWConstants.cs
- [ ] Create `Rendering/WoWConstants.cs`
- [ ] Terrain constants: `CHUNK_SCALE`, `CHUNK_OFFSET`, `MAX_CHUNK`, `CHUNK_SIZE`, `CELL_SIZE`, `NUM_CELLS`, `NUM_VERTICES`
- [ ] Movement constants: `WALK_SPEED`, `RUN_SPEED`, `SWIM_SPEED`, `JUMP_VELOCITY`, `GRAVITY`
- [ ] Liquid constants: `LIQUID_WATER`, `LIQUID_OCEAN`, `LIQUID_MAGMA`, `LIQUID_SLIME`, `LIQUID_NONE`
- [ ] Detail doodad constants: `DETAIL_DOODAD_DISTANCE`, `MAX_DETAIL_DOODADS`
- [ ] AOI constants: `GROUP_AOI_SIZE`, `OBJECT_AOI_SIZE`

### 0.2 BlendStateManager
- [ ] Create `Rendering/BlendStateManager.cs`
- [ ] Implement 4 blend modes from Ghidra (`SetMaterialBlendMode @ 0x00448cb0`):
  - `Opaque` — no blend, depth write ON
  - `Blend` — SrcAlpha/InvSrcAlpha, depth write OFF
  - `Add` — SrcAlpha/One, depth write OFF
  - `AlphaKey` — SrcAlpha/InvSrcAlpha + alpha test >0.5, depth write ON
- [ ] Refactor `MdxRenderer` and `WmoRenderer` to use shared BlendStateManager

### 0.3 Material System
- [ ] Create `Rendering/Material.cs`
- [ ] Fields: BlendMode, DiffuseTexture, EmissiveColor, DiffuseColor, Opacity, TwoSided, DepthWrite
- [ ] `Apply()` method sets GL blend + texture + cull state

### 0.4 RenderQueue + Transparency Sorting
- [ ] Create `Rendering/RenderQueue.cs`
- [ ] Opaque items: sorted front-to-back (early-Z optimization)
- [ ] Transparent items: sorted back-to-front (correct alpha compositing)
- [ ] Per-frame clear and re-sort

### 0.5 FrustumCuller
- [ ] Create `Rendering/FrustumCuller.cs`
- [ ] Extract 6 planes from view-projection matrix (per Ghidra `07_Frustum_Culling.md`)
- [ ] `TestPoint()`, `TestSphere()`, `TestAABB()` methods
- [ ] Normalize planes after extraction

### 0.6 Shader Abstraction
- [ ] Create `Rendering/ShaderProgram.cs` — compile, link, uniform cache
- [ ] Extract duplicated shader code from MdxRenderer + WmoRenderer into shared module
- [ ] Shared vertex format: position, normal, texcoord, color

---

## Phase 1 — MDX Animation System

Based on `01_MDX_Animation_System.md`. The MDX parser already loads animation chunks; this phase adds runtime playback.

### 1.1 Keyframe Track System
- [ ] Create `Animation/KeyframeTrack.cs`
- [ ] Generic `IKeyframeTrack<T>` with `Evaluate(float time)` 
- [ ] 4 interpolation types:
  - Linear (`InterpolateLinear @ 0x0075de20`)
  - Hermite (`InterpolateHermite @ 0x0075dc00`)
  - Bezier (`InterpolateBezier @ 0x0075de00`)
  - Constant (step/hold)
- [ ] Binary search for surrounding keyframes

### 1.2 Quaternion Compression
- [ ] Create `Animation/QuaternionCompression.cs`
- [ ] Decompress 30-bit packed format (10 bits per X/Y/Z, W = sqrt(1-sum²))
- [ ] Squad interpolation for smooth rotation curves

### 1.3 Bone Transform Hierarchy
- [ ] Create `Animation/BoneSystem.cs`
- [ ] Evaluate position/rotation/scale tracks per bone
- [ ] Parent-child matrix composition (world = parent.world × local)
- [ ] Inverse bind pose for GPU skinning

### 1.4 Geoset Animation
- [ ] Wire up KGAC (color) and KGAL (alpha) tracks from parser
- [ ] Per-geoset visibility, alpha, and C3Color modulation
- [ ] Apply to renderer: multiply vertex colors by animated values

### 1.5 Material & Texture Animation
- [ ] Animate material properties (alpha, color per layer)
- [ ] UV scrolling via TXAN texture animation tracks
- [ ] Global sequence support (animations not tied to a specific sequence)

### 1.6 Animation Playback Controller
- [ ] Create `Animation/AnimationPlayer.cs`
- [ ] Sequence selection, play/pause, loop, speed control
- [ ] Time accumulation with delta-time
- [ ] Animation blending for transitions (`AnimEnableBlending @ 0x00741590`)

### 1.7 Animation UI
- [ ] ImGui panel: sequence selector dropdown
- [ ] Timeline scrubber / progress bar
- [ ] Play/pause/speed controls
- [ ] Bone count and track info display

---

## Phase 2 — MDX Particle System

Based on `02_Particle_System.md`. Adds visual effects attached to model bones.

### 2.1 Particle Emitter
- [ ] Create `Particles/ParticleEmitter.cs`
- [ ] 4 emitter types: Base, Plane, Sphere, Spline
- [ ] 3 particle types: Quad (billboard), Vertex, UpVertex
- [ ] Alive/dead pool system (swap-remove for O(1) recycle)

### 2.2 Particle Physics
- [ ] Velocity, acceleration, gravity integration
- [ ] Wind vector application
- [ ] Follow-model offset
- [ ] Lifetime countdown (0→1 normalized)

### 2.3 Particle Keyframe Tracks
- [ ] KPEM (emission rate), KLIF (lifetime), KVEL (velocity)
- [ ] KGRA (gravity), KSCALE (scale), KCOL (color), KALP (alpha), KROT (rotation)
- [ ] Evaluate from emitter's animation time

### 2.4 Texture Sheet Animation
- [ ] Row/column UV subdivision
- [ ] Frame time, looping
- [ ] Twinkle effect (on/off, scale min/max)

### 2.5 Particle Rendering
- [ ] Billboard quad generation (camera-facing)
- [ ] Per-particle color, alpha, size, rotation
- [ ] Blend mode from ParticleMaterial (Opaque/Blend/Add/AlphaKey)
- [ ] Batch rendering with shared VBO

---

## Phase 3 — Terrain System (ADT/WDT)

Based on `01_Terrain_System.md`, `09_Texture_Layering.md`, `08_Terrain_Lighting.md`.

**EXISTING PARSERS** (via `gillijimproject-csharp`, already a transitive dependency):
- `GillijimProject.WowFiles.Alpha.WdtAlpha` — parses MPHD, MAIN (64×64 tile presence + MHDR offsets), MDNM, MONM
- `GillijimProject.WowFiles.Alpha.AdtAlpha` — parses MHDR, MCIN, MTEX, MDDF, MODF per tile
- `GillijimProject.WowFiles.Alpha.McnkAlpha` — full MCNK with all subchunks:
  - `McvtAlpha` — 145 height floats (non-interleaved: 81 outer then 64 inner; has `ToMcvt()` reorder)
  - `McnrAlpha` — 145 normals × 3 bytes (non-interleaved; has `ToMcnrLk()` reorder)
  - MCLY — texture layers (texture ID, flags, alpha offset)
  - MCAL — alpha maps for blending
  - MCSH — shadow map
  - MCLQ — liquid data
  - MCRF — M2/WMO reference indices
  - Holes mask from header

**What we need to BUILD** is the rendering bridge — extracting parsed data into GPU-ready buffers.

### 3.1 AlphaTerrainAdapter
- [ ] Create `Terrain/AlphaTerrainAdapter.cs`
- [ ] Wraps `WdtAlpha` + `AdtAlpha` + `McnkAlpha` for the renderer
- [ ] Expose typed accessors: `GetHeights()` → float[145], `GetNormals()` → Vector3[145]
- [ ] Handle Alpha non-interleaved → interleaved vertex reorder for mesh gen
- [ ] Parse MCLY entries into `TerrainLayer[]` (textureIndex, flags, alphaOffset)
- [ ] Extract alpha maps from MCAL as byte arrays
- [ ] Read MDDF/MODF placements into structured lists

### 3.4 Terrain Mesh Generation
- [ ] Build vertex buffer: 145 vertices per chunk with position, normal, texcoord
- [ ] Build index buffer: 128 triangles per chunk (8×8 cells × 2 tris)
- [ ] Hole mask → skip triangles for holed cells
- [ ] Upload to GPU (VAO/VBO/EBO per chunk)

### 3.5 Texture Layering
- [ ] Multi-pass or multi-texture terrain rendering
- [ ] Alpha map upload as texture
- [ ] 5 blend modes: Opaque(4), Alpha(0), Add(1), Modulate(2), Modulate2X(3)
- [ ] Base layer (opaque) + up to 3 alpha-blended layers

### 3.6 Terrain Lighting
- [ ] Create `Terrain/TerrainLighting.cs`
- [ ] Day/night cycle: gameTime [0,1) → sun angle
- [ ] Light color, ambient color, fog color per time-of-day band
- [ ] Per-vertex Lambertian diffuse: `max(0, dot(normal, lightDir))`
- [ ] Vertex color = ambient + diffuse × factor

### 3.7 TerrainManager (Area of Interest)
- [ ] Create `Terrain/TerrainManager.cs`
- [ ] Camera position → chunk coordinates conversion
- [ ] AOI rectangle calculation + clamping to [0, 1023]
- [ ] Load/unload chunks as camera moves
- [ ] Dictionary<(int,int), TerrainChunk> for loaded chunks

### 3.8 Chunk Rendering
- [ ] Camera-relative transform matrix (avoid floating-point jitter)
- [ ] Sort chunks by distance
- [ ] Render opaque terrain, then transparent layers
- [ ] Per-chunk frustum cull before draw

---

## Phase 4 — World Scene Integration

Composes all systems into a unified world renderer.

### 4.1 WorldScene Class
- [ ] Create `Rendering/WorldScene.cs` implementing `ISceneRenderer`
- [ ] Owns: TerrainManager, FrustumCuller, RenderQueue, TerrainLighting, FogSystem
- [ ] Main render loop:
  1. Update frustum from camera VP matrix
  2. Update terrain AOI
  3. Collect visible terrain chunks
  4. Collect visible WMO placements (from MODF)
  5. Collect visible MDX doodad placements (from MDDF)
  6. Sort render queue
  7. Render opaque pass (terrain → WMOs → doodads)
  8. Render transparent pass (back-to-front)

### 4.2 MDDF Doodad Placement
- [ ] Parse MDDF entries from ADT: model path, position, rotation, scale
- [ ] Instance MDX models at world positions
- [ ] Frustum cull per-doodad bounding sphere
- [ ] Share MDX model data across instances (model cache)

### 4.3 MODF WMO Placement
- [ ] Parse MODF entries from ADT: WMO path, position, rotation, bounds
- [ ] Instance WMO renderers at world positions
- [ ] Frustum cull per-WMO bounding box
- [ ] Share WMO data across instances (WMO cache)

### 4.4 Fog System
- [ ] Create `Rendering/FogSystem.cs`
- [ ] Linear fog: `(end - depth) / (end - start)` (from `ComputeFogBlend @ 0x00689b40`)
- [ ] Exponential fog: `exp(-density × depth)`
- [ ] Fog color from day/night cycle
- [ ] Apply via shader uniform

### 4.5 Day/Night Cycle
- [ ] Game time slider in UI (0.0 midnight → 0.5 noon → 1.0 midnight)
- [ ] Drives: sun position, light color, ambient, fog color
- [ ] Smooth interpolation between time bands (night/dawn/day/dusk/night)

### 4.6 ViewerApp Integration
- [ ] Add "Open WDT/ADT" to File menu
- [ ] Route terrain loading to WorldScene renderer
- [ ] Camera positioned at world center or first populated chunk
- [ ] UI panel: day/night slider, terrain stats, chunk count

---

## Phase 5 — Liquid Rendering

Based on `05_Liquid_Rendering.md`.

### 5.1 Liquid Mesh
- [ ] Create `Terrain/LiquidRenderer.cs`
- [ ] Parse MCLQ from MCNK chunks
- [ ] Build liquid surface mesh (height grid + type)
- [ ] 4 liquid types: Water, Ocean, Magma, Slime

### 5.2 Liquid Rendering
- [ ] Animated UV scrolling for water/magma textures
- [ ] Alpha blending for water transparency
- [ ] Type-specific color tinting
- [ ] Render after opaque terrain, before transparent objects

### 5.3 Liquid Detection
- [ ] `QueryLiquidStatus(position)` → type + level + direction
- [ ] WMO interior liquid support (local space query)
- [ ] Underwater camera detection (tint + fog change)

---

## Phase 6 — Detail Doodads

Based on `06_Detail_Doodads.md`.

### 6.1 Detail Doodad Generator
- [ ] Create `Terrain/DetailDoodadManager.cs`
- [ ] Per-chunk seeded random placement (up to 64 doodads)
- [ ] Terrain height sampling for Z placement
- [ ] Random rotation and scale variation

### 6.2 Detail Doodad Rendering
- [ ] Distance-based visibility (100 unit cutoff)
- [ ] Billboard or instanced rendering
- [ ] Alpha fade-out at distance boundary

---

## Phase 7 — Polish & Optimization

### 7.1 Performance
- [ ] GPU instancing for repeated doodads
- [ ] Texture atlas or array for terrain layers
- [ ] LOD system for distant terrain/models
- [ ] Async chunk loading on background thread

### 7.2 Debug Overlays
- [ ] Wireframe terrain toggle
- [ ] Chunk boundary lines
- [ ] Normal visualization
- [ ] Frustum visualization
- [ ] Bounding box/sphere display

### 7.3 Additional Features
- [ ] WDL low-res heightmap for distant terrain
- [ ] Skybox rendering
- [ ] Collision detection visualization
- [ ] Screenshot/video capture

---

## Implementation Order & Dependencies

```
Phase 0 (Foundation)
  ├── 0.1 WoWConstants ← no deps
  ├── 0.2 BlendStateManager ← no deps
  ├── 0.3 Material ← 0.2
  ├── 0.4 RenderQueue ← 0.3
  ├── 0.5 FrustumCuller ← no deps
  └── 0.6 ShaderProgram ← no deps

Phase 1 (Animation) ← Phase 0
  ├── 1.1 KeyframeTrack ← no deps
  ├── 1.2 QuaternionCompression ← no deps
  ├── 1.3 BoneSystem ← 1.1, 1.2
  ├── 1.4 GeosetAnimation ← 1.1
  ├── 1.5 MaterialAnimation ← 1.1
  ├── 1.6 AnimationPlayer ← 1.1-1.5
  └── 1.7 AnimationUI ← 1.6

Phase 2 (Particles) ← Phase 0, Phase 1.1
  ├── 2.1 ParticleEmitter ← 1.1
  ├── 2.2 ParticlePhysics ← 2.1
  ├── 2.3 ParticleKeyframes ← 1.1, 2.1
  ├── 2.4 TextureSheet ← 2.1
  └── 2.5 ParticleRendering ← 0.2, 2.1-2.4

Phase 3 (Terrain) ← Phase 0
  ├── 3.1 AlphaTerrainAdapter ← existing gillijimproject-csharp parsers
  ├── 3.4 TerrainMesh ← 3.1, 0.6
  ├── 3.5 TextureLayering ← 3.1, 0.2
  ├── 3.6 TerrainLighting ← no deps
  ├── 3.7 TerrainManager ← 3.1, 3.4
  └── 3.8 ChunkRendering ← 3.4-3.7, 0.5

Phase 4 (World Scene) ← Phase 0-3
  ├── 4.1 WorldScene ← 0.4, 0.5, 3.7
  ├── 4.2 MDDF Placement ← 4.1, existing MdxRenderer
  ├── 4.3 MODF Placement ← 4.1, existing WmoRenderer
  ├── 4.4 FogSystem ← 0.6
  ├── 4.5 DayNightCycle ← 3.6, 4.4
  └── 4.6 ViewerApp Integration ← 4.1-4.5

Phase 5 (Liquids) ← Phase 3
Phase 6 (Detail Doodads) ← Phase 3
Phase 7 (Polish) ← Phase 4-6
```

## Estimated Effort

| Phase | Items | Estimate |
|-------|-------|----------|
| Phase 0 — Foundation | 6 tasks | 2-3 days |
| Phase 1 — Animation | 7 tasks | 4-5 days |
| Phase 2 — Particles | 5 tasks | 3-4 days |
| Phase 3 — Terrain | 8 tasks | 5-7 days |
| Phase 4 — World Scene | 6 tasks | 3-4 days |
| Phase 5 — Liquids | 3 tasks | 2-3 days |
| Phase 6 — Detail Doodads | 2 tasks | 1-2 days |
| Phase 7 — Polish | 3 tasks | 2-3 days |
| **Total** | **40 tasks** | **~22-31 days** |

## Success Criteria

- [ ] MDX models render with correct animations (bone transforms, geoset vis, color)
- [ ] Particles emit, animate, and billboard correctly
- [ ] Terrain loads from ADT with correct heightmap geometry
- [ ] Terrain textures layer correctly with alpha blending
- [ ] WMO and doodad placements appear at correct world positions
- [ ] Day/night cycle drives lighting, ambient, and fog
- [ ] Liquid surfaces render over terrain with type-specific effects
- [ ] Frustum culling keeps frame rate usable for a full map tile
- [ ] Camera can fly through the world freely
