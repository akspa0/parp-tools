# Progress — MdxViewer

## Status: v0.4.0 Released — 0.5.3 usable, 3.3.5 NOT usable

## Working Features

| Category | Features |
|----------|----------|
| Terrain | Alpha WDT + 0.6.0 split ADT + MCSH shadows + alpha debug + fog culling |
| Streaming | AOI 9×9, directional lookahead, persistent cache, 4 uploads/frame |
| Liquid | MCLQ (terrain) + MLIQ (WMO) — water/lava/slime, waterfall slopes |
| MDX | Two-pass, animation (GPU skinning, compressed quats), PRE2 particles, geoset anim, specular, sphere env, M2/MD20 adapter |
| WMO | v14 4-pass, doodads (distance cull, no cap), shared static shaders, MLIQ |
| DBC | Lighting, area names, replaceable textures, taxi paths, POIs |
| WDL | Preview + spawn (0.5.3 only), later clients bypass to direct WDT |
| World | Full-load mode, frustum culling, minimap, object picking, skybox backdrop |
| VLM | Load + generate + minimap |
| Catalog | SQL dump → browse/filter → JSON+GLB+screenshot |
| Build | GitHub Actions CI, self-contained publish, WoWDBDefs bundling |

## 3.3.5 (Broken)
- Split ADT loading + MPHD flags: parsed
- MH2O: code-level fix, NOT runtime verified
- Terrain texturing: BROKEN at runtime
- Patch MPQ priority + BZip2: working

## Recent Fixes (Mar 16-17)
- WMO/MDX cached-null reload → retries on re-entry
- WMO doodad cap removed → distance-only culling
- World object residency → unlimited renderer cache
- WDL preview spawn + tile indexing → confirmed working (0.5.3)
- Later-client maps → direct WDT load (no WDL preview)
- World lifecycle → clean scene reset on switch
- Skybox M2 → backdrop pass; fog M2 → depth enabled
- MH2O → full instance parsing + exists bitmaps
- Minimap → restored after dockable-panel regression
- Inspector → direct access to world-data panels
- `WorldScene` now classifies known skybox asset families into a dedicated skybox instance list, excludes them from the normal doodad passes, and renders only the nearest active skybox as a camera-anchored backdrop before terrain.
- `MdxRenderer` now has a dedicated backdrop path that forces depth testing and depth writes off for every layer so sky models cannot stamp the depth buffer ahead of world geometry.
- Validation completed at build/test level: `MdxViewer.sln` builds cleanly and `MdxViewer.Tests` passes 19/19. Runtime visual validation is still required on real maps because this path depends on actual asset content and placement behavior.

## 2026-03-17 — Fog/Cloud M2 Terrain Overlap Root Cause Reduced

- The remaining "fog M2 renders over terrain" issue was not the new skybox backdrop path; it came from `WarcraftNetM2Adapter` inferring `NoDepthTest` / `NoDepthSet` from later-client M2 render flag bits.
- That inference was too aggressive for world fog/cloud materials and allowed placed M2s to bypass terrain depth even when they were ordinary doodads.
- Active behavior now keeps depth test/write enabled by default for Warcraft.NET-converted world M2s unless a dedicated source of truth for those flags is added later.
- Validation completed at build/test level: `MdxViewer.Tests` now passes 24/24 and `MdxViewer.sln` builds cleanly. Runtime visual validation is still required on a real affected map.

**WMO render reliability fix:**
- Converted WMO main shader and liquid shader to static shared programs with ref-counted lifetime.
- Fixes intermittent WMO disappearance caused by per-instance shader program deletion/reuse.

**WL liquids iteration tooling:**
- Replaced hardcoded axis swap with matrix-based configurable transform (rotation + translation).
- Added UI controls and hot-reload path (`Apply + Reload WL`) for rapid empirical alignment.
- Final transform values not yet locked/hard-wired.

## MDX Magenta Textures — RESOLVED (Particle System Implemented)

The magenta quads on MDX doodads were unimplemented particle emitter geometry (PRE2 chunks). Now resolved: ParticleRenderer rewritten, emitters wired into MdxRenderer, particles render with proper textures, atlas UV mapping, and per-emitter blend modes.

## Upcoming: MCP Server for LLM-Orchestrated 3D

New feature planned: turn MdxViewer into an MCP server that external applications (LLMs, procedural generators) can push content into at runtime.

### MCP Server Tools (planned)
| Tool | Description |
|------|-------------|
| `load_world` | Load GLB file as terrain (single mesh + texture) |
| `spawn_npc` | Place NPC at position with GLB model, texture, name, personality |
| `remove_npc` | Remove NPC by ID |
| `move_npc` | Move/animate NPC to new position |
| `play_audio` | Play sound file |
| `set_npc_dialog` | Set chat response for NPC |
| `get_scene_state` | Return camera pos, visible NPCs, selected NPC |
| `get_click_events` | Poll for NPC click events |

### Architecture
- MCP server runs on background thread (stdio transport)
- Commands queued and executed on render thread
- GLB → OpenGL via SharpGLTF (already in deps) + ImageSharp
- Entity system: `Dictionary<string, NpcEntity>` with position, mesh, click AABB
- Click detection via ray-AABB on mouse click
- Chat UI via ImGui overlay
- Phased: (1) server + GLB terrain + spawn, (2) click + chat, (3) audio + movement

## Detailed Fix Log

### 2026-02-09 Late Evening — Performance, Fog, Culling, Failed MDX Fix

**Completed (working):**
- Batched overlay rendering — POI pins + taxi paths in single draw call (BoundingBoxRenderer rewrite)
- AreaID/MapID-aware area name lookup — filters by current map, warns on mismatch
- Minimap zoom — 4 tiles around camera, scrollable
- TaxiPath visualization — DBC-loaded flight paths as 3D lines with selection
- POI + Taxi disabled by default
- NoCullRadius (150 units) — nearby objects skip frustum cull to prevent pop-in
- Fog added to MDX shader — models blend into fog like terrain
- Fog added to WMO shader — WMOs blend into fog like terrain
- Fog-based terrain chunk culling — skip chunks beyond FogEnd+200
- AOI radius reduced 3→2 (49→25 tiles)
- Doodad cull distance reduced 1500→1200, WMO cull distance 5000→2000
- README renamed to AlphaWoW Viewer

**Failed (reverted or ineffective):**
- `.blp.MPQ` scan — WRONG, these files don't exist. Reverted.
- ResolveReplaceableTexture 4-strategy rewrite — did not fix magenta textures
- Alpha test threshold change (0.3→0.1, conditional) — did not fix magenta textures

**Key files modified:**
- `BoundingBoxRenderer.cs` — Complete rewrite for batched rendering
- `WorldScene.cs` — Batched overlays, fog passing, reduced cull distances, NoCullRadius
- `TerrainManager.cs` — AOI radius 3→2
- `TerrainRenderer.cs` — Fog-based chunk distance culling
- `ModelRenderer.cs` — Fog shader, alpha test fix, ResolveReplaceableTexture rewrite
- `WmoRenderer.cs` — Fog shader
- `AreaTableService.cs` — MapID-aware lookup
- `ViewerApp.cs` — MapID tracking, area name display
- `MpqDataSource.cs` — Reverted bad .blp.MPQ change

### 2026-02-08 Evening — Shadow, Pivot, VLM Generator

**MCSH Shadow Blending Fix** (TerrainRenderer.cs):
- Shadow was only applied on base layer (`uIsBaseLayer == 1`). Overlay texture layers drawn on top with alpha blending would wash out shadows.
- Fix: Removed `isBaseLayer` guard from both C# shadow texture binding and GLSL shader shadow application.
- C#: `bool hasShadow = isBaseLayer && chunk.ShadowTexture != 0` → `bool hasShadow = chunk.ShadowTexture != 0`
- GLSL: `if (uShowShadowMap == 1 && uIsBaseLayer == 1 && uHasShadowMap == 1)` → `if (uShowShadowMap == 1 && uHasShadowMap == 1)`

**MDX Bounding Box Pivot Offset** (WorldScene.cs, WorldAssetManager.cs):
- MDX geometry is offset from origin. MODL bounding box center = effective pivot.
- Added `WorldAssetManager.TryGetMdxPivotOffset()` → `(BoundsMin + BoundsMax) * 0.5f`
- Transform chain: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
- `pivotCorrection = Matrix4x4.CreateTranslation(-pivot)`
- Applied in both `BuildInstances()` and `OnTileLoaded()`.
- WMO models do NOT need pivot correction.
- Note: PIVT chunk is for per-bone skeletal pivots, NOT placement pivot.

**VLM Dataset Generator** (ViewerApp.cs):
- Menu: `File > Generate VLM Dataset...`
- Uses `VlmDatasetExporter.ExportMapAsync()` from WoWMapConverter.Core on ThreadPool.
- Progress log via `IProgress<string>`, "Open in Viewer" button on completion.

### 2026-02-08 Afternoon — VLM Terrain, Async Streaming, Thread Safety

**VLM Terrain Rendering Fixes** (VlmProjectLoader.cs, TerrainRenderer.cs):
- GLSL shader em-dash → ASCII hyphen (compilation error).
- NullReferenceException in DrawTerrainControls → null-conditional access.
- VLM coordinate conversion: swapped posX/posY, removed MapOrigin subtraction.

**Minimap for VLM** (ViewerApp.cs, VlmTerrainManager.cs):
- `DrawMinimap()` refactored to support both `_terrainManager` and `_vlmTerrainManager`.
- Added `IsTileLoaded()` to VlmTerrainManager.

**Async Tile Streaming** (TerrainManager.cs, VlmTerrainManager.cs):
- `UpdateAOI()` queues tiles to ThreadPool for background JSON/ADT parsing.
- `ConcurrentQueue<TileLoadResult>` for render-thread submission.
- `SubmitPendingTiles()` uploads max 2 tiles/frame.
- `_disposed` flag prevents post-dispose background access.

**Thread Safety** (VlmProjectLoader.cs, AlphaTerrainAdapter.cs, TerrainRenderer.cs):
- `TileTextures` → `ConcurrentDictionary` in both adapters.
- `_placementLock` protects `_seenMddfIds`, `_seenModfIds`, placement lists.
- `CollectMddfPlacements()` and `CollectModfPlacements()` also locked.
- `TerrainRenderer.AddChunks()` parameter: `Dictionary` → `IDictionary`.

### 2026-02-08 Morning — MDX, BIDX, MTLS, GLB

- Fixed standalone MDX rendering — MirrorX model matrix for LH→RH conversion
- Fixed BIDX parsing — 1 byte per vertex (not 4)
- Reverted MTLS dual-format heuristic — back to stable count-header format
- Fixed WMO WorldScene regression — reverted to stable commit a1b0b41
- Fixed GLB export — Z-up → Y-up conversion for MDX and WMO
- Added terrain alpha mask debug view (Show Alpha Masks toggle)
- Applied Noggit edge fix for terrain alpha map seams

### 2026-02-07 — World Scene, Minimap, AreaPOI

- Phase 4 — World Scene: lazy tile loading, MDDF/MODF placements, object rendering
- Fixed MDX holes — disabled backface culling
- Fixed WMO textures — BLP loaded per-batch from materials
- Fixed MODF bounding boxes — actual extents with min/max swap
- Added live minimap with click-to-teleport
- Added AreaPOI system (DBC loading, 3D/minimap markers, UI list)
- Fixed MDX blend modes — depth mask off for transparent layers, alpha discard 0.1
- WMO doodads now rendered with WMO's modelMatrix

### 2026-02-06 — Foundation + Terrain

- Phase 0 COMPLETE — 6 foundation files
- Phase 3 COMPLETE — 7 terrain files + ViewerApp integration
