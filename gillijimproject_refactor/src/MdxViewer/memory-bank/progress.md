# Progress — AlphaWoW Viewer (MdxViewer)

## Status: Asset Catalog Enhancement + Animation/Lighting Next

## What Works Today

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI lazy loading | ✅ (AOI radius=3, 7×7 tiles, 4 uploads/frame) |
| Terrain MCSH shadow maps | ✅ Applied on ALL layers (not just base) |
| Terrain alpha map debug view | ✅ Show Alpha Masks toggle, Noggit edge fix |
| Terrain fog-based chunk culling | ✅ Skip chunks beyond FogEnd+200 |
| Terrain liquid rendering | ✅ Water/lava/slime (WMO MLIQ + terrain) |
| Async tile streaming | ✅ Background parse, render-thread GPU upload, max 2/frame |
| Standalone MDX rendering | ✅ MirrorX for LH→RH, front-facing, textured |
| MDX pivot offset correction | ✅ BB center pre-translation for correct placement |
| MDX blend modes + depth mask | ✅ Transparent layers don't write depth |
| MDX fog blending | ✅ Models blend into fog like terrain |
| MDX doodads in WorldScene | ⚠️ Position correct, magenta = unimplemented particles (PRE2/RIBB) |
| WMO v14 loading + rendering | ✅ Groups, BLP textures per-batch |
| WMO fog blending | ✅ WMOs blend into fog like terrain |
| WMO liquid rendering (MLIQ) | ✅ Semi-transparent water surfaces |
| WMO doodad sets | ✅ Loaded and rendered with WMO modelMatrix |
| WMO rotation/facing in WorldScene | ✅ Fixed — `-rz` negation for handedness |
| MDDF/MODF placements | ✅ Position + pivot correct |
| Bounding boxes | ✅ Actual MODF extents with correct min/max swap |
| Batched overlay rendering | ✅ POI pins + taxi paths in single draw call |
| Minimap zoom (4 tiles around camera) | ✅ |
| TaxiPath visualization | ✅ DBC-loaded flight paths as 3D lines |
| Taxi path selection (sidebar) | ✅ |
| POI + Taxi lazy-load UI | ✅ Load buttons → toggle checkboxes after load |
| AreaID/MapID-aware area names | ✅ Filters by current map, warns on mismatch |
| NoCullRadius (150 units) | ✅ Nearby objects skip frustum cull |
| VLM terrain loading | ✅ JSON dataset → renderer |
| VLM minimap | ✅ Works for VLM projects |
| VLM dataset generator | ✅ File > Generate VLM Dataset (background export) |
| BLP2 texture loading | ✅ DXT1/3/5, palette, JPEG |
| MPQ data source | ✅ Listfile, nested WMO/WDT/WDL archives |
| DBC integration | ✅ DBCD, CreatureModelData, CreatureDisplayInfo, TaxiPath, AreaPOI, AreaTable |
| Camera | ✅ Free-fly WASD + mouse look |
| ImGui UI | ✅ File browser, model info, visibility toggles |
| Live minimap + click-to-teleport | ✅ WDT + VLM |
| AreaPOI system | ✅ DBC loading, 3D markers, minimap markers, UI list |
| Object picking/selection | ✅ |
| GLB export | ✅ MDX + WMO, Z-up → Y-up conversion |
| Thread safety | ✅ ConcurrentDictionary for TileTextures, locks for placement dedup |
| **Asset Catalog** | ✅ SQL dump parser (no MySQL), browse/search/filter, JSON+GLB+screenshot export |
| **Loading screen** | ✅ BLP-based with progress bar |

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Foundation | ✅ Complete |
| 3 | Terrain | ✅ Complete (shadow fix, alpha seam fix, async streaming, fog culling) |
| 4 | World Scene | ✅ WMOs, MDX placement, rotation. Particles deferred. |
| VLM | VLM Dataset Support | ✅ Load + Generate + Minimap |
| Overlays | POI, Taxi, Minimap Zoom | ✅ Complete (batched rendering, lazy-load UI) |
| Loading | Loading Screen | ✅ Complete |
| Catalog | Asset Catalog | ✅ SQL dump reader, ImGui browse/filter, JSON+GLB+screenshot export |
| — | **Per-object folders + multi-angle screenshots** | 🔧 Next up |
| 1 | MDX Animation | ⏳ Not started |
| 2 | Particles (PRE2/RIBB) | ⏳ Not started — causes magenta on some MDX geosets |
| 5-7 | Liquids, Detail Doodads, Polish | ⏳ Lava type mapping still broken (green) |
| MCP | MCP Server | ⏳ Designed — GLB terrain, NPC spawn, click-to-chat, audio |

## MDX Magenta Textures — DEFERRED (Root Cause: Particles)

The magenta quads on MDX doodads are **unimplemented particle emitter geometry** (PRE2/RIBB chunks). These are separate geosets that reference particle textures. Regular model textures load fine. This will be fixed when the particle system is implemented (Phase 2).

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
