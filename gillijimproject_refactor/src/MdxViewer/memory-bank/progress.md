# Progress — AlphaWoW Viewer (MdxViewer)

## Status: v0.4.0 Released — 0.5.3 Rendering Improvements + 3.3.5 Groundwork (In Progress)

**Supported client versions: 0.5.3 through 0.12** — fully usable
**3.3.5 WotLK: IN PROGRESS** — scaffolding exists but MH2O liquid and terrain texturing are broken

## What Works Today

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI lazy loading | ✅ (AOI radius=3, 7×7 tiles, 4 uploads/frame) |
| Terrain MCSH shadow maps | ✅ Applied on ALL layers (not just base) |
| Terrain alpha map debug view | ✅ Show Alpha Masks toggle, Noggit edge fix |
| Terrain fog-based chunk culling | ✅ Skip chunks beyond FogEnd+200 |
| Terrain liquid rendering | ✅ Water/lava/slime (WMO MLIQ + terrain + MH2O) |
| **WotLK 3.3.5 terrain support** | 🔧 In progress — split ADT, MPHD flags parsed. **MH2O broken, texturing broken** |
| WDL parser (MVER/MAOF/MARE, v0x12) | ✅ Strict parsing + version validation |
| WDL terrain tile scale | ✅ Uses TileSize (8533.3333), not ChunkSize |
| WDL preview window reliability | ✅ Error reporting + `.wdl.mpq` fallback |
| WDL preview world spawn selection (Alpha 0.5.3) | ✅ Runtime-confirmed: preview orientation, clicked tile, and loaded-world spawn now agree |
| Later-client world map opening after preview UI change | ✅ Double-click map load restored to direct WDT open; unsupported WDL previews no longer block map loading |
| WDL runtime visibility toggle | ✅ UI checkbox for testing overlap issues |
| Async tile streaming | ✅ Background parse, render-thread GPU upload, max 2/frame |
| Standalone MDX rendering | ✅ MirrorX for LH→RH, front-facing, textured |
| **MDX skeletal animation** | ✅ Compressed quats, GPU skinning, standalone + terrain |
| MDX pivot offset correction | ✅ BB center pre-translation for correct placement |
| MDX blend modes + depth mask | ✅ Transparent layers don't write depth |
| MDX fog blending | ✅ Models blend into fog like terrain |
| MDX doodads in WorldScene | ✅ Position + animation + particles working |
| **MDX particle effects (PRE2)** | ✅ Billboard quads, texture atlas, bone-following, per-emitter blend |
| **Geoset animation alpha (ATSQ)** | ✅ Per-frame keyframe evaluation, alpha modulation |
| **M2 (MD20) model loading** | ✅ WarcraftNetM2Adapter: MD20→MdxFile conversion |
| **Half-Lambert lighting** | ✅ Softer shading on MDX + WMO (no harsh black shadows) |
| **Improved ambient lighting** | ✅ Day/night with WoW-like brightness levels |
| WMO v14 loading + rendering | ✅ Groups, BLP textures per-batch |
| WMO fog blending | ✅ WMOs blend into fog like terrain |
| WMO liquid rendering (MLIQ) | ✅ Semi-transparent water surfaces |
| WMO intermittent render race | ✅ Fixed via shared static shaders + ref counting |
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
| Direct Inspector access to minimap + world panels | ✅ Restored after dockable-panel regression |

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
| 1 | **MDX Animation** | ✅ Complete (compressed quats, GPU skinning, terrain doodads) |
| — | **Per-object folders + multi-angle screenshots** | 🔧 Next up |
| 2 | Particles (PRE2/RIBB) | ✅ PRE2 complete — billboard quads, texture atlas, blend modes. RIBB still pending. |
| WL | WL loose liquids transform alignment | 🔧 In progress — matrix tuning UI added, values not finalized |
| LK | **WotLK 3.3.5 Support** | 🔧 In progress — scaffolding exists. MH2O + texturing broken. **Not usable yet** |
| 5-7 | Liquids, Detail Doodads, Polish | ⏳ Lava type mapping still broken (green) |
| MCP | MCP Server | ⏳ Designed — GLB terrain, NPC spawn, click-to-chat, audio |

## 2026-02-15 — v0.4.0 Release: 0.5.3 Rendering Improvements + 3.3.5 Groundwork

**Rendering improvements for 0.5.3. Initial 3.3.5 scaffolding added (NOT ready for use).**

**3.3.5 WotLK support (IN PROGRESS — NOT USABLE):**
- StandardTerrainAdapter: split ADT loading, MPHD bigAlpha flag — but **MH2O broken, texturing broken**
- WarcraftNetM2Adapter: MD20→MdxFile conversion for M2 models (works in isolation)
- WMO v17: multi-MOTV/MOCV, strict validation
- Fixed terrain regression from initial 3.3.5 commit (surgical revert of shared renderer code)
- **Only client versions 0.5.3 through 0.12 are currently usable**

**Lighting overhaul:**
- Half-Lambert diffuse shading on MDX + WMO (wraps light, no harsh black shadows)
- Raised ambient: day 0.4→0.55, night 0.08→0.25 (WoW-like brightness)
- WMO shader: proper vec3 lighting instead of lossy scalar average
- Reduced specular intensity (0.3→0.15)

**Particle system (PRE2):**
- Rewrote ParticleRenderer with per-particle uniforms + texture atlas + per-emitter blend
- Wired into MdxRenderer: emitters created from parsed data, bone-following, transparent pass

**Geoset animation alpha (ATSQ):**
- Per-frame alpha keyframe evaluation with global sequence support
- Alpha modulates layer alpha in RenderGeosets; invisible geosets skipped

**Key files:** TerrainLighting.cs, ModelRenderer.cs, WmoRenderer.cs, ParticleRenderer.cs, StandardTerrainAdapter.cs, TerrainRenderer.cs, WarcraftNetM2Adapter.cs, WorldAssetManager.cs

## 2026-02-13 — MDX Animation System Complete

**Three bugs fixed:**
1. **KGRT Compressed Quaternions** — Rotation keys are 8-byte `C4QuaternionCompressed`, not 16-byte float4. Ghidra-verified decompression formula.
2. **Animation Never Updated** — `ViewerApp` called `RenderWithTransform()` directly, bypassing `Render()` which contained `Update()`. Extracted `UpdateAnimation()` as public method.
3. **PIVT Chunk Order** — PIVT comes after BONE in MDX files. All bone pivots were (0,0,0). Added deferred pivot assignment after all chunks parsed.

**Terrain animation** — Added `UpdateAnimation()` for unique MDX renderers in `WorldScene.cs` before render passes.

**Key files:** `MdxTypes.cs`, `MdxFile.cs`, `MdxAnimator.cs`, `ModelRenderer.cs`, `ViewerApp.cs`, `WorldScene.cs`

## 2026-02-13 — WDL/WL/WMO Rendering Fix Pass

**WDL parser + rendering fixes:**
1. `WdlParser` rewritten for strict chunk parsing and version check (`0x12`)
2. `MARE` chunk header handling corrected before reading 545 heights
3. WDL renderer now uses `WoWConstants.TileSize` (8533.3333)
4. WDL preview improved: `.wdl` / `.wdl.mpq` fallback + explicit failure reason
5. WDL tile overlap mitigation: hide preloaded ADT tiles + depth polygon offset
6. Added UI toggle to disable WDL rendering for overlap testing

## 2026-03-16 — WDL Preview Spawn/Orientation Fix Verified

- Fixed the remaining WDL preview click bug after the earlier out-of-bounds scale regression.
- Root cause was not camera reset during world load; it was mismatched preview orientation and preview-to-terrain tile conversion.
- `WdlPreviewDataBuilder` now builds the preview in the correct screen-major orientation, `WdlPreviewRenderer` converts preview tiles back into terrain tile coordinates before computing the spawn, and `WdlTerrainRenderer.HideTile/ShowTile` now matches the WDL load index convention.
- Runtime validation is complete for Alpha 0.5.3: clicking the preview now loads the world at the intended in-map location instead of empty sky / zero-tile space.

## 2026-03-16 — Later-Client Map Load Regression Fixed

- The new preview-first world map flow accidentally blocked normal map loading for later clients because double-clicking a map with a WDL tried to open the Alpha-only preview dialog first.
- Since the current `WdlParser` only handles the 0.5.3 WDL format, that produced preview parse failures for 0.6.0+ / 3.3.5 maps and prevented the expected WDT load flow.
- Active behavior now restores direct WDT loading on map double-click for all clients and only exposes the preview button on supported Alpha 0.5.3 data.

## 2026-03-17 — World Skybox M2s Routed To Backdrop Pass

- Later-client skybox assets such as `Environments/Stars/IceCrownCitadelSky.m2` were being treated as ordinary MDDF doodads, so they spawned inside the map and rendered in front of terrain with other transparent/reflective model layers.
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
