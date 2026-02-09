# Active Context — MdxViewer Renderer Reimplementation

## Current Focus

**Terrain rendering, shadow blending, and MDX placement are now correct.** VLM dataset support added (load + generate). Async tile streaming implemented. Thread safety hardened. Next: verify all fixes visually, then move to MDX animation or particle systems.

## Immediate Next Steps (Next Session)

1. **Visual verification** — Load a map and confirm: MCSH shadows visible through overlay layers, MDX models correctly positioned (pivot offset fix), minimap working for both WDT and VLM.
2. **MDX doodad textures in WorldScene** — still untextured/magenta (pre-existing). Likely texture path resolution issue in WorldAssetManager.
3. **MDX animation** — Phase 1: keyframes, bones, geoset animation, playback UI.

## Session 2026-02-08 (Evening) Summary

### What Was Fixed

#### MCSH Shadow Blending (TerrainRenderer.cs)
- **Problem**: Shadow map (MCSH) was only applied on the base terrain layer. Alpha-blended overlay texture layers drawn on top would cover/wash out the shadows.
- **Root cause**: Both the C# render code and GLSL shader had `isBaseLayer` guards on shadow binding/application.
- **Fix**: Removed `isBaseLayer` condition from both:
  - C# `RenderChunkPass()`: Changed `bool hasShadow = isBaseLayer && chunk.ShadowTexture != 0` → `bool hasShadow = chunk.ShadowTexture != 0`
  - GLSL fragment shader: Changed `if (uShowShadowMap == 1 && uIsBaseLayer == 1 && uHasShadowMap == 1)` → `if (uShowShadowMap == 1 && uHasShadowMap == 1)`
- **Result**: Shadows now darken all texture layers consistently.

#### MDX Bounding Box Pivot Offset (WorldScene.cs, WorldAssetManager.cs)
- **Problem**: MDX model geometry is offset from origin (0,0,0). The MODL bounding box describes where geometry actually sits. MDDF placement position targets origin, but geometry center is elsewhere, causing models to appear displaced.
- **Fix**: Pre-translate geometry by negative bounding box center before scale/rotation/translation:
  - Added `WorldAssetManager.TryGetMdxPivotOffset()` — returns `(BoundsMin + BoundsMax) * 0.5f`
  - Transform chain: `pivotCorrection * mirrorX * scale * rotX * rotY * rotZ * translation`
  - `pivotCorrection = Matrix4x4.CreateTranslation(-pivot)`
  - Applied in both `BuildInstances()` and `OnTileLoaded()` in WorldScene.cs
- **WMO models**: Do NOT need pivot correction — their geometry is already correctly positioned relative to origin.

#### VLM Terrain Rendering (Previous session, 2026-02-08 afternoon)
- **GLSL shader em-dash**: Replaced unicode em-dash with ASCII hyphen in shader comment.
- **NullReferenceException**: Fixed null-conditional access in `DrawTerrainControls`.
- **VLM coordinate conversion**: Fixed `WorldPosition` in `VlmProjectLoader.cs` — swapped posX/posY, removed MapOrigin subtraction.
- **Minimap for VLM projects**: Refactored `DrawMinimap()` to work with either `_terrainManager` or `_vlmTerrainManager`. Added `IsTileLoaded()` to `VlmTerrainManager`.

#### Async Tile Streaming (TerrainManager.cs, VlmTerrainManager.cs)
- Both terrain managers now queue tile parsing to `ThreadPool` background threads.
- Parsed `TileLoadResult` objects enqueued to `ConcurrentQueue`.
- `SubmitPendingTiles()` runs on render thread each frame, uploading max 2 tiles/frame to avoid GPU stalls.
- `_disposed` flag prevents background threads from accessing disposed resources.

#### Thread Safety (VlmProjectLoader.cs, AlphaTerrainAdapter.cs, TerrainRenderer.cs)
- `TileTextures` → `ConcurrentDictionary` in both adapters.
- `_placementLock` protects dedup sets (`_seenMddfIds`, `_seenModfIds`) and placement lists in both adapters.
- `TerrainRenderer.AddChunks()` parameter widened from `Dictionary` to `IDictionary` to accept both.

#### VLM Dataset Generator (ViewerApp.cs)
- New menu item: `File > Generate VLM Dataset...`
- Dialog UI: client path (folder picker), map name, output dir, tile limit, progress log.
- Runs `VlmDatasetExporter.ExportMapAsync()` on `ThreadPool` with `IProgress<string>` feeding real-time log.
- "Open in Viewer" button after export completes.

### Key Technical Decisions
- **Coordinate system**: Renderer X = WoW Y, Renderer Y = WoW X, Z = height. MapOrigin = 17066.66666f, ChunkSize = 533.33333f.
- **MDX pivot**: Bounding box center, NOT PIVT chunk (PIVT is for per-bone skeletal animation pivots).
- **Shadow blending**: Apply to ALL layers, not just base. Overlay layers must also be darkened.
- **Thread safety**: `ConcurrentDictionary` for shared tile data, `lock` for placement dedup sets.

## What Works

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI loading | ✅ |
| Terrain MCSH shadow maps | ✅ (all layers, not just base) |
| Terrain alpha map debug view | ✅ (Show Alpha Masks toggle) |
| Async tile streaming | ✅ (background parse, render-thread GPU upload) |
| Standalone MDX rendering | ✅ (MirrorX, front-facing) |
| MDX pivot offset correction | ✅ (bounding box center pre-translation) |
| MDX doodads in WorldScene | ⚠️ Position fixed, textures still broken |
| WMO rendering + textures | ✅ (BLP per-batch) |
| WMO rotation/facing in WorldScene | ✅ |
| WMO doodad sets | ✅ |
| MDDF/MODF placements | ✅ (position + pivot correct) |
| Bounding boxes | ✅ (actual MODF extents) |
| VLM terrain loading | ✅ (JSON dataset → renderer) |
| VLM minimap | ✅ |
| VLM dataset generator | ✅ (File > Generate VLM Dataset) |
| Live minimap + click-to-teleport | ✅ (WDT + VLM) |
| AreaPOI system | ✅ |
| GLB export (Z-up → Y-up) | ✅ |
| Object picking/selection | ✅ |

## Key Files

- `Terrain/WorldScene.cs` — Object instance building, pivot offset, rotation transforms, rendering loop
- `Terrain/WorldAssetManager.cs` — Model loading, bounding box/pivot queries
- `Terrain/AlphaTerrainAdapter.cs` — MDDF/MODF parsing, coordinate conversion, thread-safe placement dedup
- `Terrain/VlmProjectLoader.cs` — VLM JSON tile loading, thread-safe TileTextures/placements
- `Terrain/VlmTerrainManager.cs` — VLM terrain AOI, async streaming
- `Terrain/TerrainManager.cs` — WDT terrain AOI, async streaming
- `Terrain/TerrainRenderer.cs` — Terrain shader, shadow maps on all layers, alpha maps, debug views
- `Rendering/WmoRenderer.cs` — WMO geometry, textures, doodad sets
- `Rendering/ModelRenderer.cs` — MDX rendering, MirrorX, blend modes, textures
- `ViewerApp.cs` — Main app, UI, DBC loading, minimap, VLM export dialog
- `Export/GlbExporter.cs` — GLB export with Z-up → Y-up conversion

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser, VLM dataset export
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access
