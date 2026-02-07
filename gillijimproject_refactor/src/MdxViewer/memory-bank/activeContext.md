# Active Context — MdxViewer Renderer Reimplementation

## Current Focus

Building a C# reimplementation of the WoW Alpha 0.5.3 renderer within the existing MdxViewer project. The full itemized plan lives in `renderer_plan.md`.

## Strategy: Terrain-First Reference Implementation

Build a working reference implementation of the engine in this order:
1. ~~Phase 0 — Foundation~~ ✅ COMPLETE
2. **Phase 3 — Terrain rendering** ← NEXT (most impactful, gets world geometry on screen)
3. **Phase 4 — World Scene** (compose terrain + existing model/WMO renderers, MDDF/MODF placements)
4. Phase 1 — MDX Animation (enhance existing model renderer)
5. Phase 2 — Particles
6. Phase 5-7 — Liquids, Detail Doodads, Polish

Goal is a working reference implementation first, then extend in our own direction.

## Phase 0 — Foundation ✅ COMPLETE

6 files created and building (0 errors):
- `Rendering/WoWConstants.cs` — Ghidra-verified constants
- `Rendering/BlendStateManager.cs` — 4 EGxBlend modes with GL state
- `Rendering/FrustumCuller.cs` — 6-plane VP extraction, point/sphere/AABB tests
- `Rendering/ShaderProgram.cs` — Compile/link/uniform-cache wrapper
- `Rendering/Material.cs` — BlendMode, texture, color, sorting
- `Rendering/RenderQueue.cs` — Opaque front-to-back + transparent back-to-front

## Phase 3 — Terrain ✅ COMPLETE (building, 0 errors)

7 files created:
- `Terrain/TerrainChunkData.cs` — GPU-ready data structures (heights, normals, layers, alpha maps)
- `Terrain/AlphaTerrainAdapter.cs` — Bridge WdtAlpha/AdtAlpha/McnkAlpha → TerrainChunkData
- `Terrain/TerrainMeshBuilder.cs` — 145 vertices → VAO/VBO/EBO per chunk (8 floats/vert: pos+norm+uv)
- `Terrain/TerrainRenderer.cs` — Multi-pass texture layering with alpha blending, BLP texture loading
- `Terrain/TerrainLighting.cs` — Day/night cycle, per-vertex Lambertian, linear fog
- `Terrain/TerrainManager.cs` — AOI-based chunk loading/unloading, implements ISceneRenderer
- Terrain shader embedded in TerrainRenderer (vertex: model/view/proj, fragment: multi-layer blend + lighting + fog)

ViewerApp integration:
- `.wdt` added to file browser filter
- `LoadWdtTerrain()` creates TerrainManager, loads all tiles, positions camera
- Render loop passes camera position for AOI updates and fog
- Model Info panel shows terrain controls: day/night slider, fog sliders, tile/chunk stats

Also added: `AdtAlpha.GetMcnkOffsets()` public accessor (was only used internally before)

## Key Decisions

- **Rendering API**: OpenGL 3.3 Core via Silk.NET (original client used DX9; we translate to GL equivalents)
- **Architecture**: Extend MdxViewer in-place rather than creating a separate project
- **Scene model**: Single `WorldScene : ISceneRenderer` composing terrain, models, WMOs, particles, liquids
- **Blend modes**: 4 modes from Ghidra — Opaque, Blend, Add, AlphaKey (exact GL state per mode)
- **Terrain parsing**: Reuse existing `gillijimproject-csharp` Alpha parsers (no new parsers needed)
- **Alpha vertex format**: Non-interleaved (81 outer then 64 inner); McvtAlpha.ToMcvt() handles reorder
- **Approach**: Reference implementation first, then optimize and extend

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access

## Resolved Questions

- ✅ ADT parser: Reuse `gillijimproject-csharp` (WdtAlpha, AdtAlpha, McnkAlpha) — already a transitive dependency
- Terrain loading: Start synchronous MVP, add async later in Phase 7
- Target map: Load from MPQ ("H:\053-client\Data\World\Maps\Kalidar\orig\Kalidar.wdt.MPQ")
