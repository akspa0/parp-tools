# Active Context — MdxViewer Renderer Reimplementation

## Current Focus

**MDX doodads in WorldScene have no textures.** WMOs are correctly positioned and rotated. Standalone MDX viewing works (MirrorX fix). Terrain alpha map seams improved with Noggit edge fix but may need further work.

## Immediate Next Steps (Next Session)

1. **FIX MDX doodad textures in WorldScene** — models render but appear untextured/magenta. May be a texture resolution path issue when loading MDX models via WorldAssetManager vs standalone.
2. **Verify terrain alpha seam fix** — Noggit edge fix applied, needs visual confirmation.
3. **Skybox sync** — MDX viewer skybox should match WDT viewer skybox system.

## Session 2026-02-08 Summary

### What Was Fixed
- **MDX standalone rendering**: Applied `MirrorX = Matrix4x4.CreateScale(-1, 1, 1)` via model matrix in `Render()` for left-handed → right-handed conversion. WorldScene uses `RenderWithTransform()` directly (no mirror).
- **BIDX parsing**: Simplified to always 1 byte per vertex (like GNDX). The 4-byte assumption corrupted the stream.
- **WMO WorldScene regression**: Reverted all WorldScene.cs and MTLS parser changes to stable commit `a1b0b41`. WMOs are back in correct position/rotation.
- **MTLS parser**: Reverted dual-format heuristic back to stable count-header-only format. The heuristic was misidentifying formats and breaking texture resolution.
- **Terrain alpha seams**: Applied Noggit edge fix (duplicate last row/col in 64×64 alpha data). Added "Show Alpha Masks" debug toggle.
- **Camera positioning**: Adjusted for MirrorX — camera at -X with Yaw=0 faces model front.

### What's Still Broken
- **MDX doodads in WorldScene** — no textures (magenta). This was the state at stable commit too, so it's a pre-existing issue, not a regression.
- **Terrain alpha seams** — Noggit fix applied but not yet visually confirmed as fully resolved.

### Key Technical Decisions
- **Coordinate system**: Vertex data stays in raw WoW coords. MirrorX only for standalone MDX `Render()`. WorldScene callers use `RenderWithTransform()` with WoW-space transforms.
- **BIDX = 1 byte** per vertex, not 4. Reading 4x corrupts MaterialId.
- **MTLS parser**: Count-header format only (stable). Dual-format heuristic was unreliable.
- **WMO Z rotation**: `-p.Rotation.Z` (negated for handedness). No +180° needed.

## What Works

| Feature | Status |
|---------|--------|
| Terrain rendering + AOI loading | ✅ |
| Terrain alpha map debug view | ✅ (Show Alpha Masks toggle) |
| Standalone MDX rendering | ✅ (MirrorX, front-facing) |
| MDX doodads in WorldScene | ⚠️ Position OK, textures broken |
| WMO rendering + textures | ✅ (BLP per-batch) |
| WMO rotation/facing in WorldScene | ✅ |
| WMO doodad sets | ✅ |
| MDDF/MODF placements | ✅ (position correct) |
| Bounding boxes | ✅ (actual MODF extents) |
| Live minimap + click-to-teleport | ✅ |
| AreaPOI system | ✅ |
| GLB export (Z-up → Y-up) | ✅ |
| Object picking/selection | ✅ |

## Key Files

- `Terrain/WorldScene.cs` — Object instance building, rotation transforms, rendering loop
- `Terrain/AlphaTerrainAdapter.cs` — MDDF/MODF parsing, coordinate conversion
- `Terrain/TerrainRenderer.cs` — Terrain shader, alpha maps, debug views
- `Rendering/WmoRenderer.cs` — WMO geometry, textures, doodad sets
- `Rendering/ModelRenderer.cs` — MDX rendering, MirrorX, blend modes, textures
- `ViewerApp.cs` — Main app, UI, DBC loading, minimap
- `Export/GlbExporter.cs` — GLB export with Z-up → Y-up conversion

## Dependencies (all already integrated)

- `MdxLTool` — MDX file parser
- `WoWMapConverter.Core` → `gillijimproject-csharp` — Alpha WDT/ADT/MCNK parsers, WMO v14 parser
- `SereniaBLPLib` — BLP texture loading
- `Silk.NET` — OpenGL + windowing + input
- `ImGuiNET` — UI overlay
- `DBCD` — DBC database access
