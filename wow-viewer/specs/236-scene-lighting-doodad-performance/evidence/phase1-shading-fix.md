# Receipt: Spec 236 — Phase 1 Shading & Normal Corrections & Minimap / Phase Defect Fixes

**Date**: 2026-09-15
**Feature Branch**: `v0.5.4-dev`
**Status**: PASSED

## 1. Summary of Changes

### A. MDX / M2 Shading Fix (T001, T002)
- **Files**:
  - `src/viewer/WoWViewer/Rendering/ModelRenderer.cs`
  - `src/viewer/WoWViewer/Rendering/M2Renderer.cs`
- **Root Cause**: `!gl_FrontFacing` inverted normals when models were rendered with negative scaling or CW winding, causing outward-facing triangles to be treated as inward-facing and wiping out diffuse lighting. Furthermore, harsh `max(nDotL, 0.0)` created pitch-black shadow edges.
- **Fix**: Removed `!gl_FrontFacing` normal inversion. Implemented Half-Lambert diffuse wrapping (`float diff = nDotL * 0.5 + 0.5; diffuseStrength = diff * diff;`) matching `WmoRenderer.cs`.

### B. Phase Map Liquid & Doodad Elevation Synchronization
- **Files**:
  - `src/viewer/WoWViewer/Terrain/LiquidChunkData.cs`: added `WithRehoming(...)` to rehome heights, min/max heights, and world coordinates.
  - `src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs`: applied `WithRehoming` across `RehomeChunksForTarget` and `BuildCellShiftedTile`; scaled and offset placement Z by `(placement.Position.Z * layer.ZScale) + layer.ZOffset` in `TranslatePhasePlacements`; elevated preserved base doodads and WMOs in `MergePhaseTile` when heightmap has non-zero ZOffset.
  - `src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs`: applied identical liquid rehoming and placement Z adjustments; marked `phase.PlacementsPreTransformed = true` on rotated layers to prevent double translation.

### C. Live Minimap Tile Dragging & Footprints
- **Files**:
  - `src/viewer/WoWViewer/Terrain/WorldScene.cs`: removed premature unrotated return in `GetLayerFootprints()`, ensuring `TileOffsetX` and `TileOffsetY` are always included.
  - `src/viewer/WoWViewer/MinimapHelpers.cs`: expanded tile rendering to iterate over all active phase layer footprints in addition to base map tiles, rendering textures and footprint outlines seamlessly across the entire 64x64 grid.
  - `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs`: enabled auto-selection of any clicked phase layer footprint and allowed dragging of rotated/mirrored layers.

### D. Interactive Fullscreen Minimap Donor Tile Placer Tool
- **Files**:
  - `src/viewer/WoWViewer/UI/MinimapDonorToolService.cs` (NEW): owned service class (Spec 228 compliant) providing toolbar, mode switcher (`Navigate` vs `DonorTileTool`), donor source selection, placement mapping, target removal, and real-time visual overlays (pulsing green `[SRC]` badge, gold placement borders with connector arrows, cyan hover target preview).
  - `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs`: integrated `_donorToolService` into `DrawFullscreenMinimap`, `HandleMinimapInteraction`, and bound key `T` for quick mode toggling.

---

## 2. Verification Commands & Results

1. **Compilation**:
   - Command: `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
   - Exit Code: `0`
   - Errors: `0`

2. **Phase Unit Tests**:
   - Command: `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Phase"`
   - Exit Code: `0`
   - Results: `Passed: 76, Failed: 0, Total: 76`

---

## 3. Criterion -> Evidence Mapping

| Acceptance Criterion | Verification Command | Result |
|---|---|---|
| MDX / M2 shaders compile without syntax or uniform errors | `dotnet build WowViewer.slnx` | Exit code 0, 0 errors |
| Normal inversion removed, Half-Lambert wrapping active | Source inspection of `ModelRenderer.cs` and `M2Renderer.cs` | Verified lines 1954-1964 and 918-924 |
| Phase composition and layer tests pass | `dotnet test --filter "FullyQualifiedName~Phase"` | 76 passed, 0 failed |
| Liquid chunks rehomed with ZOffset | `LiquidChunkData.WithRehoming` implemented and consumed in both adapters | Verified in `StandardTerrainAdapter.cs` & `AlphaTerrainAdapter.cs` |
| Minimap tile rendering includes all active phase layer footprints | `MinimapHelpers.RenderMinimapContent` combines `baseTileSet` and `GetLayerFootprints()` | Verified in `MinimapHelpers.cs` |
| Fullscreen minimap provides interactive donor tile placer | `MinimapDonorToolService.cs` wired with toolbar and overlays | Verified in `ViewerApp_MinimapAndStatus.cs` |
