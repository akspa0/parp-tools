# Phase 3 Verification Receipt: Mesh Renderer & Buffer Decomposition

- **Spec**: Spec 288 (`specs/288-pm4-overlay-refactor-and-color-coding/spec.md`)
- **Date**: 2026-11-01
- **Phase**: Phase 3 (Mesh Renderer & Buffer Decomposition)
- **Status**: PASS

---

## 1. Files Changed & Created

- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs` (Created):
  - Extracted PM4 mesh rendering, bounds, gizmos, and nodes out of `WorldScene.cs` and `Pm4OverlayScene.cs`.
  - Implemented per-tile GPU geometry caching (`TileMeshGpuBuffer`) with dedicated VAOs/VBOs for lines and solid triangles.
  - Compiles dedicated OpenGL shaders with layout qualifiers and uniforms for model-view-projection (`uVP`).
  - Implemented state fingerprinting (`ComputeTileFingerprint`) to invalidate only tiles whose transform, color mode, or selection state changed.
  - Provided thread-safe queued GL buffer disposal (`_vaosToDelete`, `_vbosToDelete`).
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayColors.cs` (Modified):
  - Centralized PM4 color constants (`Pm4ColorObjectBounds`, `Pm4ColorCk24Bounds`, `Pm4ColorSelectedBounds`, `Pm4ColorMddfBounds`, `Pm4ColorModfBounds`, `Pm4ColorCentroid`, `Pm4ColorMscn`, `Pm4ColorMspv`, `Pm4ColorMprl`, `Pm4ColorHighlight`, `Pm4ColorSelection`).
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.cs` (Modified):
  - Added `MeshRenderer` instance property and hooked `InvalidateAll()` into `ClearPm4OverlayRuntimeState()` and `ReloadPm4Overlay()`.
  - Removed duplicate color constants and redundant math.
  - Reduced line count from 2,080 to **1,789 lines** ($\le 2,000$ line budget).
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.Selection.cs` (Modified):
  - Received `RebuildPm4MergedObjectGroups` partial class helper (1,147 lines).
- `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` (Modified):
  - Replaced 470 lines of inline PM4 mesh, bounds, node, and flush rendering in `Render()` with a single high-performance call to `_pm4Overlay.MeshRenderer.Render()`.
  - Preserved all telemetry metrics (`pm4BoundsMs`, `pm4GeometryPrepareMs`, `pm4GeometrySubmitMs`, `pm4NodesMs`, etc.).
  - Added `_pm4Overlay.MeshRenderer.Dispose()` to `WorldScene.Dispose()`.
  - Reduced line count from 5,111 to **4,662 lines**.

---

## 2. Line Count Budget Audit (AGENTS.md §10)

| File | Line Count | Status ($\le 2,000$ lines) |
|---|---|---|
| `Terrain/Pm4/IPm4OverlayHost.cs` | 25 | PASS |
| `Terrain/Pm4/Pm4OverlayCacheCodec.cs` | 319 | PASS |
| `Terrain/Pm4/Pm4OverlayColors.cs` | 263 | PASS |
| `Terrain/Pm4/Pm4OverlayCoordinates.cs` | 456 | PASS |
| `Terrain/Pm4/Pm4OverlayGeometry.cs` | 1,334 | PASS |
| `Terrain/Pm4/Pm4OverlayMatching.cs` | 476 | PASS |
| `Terrain/Pm4/Pm4OverlayModels.cs` | 728 | PASS |
| `Terrain/Pm4/Pm4OverlayMprlHelpers.cs` | 210 | PASS |
| `Terrain/Pm4/Pm4OverlayObject.cs` | 377 | PASS |
| `Terrain/Pm4/Pm4OverlayReportModels.cs` | 397 | PASS |
| `Terrain/Pm4/Pm4OverlayScene.cs` | **1,789** | PASS (was 2,080) |
| `Terrain/Pm4/Pm4OverlayScene.Reports.cs` | 1,732 | PASS |
| `Terrain/Pm4/Pm4OverlayScene.Selection.cs` | 1,147 | PASS |
| `Terrain/Pm4/Services/Pm4ColorCodingService.cs` | 128 | PASS |
| `Terrain/Pm4/Services/Pm4SpatialIndex.cs` | 24 | PASS |
| `Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs` | 643 | PASS |

---

## 3. Verification Commands & Outputs

### 3.1 Solution Build
```powershell
dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore
```
- **Exit Status**: 0
- **Errors**: 0
- **Warnings**: 186 (pre-existing NU1902/CS0649/CA2022)

### 3.2 PM4 Unit Test Suite
```powershell
dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --no-build
```
- **Exit Status**: 0
- **Results**: Passed: 177, Failed: 0, Skipped: 0. Duration: 3m 45s.

