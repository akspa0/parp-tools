# Spec 302 Receipt: PM4 Semantic Color-Coding, Single-Object Disambiguation & Null-Safety Fix

**Date:** 2026-11-08  
**Scope:** Fix `NormalizeKey` null-safety propagation crash, fix semantic role color-coding engine to map `DominantGroupKey` (`MSUR.0x08` SurfaceClass), preserve semantic colors in viewport selection rendering, enforce single-object disambiguation in `Pm4SpatialClustering` (doodads decoupled from WMO floors, no blind transitive merging across disparate objects, no cross-tile merging without MSCN).

---

## 1. Files Changed

| File | Change |
|---|---|
| `wow-viewer/src/viewer/WoWViewer/Terrain/WorldAssetPathResolver.cs` | Made `NormalizeKey(string? path)` null-safe, returning `string.Empty` when input is null or whitespace. |
| `wow-viewer/src/viewer/WoWViewer/Terrain/WorldAssetManager.cs` | Guarded `QueueMdxLoad`, `PrioritizeMdxLoad`, `QueueWmoLoad`, `PrioritizeWmoLoad` with null/whitespace check, preventing invalid enqueue or normalization crashes. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4MapWidePropagationService.cs` | Populated explicit `ModelKey` when constructing `newInst` from taught rules and donor placements; guarded `FlushPendingScenePlacements` against null/empty model keys. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4ColorCodingEngine.cs` | Fixed `ComputeSemanticCategoryColor`: prioritized `dominantGroupKey` (`MSUR.0x08` SurfaceClass) over `ck24Type` (which is an IEEE-754 exponent byte). Authoritatively mapped `0x03` (doodad orange), `0x10` (lowest floor green), `0x11`/`0x12` (standard floor cyan), `0x13` (upper floor royal blue), `0x14` (roof purple), `0x01` (terrain linker lime), and `0x02` (MSCN connector turquoise). |
| `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.Reports.cs` | Updated `GetPm4LegendColor` for `SemanticCategory` to pass `(byte)value` as `dominantGroupKey`. |
| `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs` | Preserved semantic surface colors during selection in `SemanticCategory` mode by blending 25% with selection yellow (`Vector3.Lerp(color, Pm4ColorSelection, 0.25f)`) rather than overwriting 100% of the mesh with flat yellow. Selected object boundary remains highlighted by the 3D bounding box. |
| `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4SpatialClustering.cs` | Extended `ClusterItems` with `surfaceClassSelector` to enforce: 1) Doodad (`0x03`) isolation from WMO floors (`0x10..0x16`); 2) Separate doodads non-merging; 3) WMO floor slices requiring substantial 2D horizontal overlap ($overlapX > 0.5\text{ yd}$ and $overlapY > 0.5\text{ yd}$) and vertical tolerance to assemble the same structure, preventing side-by-side structures (ship, gangways, docks, terrain) from transitively merging. In `ClusterCrossTileCompounds`, removed the blind fallback that merged all compounds lacking MSCN points across seams. |
| `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4CompoundObjectService.cs` | Passed `surfaceClassSelector: (h, idx) => h.DominantGroupKey` to `Pm4SpatialClustering.ClusterItems`. |
| `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4SpatialClusteringTests.cs` | Added 4 unit tests: doodad isolation from WMO floors, separate doodads non-merging, side-by-side structures non-merging, and cross-tile compounds non-merging without MSCN points. |
| `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4ColorCodingServiceTests.cs` | Added tests verifying authoritative semantic color assignments for all surface classes when passed real-world exponent bytes and dominant group keys. |

---

## 2. Exact Verification Commands Run & Exit Codes

| Command | Exit Code | Result |
|---|:---:|---|
| `dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -m:1 --no-restore` | 0 | Solution built with 0 errors. |
| `dotnet build I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --no-restore --disable-build-servers -m:1 -v minimal` | 0 | PM4 tests built with 0 errors. |
| `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~Pm4SpatialClusteringTests\|FullyQualifiedName~Pm4ColorCodingServiceTests"` | 0 | 16/16 tests passed cleanly (Duration: 60 ms). |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | `UI_REFERENCE_CHECK PASSED errors=0 warnings=0` (200 inventory rows, 188 distinct labels, 188 registry entries). |

---

## 3. Requirement to Evidence Mapping

| Requirement / Acceptance Criterion | Verification Evidence |
|---|---|
| **AC-01**: `NormalizeKey(null)` returns `string.Empty` without throwing `NullReferenceException`. | `WorldAssetPathResolver.NormalizeKey(string? path)` handles null/whitespace strings returning `string.Empty`. `WorldAssetManager` guards load queues against empty keys. |
| **AC-02**: Propagation rule application flushes placements cleanly into scene and queue loaders without crashing. | `Pm4MapWidePropagationService` assigns valid `ModelKey` from `taught.ModelPath`/`ModelName` and donor metadata, and guards `FlushPendingScenePlacements` before queueing. |
| **AC-03**: `Pm4ColorCodingEngine.ComputeSemanticCategoryColor` returns correct RGB vectors for all surface classes (`0x03`, `0x10`, `0x11`, `0x12`, `0x13`, `0x14`, `0x01`, `0x02`). | Verified by `SemanticCategory_WhenPassedDominantGroupKeySurfaceClass_ReturnsDistinctAuthoritativeColors` test in `Pm4ColorCodingServiceTests` (PASSED). |
| **AC-04**: When `SemanticCategory` is active, viewport mesh rendering displays distinct semantic colors across doodads and floors even when an object or compound is selected. | Verified in `Pm4OverlayMeshRenderer.cs` line 640–650: in `SemanticCategory` mode, `color` blends subtly with selection yellow (0.25f), leaving orange doodads, cyan floors, green lowest floors readable, while bounded by the yellow bounding box. |
| **AC-05**: Clustering tests prove that doodads (`0x03`) sitting on a WMO floor (`0x11`) remain separate clusters. | Verified by `ClusterItems_DoodadMesh_DoesNotMergeWithWmoFloor` test in `Pm4SpatialClusteringTests` (PASSED). |
| **AC-06**: Clustering tests prove that separate doodads placed near each other do not merge. | Verified by `ClusterItems_MultipleDoodads_DoNotMergeWithEachOther` test in `Pm4SpatialClusteringTests` (PASSED). |
| **AC-07**: Cross-tile compound clustering tests prove that compounds lacking MSCN points do not merge across tile seams solely by AABB proximity. | Verified by `ClusterCrossTileCompounds_WithoutMscnOrConnectorKeys_DoNotMergeAcrossBoundary` test in `Pm4SpatialClusteringTests` (PASSED). |
| **AC-08**: Side-by-side structures (ship and adjacent dock touching at edge) do not transitively merge into a single compound object. | Verified by `ClusterItems_SideBySideStructures_DoNotMergeTransitively` test in `Pm4SpatialClusteringTests` (PASSED). |

