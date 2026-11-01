# Phase 2 Receipt: Burgundy-Red MCCV / Minimap Terrain Hole Detection

**Date**: 2026-10-31  
**Spec**: Spec 287 (Phase 2: T009–T013)  
**Status**: COMPLETE  

---

## 1. Files Modified & Created

- **Created**:
  - `wow-viewer/src/core/WowViewer.Core/Maps/MccvHoleDetection.cs`
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TerrainAnalysis/MccvHoleDetectionService.cs`
  - `wow-viewer/tests/WowViewer.Core.Tests/Maps/MccvHoleDetectionTests.cs`
- **Modified**:
  - `wow-viewer/src/core/WowViewer.Core/Maps/TerrainChunkData.cs` (Mutable `HoleMask` and `HoleMask64`)
  - `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainChunkData.cs` (Mutable `HoleMask` and `HoleMask64`)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs` (`MccvHoleDetectionService MccvHoleDetection`)
  - `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` (instantiate `_mccvHoleDetection`)
  - `wow-viewer/src/viewer/WoWViewer/ViewerApp_Host.cs` (expose `MccvHoleDetection`)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Chrome/ViewerChromeService.Options.cs` (Add "Derive Holes from Burgundy MCCV" button)
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/TerrainControls/TerrainControlsPanelService.Sidebars.cs` (Add "Burgundy MCCV Minimap Holes" collapsing header)
  - `wow-viewer/eng/ui/ui-controls.json` & `eng/ui/New-UiControlRegistry.ps1` (Register `derive-holes-from-burgundy-mccv`)
  - `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` (Update generated reference)

---

## 2. Verification Commands & Exit Status

1. `dotnet test I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --no-build --filter "FullyQualifiedName~MccvHoleDetectionTests"`
   - **Exit Status**: 0 (5/5 tests passed, 23 ms duration)
2. `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1`
   - **Exit Status**: 0 (`UI_REFERENCE_CHECK PASSED errors=0 warnings=0`)
3. `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`
   - **Exit Status**: 0 (0 errors)

---

## 3. Criterion -> Real Output Table

| Criterion | Implementation & Evidence | Result |
| :--- | :--- | :--- |
| **AC-201 / T009**: Solid Burgundy Chroma & Variance Detection | `MccvHoleDetection.IsSolidBurgundyChunk` checks $R \ge 70, G \le 35, B \le 35$ and color variance $\sigma \le 18.0$; verified by unit tests `IsSolidBurgundyChunk_DetectsSolidBurgundyColor` (passes for (85, 10, 10)) and rejects noisy/shaded chunks. | PASS |
| **AC-202 / T009**: Connected Component Cluster Filtering | `MccvHoleDetection.FilterContiguousClusters` runs 4-neighborhood BFS; verified by `FilterContiguousClusters_RejectsIsolatedSingleChunks` and `FilterContiguousClusters_AcceptsContiguousGroupsOfTwoOrMore`. | PASS |
| **AC-203 / T010**: MccvHoleDetectionService Analysis & Rebuild | `MccvHoleDetectionService.DeriveBurgundyHoles` iterates loaded tiles, identifies candidates, filters $\ge 2$ clusters, updates `chunk.HoleMask = 0xFFFF` and `chunk.HoleMask64 = ~0UL`, and invokes `terrainManager.ReplaceTileChunksAndRebuild`. | PASS |
| **AC-204 / T011**: UI Integration in Sidebar & Options | "Derive Holes from Burgundy MCCV" button added in `TerrainControlsPanelService.Sidebars.cs` (under `DrawTerrainControlsContent` and `DrawTerrainLabSubTab`) and in `ViewerChromeService.Options.cs` (under `DrawOptionsLayersContent`). | PASS |
| **AC-205 / T012**: Live Dynamic GPU Mesh Rebuild | Setting `HoleMask = 0xFFFF` on `TerrainChunkData` causes `TerrainChunkMath.BuildChunkIndices` to emit 0 triangle indices for all holed units when re-meshed via `terrainManager.ReplaceTileChunksAndRebuild`. | PASS |

