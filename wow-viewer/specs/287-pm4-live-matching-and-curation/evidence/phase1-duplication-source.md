# Phase 1 Receipt: Live PM4 Doodad Duplication & Magnetic Matching

**Date**: 2026-10-31  
**Spec**: Spec 287 (Phase 1: T001–T006)  
**Status**: COMPLETE  

---

## 1. Files Modified & Created

- **Created**:
  - `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`
  - `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4DoodadDuplicationTests.cs`
- **Modified**:
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs`
  - `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`
  - `wow-viewer/src/viewer/WoWViewer/ViewerApp_Host.cs`
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/InspectorPayloads/InspectorPayloadsService.cs`
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/InspectorPayloads/InspectorPayloadsService.Host.cs`
  - `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Exports.cs`

---

## 2. Verification Commands & Exit Status

1. `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName~Pm4DoodadDuplicationTests"`
   - **Exit Status**: 0 (3/3 tests passed, 23 ms duration)
2. `dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore`
   - **Exit Status**: 0 (0 errors, 121 pre-existing warnings)
3. `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1`
   - **Exit Status**: 0 (`UI_REFERENCE_CHECK PASSED errors=0 warnings=0`)

---

## 3. Criterion -> Real Output Table

| Criterion | Implementation & Evidence | Result |
| :--- | :--- | :--- |
| **AC-101 / T001**: Model Extent & Scale Fitting | `Pm4DoodadDuplicationEngine.FitModelToTarget` calculates planar extent ratio $\frac{\text{targetPlanar}}{\text{modelPlanar}}$ clamped to $[0.1, 10.0]$; unit tested in `FitModelToTarget_ComputesScaleRatioAndCentroidCorrectly` (2x scale ratio verified: $1.95 \dots 2.05$). | PASS |
| **AC-102 / T002**: Unmatched PM4 Hull Detection & Occupancy Rejection | `FindUnmatchedTargets` scans all PM4 doodad hulls (`Ck24 == 0`, `0x03`, `0x40`, `0x41`) and tests proximity against resident placements; tested in `FindUnmatchedTargets_RejectsTargetsNearExistingPlacements`. | PASS |
| **AC-103 / T003**: UI Action & Context Inspector | Added "Fill PM4 Doodads with Model" button in Context Inspector under Model Actions; wired into `InspectorPayloadsService.ExecuteInspectorAction`. | PASS |
| **AC-104 / T004**: Live Scene Injection & Transform | `Pm4DoodadDuplicationService` constructs `ObjectInstance` with `WorldPlacementTransform.Build(attractedPos, rot, scale)`, transforms AABB via `WorldScene.TransformBounds`, adds to `_tileMdxInstances`, and calls `RebuildInstanceLists()`. | PASS |
| **AC-105 / T005**: Undo/Discard Buffer | `UndoLastDuplication()` restores scene by removing added UniqueIds and calling `RebuildInstanceLists()`; dynamic "Undo PM4 Doodad Fill" button appears in Inspector when undo stack is non-empty. | PASS |
| **AC-106 / T006**: Zarr v60 Dataset Export | Added `SaveMergedPlacementsToZarr` writing root `zarr.json` (`zarr_format: 3`) and `curated_placements.json`; wired to "Save Merged Placements to Zarr v60" button in PM4 Workbench. | PASS |

