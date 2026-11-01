# Spec 289 Phase 1 Receipt: Top-Surface Magnetization & Placement Fix

**Date**: 2026-11-01  
**Owner**: Spec 289 (PM4 3D Footprint Extrusion & Direct MPQ Matching)  
**Status**: COMPLETE / VERIFIED  

---

## 1. Summary of Changes

Empirical inspection revealed that PM4 collision surfaces for doodads (`MSUR.0x00 == 0x03`, group key 3) represent walkable *top surfaces* where characters walk (e.g. tops of stone pedestals/altars, crates, tables). Previous code set placement origin $Z = Z_{\text{pm4\_centroid}}$. Because an M2 model's local coordinate origin $(0, 0, 0)$ is at its bottom base, this caused the model to sit on top of the PM4 surface, trapping the PM4 mesh on the ground underneath the object.

Phase 1 implemented top-surface alignment across all magnetization and duplication paths:
$$P_Z = Z_{\text{pm4\_top}} - (H_{\text{model\_top}} \cdot s)$$
and horizontal centering:
$$\vec{P}_{XY} = \vec{C}_{\text{pm4\_XY}} - \vec{C}_{\text{model\_XY}}$$

### Files Modified:
1. `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4UnauthoredPlacementResolver.cs`: Updated `AttractPositionToPm4Mesh` to align top surface when `targetZ == 0f` ($z = \text{pm4Centroid.Z} - \text{modelTop}$).
2. `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`: Calculated `modelTopZ = scaledLocalMax.Z` and passed `topAlignedZ = target.WorldCentroid.Z - modelTopZ` into `AttractPositionToPm4Mesh`; updated `worldBoundsMax.Z` to equal `target.WorldCentroid.Z`.
3. `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`: Updated `MagnetizePlacementToPm4` and `MagnetizeTilePlacementsToPm4` to compute `unpositioned` transform bounds, deriving `modelTopZ` and `rotCenter`, and setting `PlacementPosition` so the model's top surface aligns with `bestCentroid.Z`.
4. `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4DoodadDuplicationTests.cs`: Updated `FitModelToTarget_ComputesScaleRatioAndCentroidCorrectly` to verify $P_Z = 96.0$ ($100 - 2 \cdot 2$) and `WorldBoundsMax.Z = 100.0`.

---

## 2. Verification Receipts

| Command | Exit Code | Result |
|---|---|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/ -c Debug` | 0 | **Passed: 177, Failed: 0, Skipped: 0** |
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore` | 0 | **0 Errors, 131 Warnings** |
| `pwsh -NoProfile -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | **PASSED errors=0 warnings=0 (127 controls)** |

### Criterion Mapping:
- **AC-001**: Top-aligned magnetization verified in unit tests (`result.WorldBoundsMax.Z == 100.0f` matching target centroid) and compilation of both single and tile magnetization services.
