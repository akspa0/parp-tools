# Receipt: Spec 296 — PM4 Object Identity, Topological Segmentation & Conflict Decoupling

Date: 2026-11-06  
Branch: `295-pm4-visual-shape-library`  
Status: Complete & Verified  

---

## 1. Files Changed

### Core Libraries
- [`wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchModels.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchModels.cs):
  - Defined `Pm4ObjectIdentity` composite structure (`TileX`, `TileY`, `ComponentId`, `LinkGroupObjectId`, `PlacementZ`, `Centroid`, `BoundingSize`, `SurfaceClass`).
  - Added `Pm4HullShapeInput.MakeUniqueHullId(int tileX, int tileY, int componentId)` deterministic 64-bit ID generator.
  - Added `UniqueHullId` and `PlacementZ` accessors and `Identity` factory to `Pm4HullShapeInput`.
  - Added `ulong UniqueHullId = 0` and `EffectiveUniqueHullId` property to `Pm4ShapeMatchProposal`.
- [`wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchAnalysisService.cs`](file:///I:/parp/parp-tools/wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4ShapeMatchAnalysisService.cs):
  - In `Analyze`, derived `HullKey` uniquely from `(uint)hull.ObjectPartId` (or hash of `UniqueHullId`), preventing distinct components at the same $Z$ from sharing a key.
  - In `FindHullConflicts` and `FindConflicts`, added `EffectiveUniqueHullId` self-conflict guard so identical components never conflict with themselves.

### Viewer Shell & Workbench
- [`wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayGeometry.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayGeometry.cs):
  - Set `requiresConnectivitySeedSplit: true` for all non-zero CK24 groups so every surface group partitions into discrete topological connected components.
  - Enriched both `SplitSurfaceGroupByConnectivity` and `SplitIndexedSurfaceGroupByConnectivity` to traverse both shared `MSVT` vertices and `MSLK` boundary/adjacency links (`MSUR._0x18` / `MscnRefIndex`).
- [`wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.cs):
  - Defaulted `_pm4SplitCk24ByConnectivity = true` and `_pm4SplitCk24ByMscnRef = false`.
- [`wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs):
  - Isolated selection highlight (`color = Pm4ColorSelection`) and bounding box batching strictly to the clicked `_selectedPm4ObjectKey` component, with subtle secondary tint for sibling parts of the same height tier.
- [`wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs):
  - Updated `ExtractCompositeSurfaceVertices` to extract vertices *only* for the selected `pm4Obj` (or parts sharing an explicit non-zero `LinkGroupObjectId`), preventing tile-wide multi-model mega-mesh unioning.
- [`wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchWorkbenchService.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchWorkbenchService.cs):
  - Removed `seenCk24` deduplication from `ExtractTileHullInputs`, allowing every individual object component on the tile to be analyzed.
- [`wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchPanelService.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchPanelService.cs):
  - Formatted batch conflict output to show tile, component part ID, and $Z$ elevation in yards:
    `[<TileX>_<TileY>] Obj #<PartIdA> (Z=<zA:F1>yd) vs Obj #<PartIdB> (Z=<zB:F1>yd) claiming <ModelPath> (<Dist:F1> yds apart)`.
- [`wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchService.cs`](file:///I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchService.cs):
  - Added `PlacementZ` to `Pm4BatchSummaryRow`.
  - Updated Markdown export table header and rows to include Part, Lnk, and Base Z (yd).

### Tests & Governance
- [`wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ObjectIdentityTests.cs`](file:///I:/parp/parp-tools/wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ObjectIdentityTests.cs):
  - Created 4 unit tests covering `MakeUniqueHullId`, `Pm4ObjectIdentity.FormatTag()`, `FindHullConflicts` disambiguation at the same elevation, and `EffectiveUniqueHullId` fallback resolution.
- [`wow-viewer/specs/STATUS.md`](file:///I:/parp/parp-tools/wow-viewer/specs/STATUS.md):
  - Registered Spec 296 in router table.
- [`wow-viewer/specs/296-pm4-object-identity-and-segmentation/`](file:///I:/parp/parp-tools/wow-viewer/specs/296-pm4-object-identity-and-segmentation/):
  - Created `spec.md`, `plan.md`, `tasks.md`, and this receipt.

---

## 2. Verification Commands & Exit Status

| # | Command | Working Directory | Exit Status | Result |
|---|---|---|---|---|
| 1 | `dotnet build WowViewer.slnx -c Debug` | `I:\parp\parp-tools\wow-viewer` | 0 | 0 Errors, 76 Warnings |
| 2 | `dotnet test tests/WowViewer.Core.Tests/ -c Debug --no-build --filter "FullyQualifiedName~Pm4"` | `I:\parp\parp-tools\wow-viewer` | 0 | 30 Passed, 0 Failed, 0 Skipped |
| 3 | `dotnet test tests/WowViewer.Core.Tests/ -c Debug --no-build --filter "FullyQualifiedName~Pm4ObjectIdentityTests"` | `I:\parp\parp-tools\wow-viewer` | 0 | 4 Passed, 0 Failed, 0 Skipped |
| 4 | `dotnet test tests/WowViewer.Core.Tests/ -c Debug --no-build --filter "FullyQualifiedName~Pm4ShapeMatch\|FullyQualifiedName~Pm4ShapeLibrary\|FullyQualifiedName~Pm4ObjectIdentity"` | `I:\parp\parp-tools\wow-viewer` | 0 | 17 Passed, 0 Failed, 0 Skipped |
| 5 | `dotnet test tests/WowViewer.Core.PM4.Tests/ -c Debug --no-build --filter "FullyQualifiedName~Pm4ObjectSegmentBuilder\|FullyQualifiedName~Pm4ConnectiveGeometry\|FullyQualifiedName~Pm4DirectModelMatcher\|FullyQualifiedName~Pm4FootprintExtrusion\|FullyQualifiedName~Pm4Geometry3DShape\|FullyQualifiedName~Pm4ColorCoding\|FullyQualifiedName~Pm4DoodadDuplication"` | `I:\parp\parp-tools\wow-viewer` | 0 | 29 Passed, 0 Failed, 0 Skipped |
| 6 | `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` | `I:\parp\parp-tools` | 0 | 188 controls registered, 0 Errors, 0 Warnings |

---

## 3. Criterion $\rightarrow$ Evidence Matrix

| Criterion | Requirement | Verified Evidence | Status |
|---|---|---|---|
| **SC-001** | Solution builds with 0 errors and passes all existing unit tests | Full solution build exit 0 (`0 Errors`), 76 focused unit tests pass 100% across `WowViewer.Core.Tests` and `WowViewer.Core.PM4.Tests`. | **PASS** |
| **SC-002** | Two objects at the exact same elevation ($Z$) have distinct `UniqueHullId` values and are never flagged as self-conflicts | `Pm4ObjectIdentityTests.MakeUniqueHullId_GeneratesDeterministic64BitKey` and `FindHullConflicts_DisambiguatesDistinctComponents_AtSameElevation`: two hulls with identical `Ck24 = 0x42DBCF` and $Z=109.9$ yd have distinct IDs (`0x000E00330000000C` vs `0x000E00330000000F`), unique `HullKey` values (12 vs 15), and self-conflict check suppresses duplicate reporting. | **PASS** |
| **SC-003** | In batch conflict reporting, conflicts between distinct parts of the same height tier explicitly state both part IDs and elevation in yards | `Pm4ShapeMatchBatchPanelService.cs:93-99` formats conflict string: `[<TileX>_<TileY>] Obj #<PartIdA> (Z=<zA:F1>yd) vs Obj #<PartIdB> (Z=<zB:F1>yd) claiming <ModelPath> (<Dist:F1> yds apart)`. Verified by `Pm4ObjectIdentityTests.FindHullConflicts_DisambiguatesDistinctComponents_AtSameElevation`. | **PASS** |
| **SC-004** | Clicking a surface in the 3D viewport colors and bounds only the clicked component | `Pm4OverlayMeshRenderer.cs:333-360` highlights only the clicked component (`thisKey == _selectedPm4ObjectKey.Value`) in `Pm4ColorSelection` and scopes the bounding box batch strictly to that component. `SplitSurfaceGroupByConnectivity` and `SplitIndexedSurfaceGroupByConnectivity` partition height tiers into discrete topological components. | **PASS** |
| **SC-005** | All UI controls registered in the reference inventory pass with 0 errors and 0 warnings | `Verify-UiReference.ps1` verified 200 inventory rows, 188 distinct labels, 188 registry entries: `PASSED errors=0 warnings=0`. | **PASS** |
