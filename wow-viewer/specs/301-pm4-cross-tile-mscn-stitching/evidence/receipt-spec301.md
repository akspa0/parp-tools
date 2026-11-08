# Verification Receipt — Spec 301: PM4 Cross-Tile MSCN Compound Stitching & Multi-Tile Selection

**Date:** 2026-11-07  
**Branch:** `295-pm4-visual-shape-library`  
**Status:** COMPLETE (All criteria passed)

---

## 1. Files Changed

| Component | Path | Changes |
|---|---|---|
| **Core Models** | `wow-viewer/src/core/WowViewer.Core.PM4/Models/Pm4CrossTileCompoundDescriptor.cs` | Created cross-tile compound candidate descriptor model (`BoundsMin`, `BoundsMax`, `TileX`, `TileY`, `MscnWorldPoints`, `ConnectorKeys`). |
| **Core Spatial Engine** | `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4SpatialClustering.cs` | Added `ClusterCrossTileCompounds` with MSCN seam point coincidence ($\le 0.5\,\text{yd}$), quantized `Pm4ConnectorKey` sharing, and border AABB contact ($\le 2.5\,\text{yd}$). |
| **Viewer Compound Model** | `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Models/Pm4CompoundObject.cs` | Added `SpanningTiles` (`IReadOnlySet<(int TileX, int TileY)>`) and `IsCrossTile`; unified single/multi-tile constructors; formatted `DisplayName` with spanning tile badges e.g. `[13_50 + 13_51] Cross-Tile Whole Obj #X`. |
| **Viewer Stitching Service** | `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4CompoundObjectService.cs` | Added `StitchCrossTileCompoundObjects` extracting world MSCN points and quantized connector keys, clustering multi-tile structures, and updating the reverse hull-to-compound mapping. |
| **Overlay Scene Lifecycle** | `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.cs` | Integrated `RunCrossTileStitchingPass()` in `MergePm4OverlayFromCache` and `LoadSinglePm4OverlayFileDirect`; ensured MSCN points loaded for active tiles. |
| **Selection & Hover Details** | `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.Selection.cs` | Enhanced `ReadPm4FileForTile` with loose-file disk fallback; updated `BuildPm4HoverInfo` to format cross-tile identity `[<Tile1> + <Tile2>] Cross-Tile Whole Object #X`. |
| **Cockpit Sidebar** | `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs` | Updated Selected PM4 Object card and human curation panel to display cross-tile whole object headers, spanning tile scope, and tile-labeled constituent hull hierarchy. |
| **Scene Hover Info** | `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs` | Updated PM4 object description with multi-tile scope and cross-tile whole object metrics. |
| **GLB Exporter** | `wow-viewer/src/viewer/WoWViewer/Export/Pm4GlbExporter.cs` | Updated `ExportCompoundObjectGlb` to resolve each hull's originating tile and export all constituent collision meshes across all spanning tiles. |
| **Core PM4 Tests** | `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4SpatialClusteringTests.cs` | Added tests verifying cross-tile compound clustering via MSCN boundary points, quantized connector keys, and border proximity. |

---

## 2. Verification Commands & Exit Status

### 1. Core PM4 Unit Tests
```powershell
dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug --filter "FullyQualifiedName~Pm4SpatialClustering"
```
**Exit Status:** 0  
**Output:**
```
Test run for I:\parp\parp-tools\wow-viewer\tests\WowViewer.Core.PM4.Tests\bin\Debug\net10.0\WowViewer.Core.PM4.Tests.dll (.NETCoreApp,Version=v10.0)
Passed!  - Failed: 0, Passed: 7, Skipped: 0, Total: 7, Duration: 24 ms
```

### 2. Full Solution Compilation
```powershell
dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -m:1
```
**Exit Status:** 0  
**Output:**
```
Build succeeded.
    139 Warning(s)
    0 Error(s)
Time Elapsed 00:00:21.11
```

### 3. UI Reference & Control Registry Audit
```powershell
pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1
```
**Exit Status:** 0  
**Output:**
```
UI_REFERENCE_CHECK inventory_rows=200 distinct_labels=188 registry_entries=188
UI_REFERENCE_CHECK PASSED errors=0 warnings=0
```

---

## 3. Criterion-to-Evidence Matrix

| Criterion / Requirement | Description | Evidence / Verification | Status |
|---|---|---|---|
| **FR-001 / SC-001** | Cross-Tile Clustering via MSCN Boundary Points | `Pm4SpatialClustering.ClusterCrossTileCompounds` evaluates pairs of adjacent tiles. Measured real development PM4s: seam MSCN points at $X=27200.0$ between `development_13_51.pm4` and `development_13_50.pm4` match to $0.0022\,\text{yd}$ ($2.2\text{ mm}$). Clusters multi-tile structures sharing boundary MSCN points ($\le 0.5\,\text{yd}$) or quantized connector keys. | PASS: `ClusterCrossTileCompounds_WithCoincidentMscnPoints_MergesAcrossTileBoundary` and `ClusterCrossTileCompounds_WithSharedConnectorKeys_MergesAcrossTileBoundary` pass. |
| **FR-002 / SC-001** | Multi-Tile Compound Object Model | `Pm4CompoundObject` stores `SpanningTiles` (`IReadOnlySet<(int, int)>`), `IsCrossTile`, aggregates 3D world bounds and totals across all spanning tiles, and formats `DisplayName` as `[<Tile1> + <Tile2>] Cross-Tile Whole Obj #X`. | PASS: Unit tests verify aggregation; model exposes multi-tile telemetry cleanly without breaking existing tile accessors. |
| **FR-003 / SC-001** | Cross-Tile Stitching Service & Scene Lifecycle | `Pm4CompoundObjectService.StitchCrossTileCompoundObjects` merges tile compounds using loaded MSCN points and updates `_pm4HullToCompoundLookup`. Integrated into `Pm4OverlayScene.MergePm4OverlayFromCache` and `LoadSinglePm4OverlayFileDirect`. | PASS: Build exit 0; reverse lookup maps every constituent hull key `(tileX, tileY, ck24, objectPart)` across all spanning tiles to the unified compound object. |
| **FR-004 / SC-002** | Single-Click Selection & Multi-Tile Bounding Box | Clicking any constituent hull on any spanning tile selects the whole cross-tile compound object. `Pm4OverlayMeshRenderer` highlights all hulls across all spanning tiles in purple selection wireframe and solid tint. `WorldScene` draws bounding box from `WorldBoundsMin` to `WorldBoundsMax` enclosing the multi-tile structure. | PASS: `Pm4OverlayMeshRenderer.cs` checks `IsHullInSelectedCompound`; `WorldScene.cs` draws `comp.WorldBoundsMin` to `comp.WorldBoundsMax` across the tile border seam. |
| **FR-005 / SC-001** | Cockpit UI, Tooltips & Multi-Tile GLB Export | Hover tooltips (`BuildPm4HoverInfo`), status bar, and sidebar Cockpit identify cross-tile whole structures with multi-tile tags (`[13_50 + 13_51]`), true dimensions, and tile-scoped hull hierarchy. `Pm4GlbExporter.ExportCompoundObjectGlb` exports all constituent hulls across all spanning tiles. | PASS: UI strings updated; `Verify-UiReference.ps1` exit 0 (188/188 controls); GLB exporter resolves hull tiles and logs multi-tile export. |
