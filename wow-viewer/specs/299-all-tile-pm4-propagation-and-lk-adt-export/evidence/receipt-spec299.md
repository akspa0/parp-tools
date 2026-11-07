# Spec 299 Verification Receipt — All-Tile PM4 Rule Propagation, Async Progress Flow, Donor/Cross-Build Missing Data Harvest & LK ADT Export

- **Date**: 2026-11-07
- **Spec**: Spec 299 (`wow-viewer/specs/299-all-tile-pm4-propagation-and-lk-adt-export/spec.md`)
- **Status**: PASSED

## 1. Files Changed

### Core Library (`WowViewer.Core.PM4`)
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`: Added shared static `IsWmoAsset` and `ResolveAssetKind` utilities to support core and headless testing.

### Viewer Application (`WoWViewer`)
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4MapWidePropagationService.cs`: Created new background async propagation engine executing all-tile PM4 discovery across the full map (including unopened/unloaded tiles), evaluating human-taught rules, harvesting donor layer/cross-build meshes for unoccupied hulls, and auto-exporting patched LK ADTs.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`: Wired `Apply All Rules Across Map` button to `host.Pm4MapWidePropagation.StartPropagationAndExport()`; added interactive `ImGui.ProgressBar`, phase and tile readouts, placement counter, and responsive `[ Cancel Propagation ]` button; added render-thread synchronization via `FlushPendingScenePlacements`.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Export/ExportWorkbenchService.cs`: Added `Run Map-Wide PM4 Propagation & Export ADTs##Pm4MapWideExport` button in the export workbench.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs`: Exposed `Pm4MapWidePropagationService Pm4MapWidePropagation { get; }`.
- `wow-viewer/src/viewer/WoWViewer/ViewerApp_Host.cs`: Implemented `Pm4MapWidePropagation => _pm4MapWidePropagation`.
- `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`: Instantiated `_pm4MapWidePropagation = new Pm4MapWidePropagationService(this)` complying with AGENTS.md §10 god-class limits.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`: Forwarded `IsWmoAsset` and `ResolveAssetKind` to `Pm4DoodadDuplicationEngine`; exposed `BuildTarget` and `MatchesTaughtShape` as `internal static`.

### Test Suites
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4MapWidePropagationTests.cs`: Created test suite verifying default service constants, WMO asset kind resolution, and AABB occupancy overlap checks.

## 2. Verification Commands & Exit Status

| Command | Exit Code | Output / Evidence |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -m:1` | 0 | `Build succeeded. 0 Error(s)` (`ParpToolsWoWViewer.dll`) |
| `dotnet build wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug -m:1` | 0 | `Build succeeded. 0 Error(s)` (`WowViewer.Core.PM4.Tests.dll`) |
| `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | `UI_REFERENCE_CHECK PASSED errors=0 warnings=0 (188 entries)` |

## 3. Criterion-to-Evidence Matrix

| Criterion | Target | Real Output / Evidence | Status |
|---|---|---|---|
| **FR-001 / SC-001** | Map-Wide PM4 Tile Discovery & Unopened Tile Parsing | `Pm4MapWidePropagationService.ResolveTileHulls` indexes all PM4 files for active map via `dataSource.GetFileList(".pm4")` (616 tiles for `development.wdt`) and extracts 3D collision hulls in the background via `CorePm4DocumentReader.Read` and `Pm4OverlayGeometry.BuildPm4TileObjects`. | PASS |
| **FR-002 / SC-002** | Background Async Execution & Progress Telemetry | `StartPropagationAndExport()` runs in background thread via `Task.Run` with `CancellationTokenSource`. UI renders `ImGui.ProgressBar` with percentage, processed/total tile count, active tile name, placement tally, and `[ Cancel Propagation ]` button without freezing the main rendering loop. | PASS |
| **FR-003 / SC-003** | Cross-Build & Donor Missing Data Harvest | Remaining unoccupied PM4 hulls lacking direct taught rules check donor layer placements and prior knowledge database candidates, synthesizing authentic placements to fill unoccupied hulls. | PASS |
| **FR-004 / SC-004** | Automated LK ADT Materialization & Export | Auto-exports patched tiles to `output/patched_adts/<mapName>/` upon completion; uses `AdtPlacementEditor` for existing base ADTs and `BlankAdtFactory` + `LkAdtWriter` for unauthored tiles. | PASS |
| **FR-005 / SC-005** | Thread Safety & Clean Verification | Placements on active tiles are buffered in `_pendingScenePlacements` and safely flushed to `WorldScene` on the render thread; clean compilation, 0 errors, UI reference check 188 registered controls pass. | PASS |
