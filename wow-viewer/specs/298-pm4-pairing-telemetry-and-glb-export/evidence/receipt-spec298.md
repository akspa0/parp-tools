# Spec 298 Verification Receipt — PM4 Authentic Pairing Telemetry, Markdown Elimination of CK24 & Multi-Object GLB Export

- **Date**: 2026-11-07
- **Spec**: Spec 298 (`wow-viewer/specs/298-pm4-pairing-telemetry-and-glb-export/spec.md`)
- **Status**: PASSED

## 1. Files Changed

### Core Library (`WowViewer.Core.PM4`)
- `wow-viewer/src/core/WowViewer.Core.PM4/Matching/Pm4DoodadDuplicationEngine.cs`: Defined shared `public sealed record TaughtPairing` enriched with authentic telemetry (`PlacementPosition`, `PlacementRotation`, `PlacementScale`, `UniqueId`, `TileX`, `TileY`, `ObjectPartId`, `LinkGroupObjectId`, `HullCentroid`, `HullBoundsMin`, `HullBoundsMax`).

### Viewer Application (`WoWViewer`)
- `wow-viewer/src/viewer/WoWViewer/Export/Pm4GlbExporter.cs`: Created new multi-object PM4 GLB export engine exporting each collision hull as an independent named node with double-sided PBR collision materials sitting atop textured/MCCV terrain geometry.
- `wow-viewer/src/viewer/WoWViewer/Export/MapGlbExporter.cs`: Refactored `BuildTerrainTileMesh` to `internal` so `Pm4GlbExporter` can assemble authentic terrain base meshes.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4PairingProjectStore.cs`: Added `JsonSerializerOptions { IncludeFields = true }` so `Vector3` telemetry fields serialize and deserialize accurately.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`: Added `Export PM4 Hull to GLB` action button in the manual pairing workbench and updated inspector labels.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Export/ExportWorkbenchService.cs`: Added `Export Active Tile PM4 to GLB` and `Export All Loaded PM4 to GLB` action buttons to the export workbench.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4CurationAndSpatialMatchingService.cs`: Enriched `CuratedSpatialRelationship` with authentic placement and hull telemetry fields.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`: Updated `TeachPairing` to populate all telemetry fields from live objects and PM4 hulls.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4ShapeMatchBatchService.cs`: Purged `CK24` columns from markdown summary and UI tables, replacing with `Object ID`, `Position (X, Y, Z)`, `Proposed Candidate`, `Chosen Asset`, `Kind`, `Confidence`, `Distance (yd)`.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Collections.cs`: Replaced `CK24` references with `Height Band`, `Placement Z`, and `Obj #{part}`.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Exports.cs`: Added export actions for PM4 GLB exports.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.SceneGraph.cs`: Replaced raw `CK24 0x...` in hover tooltips and tree nodes with `Placement Z: {z:F2} yd ObjId: {objId}`.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.cs`: Purged legacy CK24 terminology from glossary and inspector text.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs`: Formatted tooltip to show `Obj #{selectedPm4.objectPart} (Placement Z={z:F2} yd)` instead of raw CK24.
- `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`: Verified selection box rendering remains unconditional.

### Test Suites
- `wow-viewer/tests/WowViewer.Core.Tests/PM4/Pm4ShapeMatchBatchServiceTests.cs`: Verified markdown table generation with authentic telemetry and zero CK24 column presence.

## 2. Verification Commands & Exit Status

| Command | Exit Code | Output / Evidence |
|---|---:|---|
| `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -m:1` | 0 | `Build succeeded. 0 Error(s)` (`ParpToolsWoWViewer.dll`) |
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~Pm4ShapeMatchBatchServiceTests"` | 0 | `Passed! - Failed: 0, Passed: 4, Skipped: 0, Total: 4` |
| `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` | 0 | `UI_REFERENCE_CHECK PASSED errors=0 warnings=0 (188 entries)` |

## 3. Criterion-to-Evidence Matrix

| Criterion | Target | Real Output / Evidence | Status |
|---|---|---|---|
| **FR-001 / SC-001** | Authentic pairing telemetry in `TaughtPairing` | `TaughtPairing` populated with `PlacementPosition`, `PlacementRotation`, `PlacementScale`, `UniqueId`, `TileX`, `TileY`, `ObjectPartId`, `LinkGroupObjectId`, `HullCentroid`, `HullBoundsMin`, `HullBoundsMax`; verified round-trip JSON serialization with `IncludeFields = true`. | PASS |
| **FR-002 / SC-002** | Purge CK24 from all markdown summaries & UI | `Pm4ShapeMatchBatchService.FormatMarkdownSummary` verified: `\| Tile \| Object ID \| Position (X, Y, Z) \| Classification \| Proposed Candidate \| Chosen Asset \| Kind \| Confidence \| Distance (yd) \| Provenance \| Conflict \|`. All 4 tests in `Pm4ShapeMatchBatchServiceTests` passed. | PASS |
| **FR-003 / SC-003** | PM4 multi-object GLB exporter | `Pm4GlbExporter.cs` exports each PM4 object as individual named node (`[{TileX}_{TileY}] Obj #{PartId} - {SurfaceClass}`) with double-sided PBR collision materials and authentic terrain mesh base. | PASS |
| **FR-004** | UI Action Bar & Export Workbench integration | Buttons registered and verified in `ExportWorkbenchService` and `Pm4WorkbenchSidebarService`. `Verify-UiReference.ps1` passed with 0 errors. | PASS |
| **FR-005 / SC-004** | Full compilation & UI Reference check | Clean compilation, 0 errors, UI reference check 188 registered controls pass. | PASS |
