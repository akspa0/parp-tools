# Receipt 300 — PM4 Whole-Object Assembly, Compound Hull Selection & Unified Spatial Matching

**Date**: 2026-11-07  
**Spec**: `wow-viewer/specs/300-pm4-whole-object-assembly-and-compound-selection/`  
**Owner**: Epic 251 (`wow-viewer/specs/251-epic-viewer-ux-and-code-health/spec.md`)  
**Status**: Complete & Verified

---

## 1. Summary of Deliverables

- **Core Spatial Clustering**: Created `Pm4SpatialClustering.cs` in `WowViewer.Core.PM4.Services` implementing Disjoint Set Union (Union-Find) clustering with 3D AABB proximity closure ($\le 2.5\,\text{yd}$ tolerance) and MSLK `LinkGroupObjectId` topological linkage.
- **Compound Object Assembly Engine**: Created `Pm4CompoundObject.cs` and `Pm4CompoundObjectService.cs` in `WoWViewer.Terrain.Pm4` to assemble elevation-fragmented CK24 micro-hulls into unified 3D physical structures (`Pm4CompoundObject`) with aggregate dimensions, bounding volume, total surfaces, total triangles, and constituent hulls.
- **Single-Click Whole-Object Viewport Selection**: Clicking any triangle of any constituent hull in the viewport resolves through `_pm4HullToCompoundLookup` to select the entire `Pm4CompoundObject`.
- **Viewport Wireframe & Solid Shading**: Updated `Pm4OverlayMeshRenderer.cs` to render ALL constituent hulls of the selected compound object in selection color (purple wireframe and solid tint).
- **Selection Bounding Box**: Updated `WorldScene.cs` selection bounding box renderer to enclose the entire compound 3D structure from `selectedCompoundObject.WorldBoundsMin` to `WorldBoundsMax`.
- **Hover Tooltip & Sidebar Inspection**: Updated `Pm4OverlayScene.Selection.cs` and `Pm4WorkbenchSidebarService.cs` to identify whole objects (e.g. `PM4 Whole Object #0 (7 hulls, 1142 tris, 38.5x14.2x28.1 yd)`) instead of cryptic per-part elevation slices, with an expandable constituent hull tree.
- **Whole-Object Footprint & Yaw Alignment**: Updated `ExtractCompositeSurfaceVertices` in `Pm4DoodadDuplicationService.cs` to gather all triangles across the entire compound structure for accurate 3D centroid, footprint extrusion, and yaw alignment.
- **Compound GLB Export**: Updated `Pm4GlbExporter.cs` and `Pm4WorkbenchService.Exports.cs` to export all constituent hulls of the selected compound object into a single GLB file with independent object nodes.

---

## 2. Files Changed

### Created
- `wow-viewer/src/core/WowViewer.Core.PM4/Services/Pm4SpatialClustering.cs`: Core Disjoint Set Union (Union-Find) and spatial proximity clustering algorithm.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Models/Pm4CompoundObject.cs`: Model encapsulating compound whole objects, aggregate bounding extents, centroid, dimensions, and constituent hulls.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4CompoundObjectService.cs`: Service clustering raw PM4 tile hulls into compound whole objects and reverse lookup index.
- `wow-viewer/tests/WowViewer.Core.PM4.Tests/Pm4SpatialClusteringTests.cs`: Unit tests for tolerance overlap, DSU merging, spatial proximity clustering, and bounds aggregation.
- `wow-viewer/specs/300-pm4-whole-object-assembly-and-compound-selection/spec.md`: User specification.
- `wow-viewer/specs/300-pm4-whole-object-assembly-and-compound-selection/plan.md`: Technical implementation design.
- `wow-viewer/specs/300-pm4-whole-object-assembly-and-compound-selection/tasks.md`: Task checklist and verification roadmap.

### Modified
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.cs`: Added `_pm4TileCompoundObjects`, `_pm4HullToCompoundLookup`, `_selectedCompoundObject`, and compound queries.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.Selection.cs`: Raycast selection sets `_selectedCompoundObject`; hover tooltip formats whole-object title and metrics.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Pm4OverlayScene.Reports.cs`: Key selection sets `_selectedCompoundObject`.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/Services/Pm4OverlayMeshRenderer.cs`: Selection color applies to all hulls in `SelectedCompoundObject`; mesh buffer fingerprint includes compound selection.
- `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`: Selection bounding box wraps compound object's `WorldBoundsMin` to `WorldBoundsMax`.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`: Displays compound whole-object metrics and constituent hull tree; centroid uses compound center.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4DoodadDuplicationService.cs`: `ExtractCompositeSurfaceVertices` extracts all triangles across all constituent hulls.
- `wow-viewer/src/viewer/WoWViewer/Export/Pm4GlbExporter.cs`: Added `ExportCompoundObjectGlb` exporting all constituent hulls.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.Exports.cs`: `ExportSelectedPm4ObjectGlb` exports all constituent hulls of the selected compound object.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs`: Displays compound whole-object identity, metrics, and bounds in selection info.
- `wow-viewer/specs/STATUS.md`: Updated Spec 300 entry to Complete & Verified.

---

## 3. Verification Commands & Exit Status

| Step | Command | Result / Exit Status |
|---|---|---|
| Core Spatial Tests | `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug -m:1 --filter "FullyQualifiedName~Pm4SpatialClusteringTests"` | **Passed! 5/5 passed, 0 failed, 24 ms (Exit 0)** |
| PM4 Spatial/Footprint Tests | `dotnet test wow-viewer/tests/WowViewer.Core.PM4.Tests/WowViewer.Core.PM4.Tests.csproj -c Debug -m:1 --filter "FullyQualifiedName~Spatial\|FullyQualifiedName~Clustering\|FullyQualifiedName~Footprint"` | **Passed! 19/19 passed, 0 failed, 50 ms (Exit 0)** |
| Solution Project Build | `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug -m:1` | **Build Succeeded: 0 Errors, 136 Warnings (Exit 0)** |
| UI Reference Check | `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` | **UI_REFERENCE_CHECK PASSED: 200 inventory rows, 188 controls, 0 errors, 0 warnings (Exit 0)** |

---

## 4. Criterion-to-Evidence Matrix

| Criterion | Requirement | Real Evidence / Output | Status |
|---|---|---|---|
| **FR-001 / SC-001** | Whole-Object Compound Assembly | Elevation-fragmented CK24 slices and sub-components are clustered into cohesive compound whole objects using 3D AABB proximity ($\le 2.5\,\text{yd}$) and shared MSLK link group IDs. | PASS: `Pm4SpatialClustering` and `Pm4CompoundObjectService` cluster multi-part structures into single compound objects; 5/5 dedicated unit tests verify adjacent and linked clustering. |
| **FR-002 / SC-002** | Single-Click Whole-Object Viewport Selection | Clicking any triangle of any constituent hull in the viewport selects the entire compound object. | PASS: `Pm4OverlayScene.Selection.cs` maps hit hull key to owning `Pm4CompoundObject` in `_pm4HullToCompoundLookup` and sets `_selectedCompoundObject`. |
| **FR-003 / SC-003** | Viewport Rendering & Bounding Box | All constituent hulls highlight in selection color (purple wireframe and solid tint) and the selection bounding box encloses the entire compound object's 3D bounds. | PASS: `Pm4OverlayMeshRenderer.cs` checks `IsHullInSelectedCompound`; `WorldScene.cs` renders bounding box wrapping `SelectedCompoundObject.WorldBoundsMin` to `WorldBoundsMax`. |
| **FR-004 / SC-004** | UI Cockpit Metrics & Constituent Hull Tree | Sidebar and hover tooltips identify whole structures with true 3D dimensions, total triangles, and constituent hull tree rather than raw per-CK24 slices. | PASS: `Pm4WorkbenchSidebarService.cs` renders `Selected PM4 Whole Object: [<Tile>] Obj #<Id> (<N> hulls)` with dimensions, centroid, and expandable constituent hull tree; `SceneHoverAndPickService.cs` shows compound identity. |
| **FR-005 / SC-005** | Compound GLB Export & Model Alignment | "Export Selected PM4 to GLB" exports all constituent hulls of the compound object; "Align Model to this PM4 Hull" aligns to the full compound 3D structure. | PASS: `Pm4GlbExporter.ExportCompoundObjectGlb` exports all constituent hulls into the GLB; `ExtractCompositeSurfaceVertices` gathers triangles across all compound hulls. |
