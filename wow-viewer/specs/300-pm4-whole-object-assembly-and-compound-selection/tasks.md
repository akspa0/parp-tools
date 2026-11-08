# Tasks 300 — PM4 Whole-Object Assembly, Compound Hull Selection & Unified Spatial Matching

Receipt: `evidence/receipt-spec300.md`

## Phase 1 — Compound Model & Assembly Engine (FR-001, FR-002)

- [x] T001 [Model] Create `Pm4CompoundObject.cs` in `WoWViewer.Terrain.Pm4` defining `Pm4CompoundObject` with aggregate bounds, dimensions, centroid, total triangles, and constituent hulls.
- [x] T002 [Service] Create `Pm4CompoundObjectService.cs` and `Pm4SpatialClustering.cs` implementing Union-Find spatial AABB proximity clustering ($\le 2.5\,\text{yd}$) and MSLK group ID clustering.
- [x] T003 [Assembly] Implement `AssembleTileCompoundObjects(tileX, tileY, rawObjects)` returning complete compound objects and hull-to-compound reverse mapping.

## Phase 2 — Scene & Selection Integration (FR-003)

- [x] T004 [Scene] Update `Pm4OverlayScene.cs` to store `_pm4TileCompoundObjects` and `_pm4HullToCompoundLookup`.
- [x] T005 [Selection] Update `Pm4OverlayScene.Selection.cs`: in `SelectPm4ObjectByRay`, resolve the hit hull's owning `Pm4CompoundObject` and set `_selectedCompoundObject`.
- [x] T006 [Scene] Expose `SelectedCompoundObject` and `IsHullInSelectedCompound(key)` for renderer and UI consumers.

## Phase 3 — Viewport Rendering & Bounding Box (FR-003)

- [x] T007 [Renderer] Update `Pm4OverlayMeshRenderer.cs` to highlight ALL constituent hulls of `_selectedCompoundObject` in the selection color (purple wireframe and solid tint).
- [x] T008 [Scene] Update `WorldScene.cs` selection bounding box rendering to compute the oriented/axis-aligned wireframe around `selectedCompoundObject.WorldBoundsMin` to `WorldBoundsMax`.

## Phase 4 — UI Cockpit & GLB Export (FR-004, FR-005)

- [x] T009 [UI] Update hover tooltip in `Pm4OverlayScene.Selection.cs` and selection info in `SceneHoverAndPickService.cs` to display whole-object name, hull count, and dimensions.
- [x] T010 [UI] Update `Pm4WorkbenchSidebarService.cs` Selected PM4 Hull card to display compound whole-object metrics (`[22_38] Whole Obj #1 (7 hulls, 1142 tris, 38.5x14.2x28.1 yd)`) and constituent hull list.
- [x] T011 [Export] Update `Pm4GlbExporter.cs` and `ExportSelectedPm4ToGlb` in `Pm4WorkbenchService.Exports.cs` to export all constituent hulls of the selected compound object.

## Phase 5 — Verification, UI Reference & Receipts (FR-005, SC-001..SC-005)

- [x] T012 [Tests] Create `Pm4SpatialClusteringTests.cs` in `WowViewer.Core.PM4.Tests` verifying spatial clustering, multi-hull bounds aggregation, and centroid computation.
- [x] T013 [Verification] Run full solution build (`dotnet build WoWViewer.csproj -c Debug -m:1`) and verify 0 errors.
- [x] T014 [Verification] Run unit tests and verify 100% pass (5/5 spatial clustering tests pass; 19/19 spatial/footprint tests pass).
- [x] T015 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings (188 controls verified).
- [x] T016 [Receipt] Record verification commands, exit status, and criterion-to-evidence matrix in `evidence/receipt-spec300.md`.
