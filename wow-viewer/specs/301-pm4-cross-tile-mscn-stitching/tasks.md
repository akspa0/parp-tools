# Spec 301 Tasks: PM4 Cross-Tile MSCN Compound Stitching & Multi-Tile Selection

Status: In Progress
Receipt: `evidence/receipt-spec301.md`

## Phase 1 — Core Cross-Tile Clustering Engine (FR-001)

- [ ] T001 [Model] Define `Pm4CrossTileCompoundDescriptor` in `WowViewer.Core.PM4.Services` capturing compound bounds, tile coordinates, MSCN world points, and connector keys.
- [ ] T002 [Engine] Implement `ClusterCrossTileCompounds` in `Pm4SpatialClustering.cs` with MSCN seam coincidence ($\le 0.5\,\text{yd}$) and boundary AABB proximity ($\le 2.5\,\text{yd}$).
- [ ] T003 [Tests] Add unit tests in `Pm4SpatialClusteringTests.cs` verifying cross-tile compound clustering via MSCN anchor points and boundary contact.

## Phase 2 — Multi-Tile Compound Object Model (FR-002)

- [ ] T004 [Model] Update `Pm4CompoundObject.cs` to support multi-tile structures (`SpanningTiles`, `IsCrossTile`, aggregated multi-tile bounds and metrics).
- [ ] T005 [Formatting] Update `DisplayName` in `Pm4CompoundObject.cs` to format multi-tile spanning objects (e.g. `[13_50 + 13_51] Cross-Tile Whole Obj #0`).

## Phase 3 — Stitching Service & Scene Lifecycle (FR-003)

- [ ] T006 [Service] Implement `StitchCrossTileCompoundObjects` in `Pm4CompoundObjectService.cs`.
- [ ] T007 [Scene] Integrate `StitchCrossTileCompoundObjects` into `Pm4OverlayScene.cs` after cache merge and direct file load.
- [ ] T008 [Lookup] Re-map `_pm4HullToCompoundLookup` so all constituent hulls on all spanning tiles point to the unified cross-tile compound object.
- [ ] T009 [Selection] Update `SelectPm4ObjectByRay` and selection state in `Pm4OverlayScene.Selection.cs` to preserve selection across cross-tile stitches.

## Phase 4 — Viewport Rendering, Cockpit UI & GLB Export (FR-004, FR-005)

- [ ] T010 [Renderer] Verify `Pm4OverlayMeshRenderer.cs` highlights all constituent hulls across all spanning tiles.
- [ ] T011 [Bounds] Verify `WorldScene.cs` renders the unified bounding box spanning the multi-tile structure's world extents.
- [ ] T012 [UI] Update `SceneHoverAndPickService.cs` and `Pm4WorkbenchSidebarService.cs` to display cross-tile identity, spanning tile badges, and multi-tile hull hierarchy.
- [ ] T013 [Export] Verify `Pm4GlbExporter.cs` exports all hulls across all spanning tiles.

## Phase 5 — Verification, UI Reference & Receipts (SC-001..SC-004)

- [ ] T014 [Verification] Run full solution build (`dotnet build WoWViewer.csproj -c Debug -m:1`) and verify 0 errors.
- [ ] T015 [Verification] Run unit tests and verify 100% pass (`dotnet test --filter "FullyQualifiedName~Pm4SpatialClustering"`).
- [ ] T016 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
- [ ] T017 [Receipt] Record verification commands, exit status, and criterion-to-evidence matrix in `evidence/receipt-spec301.md`.
