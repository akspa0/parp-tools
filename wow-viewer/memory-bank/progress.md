# Progress — wow-viewer

## 2026-11-07 — Spec 300 Complete (PM4 Whole-Object Assembly, Compound Hull Selection & Unified Spatial Matching)

- Solved the fundamental fragmentation issue where structures (such as ships, houses, towers) spanning multiple elevation levels were chopped into disconnected per-CK24 horizontal slices.
- Created `Pm4SpatialClustering` in `WowViewer.Core.PM4.Services`: Disjoint Set Union (Union-Find) clustering algorithm with 3D AABB proximity closure ($\le 2.5\,\text{yd}$) and shared MSLK `LinkGroupObjectId` topological linkage.
- Created `Pm4CompoundObject` and `Pm4CompoundObjectService` in `WoWViewer.Terrain.Pm4`: clusters raw tile hulls into cohesive whole physical structures with true 3D dimensions, bounding volumes, total triangles, and constituent hulls.
- Implemented single-click whole-object viewport raycast picking in `Pm4OverlayScene.Selection.cs`: clicking any triangle of any hull resolves through `_pm4HullToCompoundLookup` to select the owning `Pm4CompoundObject`.
- Updated `Pm4OverlayMeshRenderer.cs` to highlight ALL constituent hulls of the selected compound object in purple wireframe and solid tint.
- Updated `WorldScene.cs` selection bounding box to wrap the entire compound 3D structure from `selectedCompoundObject.WorldBoundsMin` to `WorldBoundsMax`.
- Updated hover tooltips (`Pm4OverlayScene.Selection.cs`), selection debug info (`SceneHoverAndPickService.cs`), and sidebar cards (`Pm4WorkbenchSidebarService.cs`) to display compound whole-object identity, dimensions ($L \times W \times H$), and an expandable constituent hull tree.
- Updated `ExtractCompositeSurfaceVertices` in `Pm4DoodadDuplicationService.cs` to extract all triangles across all constituent hulls of the compound object, ensuring accurate 3D centroid, footprint extrusion, and yaw alignment.
- Updated `Pm4GlbExporter.cs` and `Pm4WorkbenchService.Exports.cs` (`ExportCompoundObjectGlb`) to export all constituent hulls of the compound object into a single GLB file with independent object nodes.
- Verification: 5/5 spatial clustering unit tests and 19/19 spatial/footprint unit tests pass in `WowViewer.Core.PM4.Tests`; solution builds with 0 errors (`dotnet build WoWViewer.csproj -c Debug -m:1`); UI reference check passes (188/188 controls, 0 errors/warnings). Receipt: `wow-viewer/specs/300-pm4-whole-object-assembly-and-compound-selection/evidence/receipt-spec300.md`. Operator visual proof owed.

## 2026-11-07 — Spec 299 Complete (All-Tile PM4 Rule Propagation, Async Progress Flow, Donor Harvest & LK ADT Export)

- Implemented `Pm4MapWidePropagationService` in `WoWViewer`: executes whole-map PM4 rule propagation across all tiles (including unopened/unloaded tiles) using `IDataSource.GetFileList(".pm4")` (616 tiles for `development.wdt`) and on-demand background parsing via `CorePm4DocumentReader.Read` and `Pm4OverlayGeometry.BuildPm4TileObjects`.
- Replaced freezing synchronous button with non-blocking async execution via `Task.Run` and `CancellationTokenSource`.
- Added interactive UI controls in `Pm4WorkbenchSidebarService`: `ImGui.ProgressBar` with real-time percentage, processed/total tile counters, active tile name, placement counter, and responsive `[ Cancel Propagation ]` button.
- Implemented cross-build and donor missing data harvesting: unoccupied PM4 hulls lacking direct human-taught pairings harvest matching placements from active donor layers and prior knowledge candidates.
- Implemented automated LK ADT materialization and export upon propagation completion to `output/patched_adts/<mapName>/` using `AdtPlacementEditor` for existing base ADTs and `BlankAdtFactory` + `LkAdtWriter` for unauthored tiles.
- Added thread-safe scene placement queue `_pendingScenePlacements` synchronized via `FlushPendingScenePlacements` during the sidebar render loop to safely spawn newly populated instances in the viewport without concurrency hazards.
- Verification: full solution builds with 0 errors (`dotnet build WoWViewer.csproj -c Debug -m:1`); UI reference check passes (188/188 controls, 0 errors/warnings); unit tests in `Pm4MapWidePropagationTests` pass. Receipt: `wow-viewer/specs/299-all-tile-pm4-propagation-and-lk-adt-export/evidence/receipt-spec299.md`. Operator visual and LK ADT inspection verification owed.

## 2026-11-07 — Spec 298 Complete (PM4 Authentic Pairing Telemetry, Markdown Elimination of CK24 & Multi-Object GLB Export)

- Enriched `TaughtPairing` record in Core (`Pm4DoodadDuplicationEngine.cs`) with authentic placement telemetry (`PlacementPosition`, `PlacementRotation`, `PlacementScale`, `UniqueId`) and PM4 hull telemetry (`TileX`, `TileY`, `ObjectPartId`, `LinkGroupObjectId`, `HullCentroid`, `HullBoundsMin`, `HullBoundsMax`).
- Configured `Pm4PairingProjectStore.JsonOptions` with `IncludeFields = true` to guarantee accurate JSON round-trip serialization and manifest generation of `Vector3` spatial fields.
- Completely eliminated `CK24` columns from markdown batch summaries and reports in `Pm4ShapeMatchBatchService.cs`, `Collections.cs`, and `SceneGraph.cs`, replacing with authentic `Object ID`, `Position (X, Y, Z)`, `Proposed Candidate`, `Chosen Asset`, `Kind`, `Confidence`, and `Distance (yd)`.
- Purged legacy CK24 terminology from UI glossary and inspector labels across `Pm4WorkbenchService.cs`.
- Implemented `Pm4GlbExporter` in `WoWViewer.Export`: exports each PM4 collision object as an independent named node (`[{TileX}_{TileY}] Obj #{PartId} - {SurfaceClass}`) with double-sided PBR collision materials, sitting seamlessly atop authentic textured/MCCV terrain geometry.
- Wired export actions into `ExportWorkbenchService` and `Pm4WorkbenchSidebarService`.
- Verification: solution builds with 0 errors (`dotnet build WoWViewer.csproj -c Debug -m:1`); UI reference check passes (188/188 controls, 0 errors/warnings); 4/4 focused batch tests pass in `Pm4ShapeMatchBatchServiceTests`. Receipt: `wow-viewer/specs/298-pm4-pairing-telemetry-and-glb-export/evidence/receipt-spec298.md`. Operator visual proof and GLB import check in Blender/3D viewer remain owed.

## 2026-11-06 — Spec 297 Complete (Automated Known-Good Pairing Harvester & Streamlined Matching Workbench)

- Created pure static geometric correlation method `Pm4DoodadDuplicationEngine.FindBestMatchingCandidateHull` in Core: filters candidates by 2D centroid proximity ($\le 15\,\text{yd}$), scale ratio ($\le 4.5$), and vertical bounding sanity ($\le 35\,\text{yd}$ or $2.5\times$ hull height).
- Implemented `Pm4DoodadDuplicationService.HarvestPairingsFromPlacedObjects`: scans resident and donor placed objects across all loaded tiles, correlates them with authoritative PM4 navmesh hulls, auto-registers verified `TaughtPairing` rules, and auto-saves to `<map>.pairings.json` and `<map>.manifest.json`.
- Consolidated the PM4 workbench into a streamlined Cockpit layout in `Pm4WorkbenchSidebarService`:
  - Cockpit status summary pill: `Loaded Tiles: <N> | Placed Models: <M> | Taught Rules: <K>`.
  - Prominent one-click Quick Action Bar: `[ ✦ Harvest Pairings from Placed Models ]` and `[ Apply All Rules Across Map ]`.
  - Clean Active Human Pairing & Curation card: displays model name, PM4 hull with elevation and link group, distance, 90°/180° rotation shortcuts, Z nudge, and one-click multi-tile propagation.
- Tucked low-level raw chunk dumps (`MSHD`, `MSLK`, raw bytes, surface class flags, regional peer lists) into an expandable `Advanced PM4 Forensics & Raw Chunks` drawer across `Pm4WorkbenchService.cs`, `Collections.cs`, and `SceneGraph.cs`, eliminating viewport selection clutter.
- Collapsed secondary manual JSON curation and review queue panels by default (`defaultOpen: false`).
- Added 4 unit tests covering `FindBestMatchingCandidateHull` in `Pm4DoodadDuplicationTests.cs` (10/10 passed).
- Verification: full solution builds with 0 errors; UI reference check passed (188/188 controls, 0 errors/warnings). Receipt: `wow-viewer/specs/297-automated-pairing-harvester-and-streamlined-matching/evidence/receipt-spec297.md`. Operator real-client visual/harvest proof remains owed.

## 2026-11-06 — Spec 296 Complete (PM4 Object Identity, Topological Segmentation & Conflict Decoupling)

- Resolved the fundamental PM4 object identity defect: established that `CK24` is an IEEE-754 32-bit float representing ADT placement $Z$ in yards, not an object ID. On shared plateaus, multiple distinct buildings at the same elevation were previously collapsed into a mega-mesh with self-conflict false positives.
- Introduced `Pm4ObjectIdentity` struct and 64-bit deterministic `UniqueHullId` derived via `(TileX, TileY, ComponentId)`.
- Defaulted topological connectivity segmentation to active with shared `MSVT` vertex and `MSLK` boundary/adjacency link traversal, cleanly isolating click selection and bounding boxes to the clicked building component.
- Isolated surface extraction in `ExtractCompositeSurfaceVertices` and `ExtractTileHullInputs` per-component, preventing tile-wide multi-model polygon cloud unioning.
- Decoupled `HullKey` in `Pm4ShapeMatchAnalysisService` from float `Ck24` and added `EffectiveUniqueHullId` self-conflict guard to `FindHullConflicts` and `FindConflicts`.
- Fixed 180° heading flip in `Pm4DoodadDuplicationService`: compared PCA solved yaw against resident model's existing rotation to pick the collinear $180^\circ$ hemisphere within $\pm 90^\circ$ of authored heading.
- Restored selection bounding boxes: removed the `if (!drewSelectionWireframe)` gate so scene models (M2/MDX/WMO) always draw their animated oriented bounding box; transformed selected PM4 objects via `Pm4OverlayScene.BuildPm4GeometryTransform` and removed inverted tile-band clipping.
- Replaced user-facing `CK24 0x...` labels across review queue rows, Selected Hull labels, manual curation associations, export details, and scene click candidates with `[<Tile>] Obj #<PartId> (Z=<z> yd)`.
- Verification: solution builds with 0 errors; focused unit tests pass 100% across Core, PM4, and ObjectIdentity; `Verify-UiReference.ps1` clean with 188/188 controls (0 errors, 0 warnings). Receipt: `wow-viewer/specs/296-pm4-object-identity-and-segmentation/evidence/receipt-spec296.md`. Operator real-client visual/selection proof remains owed.

## 2026-11-06 — Spec 295 Phases 0–6 Complete (Visual PM4 Shape Library & Pairing)

- Measured signed -X offset on validation tile `development_00_00.pm4` + `development_0_0_obj0.adt`: 15/15 (100.0%) authored WMO placements place the model's footprint center within 1.0 yd of the hull footprint center after calibration, exceeding SC-001's 90% requirement.
- Created `Pm4ShapeLibraryVisualService` and `Pm4ShapeLibraryPanelService`: dual 2D footprint + elevation preview cards, proportional aspect ratio summaries, search and family/kind filtering with selection retention across filter updates, and side-by-side shape comparison.
- Created `Pm4ShapeMatchWorkbenchService`: wires automatic proposal generation, 3D viewport preview with calibrated -X offset, and calibration controls (readout, enable/disable, invert, reset).
- Created `Pm4ShapeMatchReviewService` and `Pm4ShapeMatchReviewPanelService`: human-in-the-loop review queue supporting Accept, Reject (candidate suppression and auto-promotion of next candidate), Replace (with proportion-filtered alternatives from visual library), and opt-in bounded auto-accept mode (SC-002, SC-006).
- Extended `ManualCurationProject` and `ManualCurationProjectFile` to persist pairing decisions and append-only audit history; verified 100% round-trip fidelity across save and reload (SC-003).
- Created `Pm4ShapeMatchBatchService` and `Pm4ShapeMatchBatchPanelService`: multi-tile scope execution, early cancellation preserving accepted decisions, and JSON/Markdown export reconciling matched, low-confidence, unmatched, and conflict counts (SC-004).
- Registered 34 new interactive UI controls in `ui-controls.json` and regenerated `UI-REFERENCE.md`; `Verify-UiReference.ps1` passed with 0 errors and 0 warnings.
- Verification: full solution builds with 0 errors; 21 unit tests passed across analysis, validation tile, visual library, review/persistence round-trip, and batch service. Complete receipt in `wow-viewer/specs/295-pm4-visual-shape-library/evidence/phase0-6-complete-2026-11-06.md`. Operator real-client visual/interaction proof remains owed.

## 2026-11-05 — Spec 292 layer panel, Reconstruction routing, skybox enumeration

- Replaced the fixed L0–L7/A1–A7 toolbar row with a single **Layers** button that opens the Layers workbench panel in the left dock (not a pop-out window); the panel lists resident layers and selects alpha channels.
- The right action bar's **Reconstruction** button now opens `ViewerPanel.Reconstruction`, which draws the map-layer stack; the old Archaeology panel remains a separate destination.
- The Sky panel's skybox combo now enumerates every `.mdx`/`.m2` under `World\Skybox` in addition to LightSkybox DBC names.
- Verification: viewer compile 0 errors; footer tests 30/30; UI registry 156/156 clean.

## 2026-11-05 — Spec 292 bottom-toolbar correction

- Corrected the toolbar to exclude Sky/Fog and show Water/Ocean/Magma/Slime, Doodads/WMOs/Terrain/PM4, separate Tile/Chunk/Cell grid toggles, and Mute; resident-terrain L0–Ln, one A1–An selector per non-base layer, and both opacity sliders. A<n> selects one channel and enables the shared overlay; clicking the active selector disables it. Texture Masks remains absent.
- Liquid families filter MCLQ/MH2O, WL*, and WMO MLIQ meshes independently. The status-bar Mute duplicate is removed and audio starts muted. Alpha/liquid state persists in viewer settings.
- UI registry/reference clean (166 labels, 0 warnings); footer tests 30/30 pass; full solution build 0 errors (266 warnings, redirected output).
- Receipt: [`bottom-toolbar-reversion-2026-11-05.md`](../specs/292-ui-bottom-bar-toggles-audio-sky/evidence/bottom-toolbar-reversion-2026-11-05.md). Operator real-client visual/restart verification remains owed.

## 2026-11-05 — Spec 295 Phase 0–1 core

- Added `Pm4CentroidCalibration` (signed offset, apply-once, invert, reset), `Pm4ShapeMatchAnalysisService` (ranked proposals, confidence, unmatched/low-confidence, batch conflicts, measurement harness), and `Pm4ShapeLibraryHealth`.
- 6 focused tests pass; real-tile calibration measurement and all UI/review/batch work remain.
- Receipt: [`phase0-1-core-2026-11-05.md`](../specs/295-pm4-visual-shape-library/evidence/phase0-1-core-2026-11-05.md).

## 2026-11-05 — Spec 293 donor picker minimap surface and chunk editing

- The donor tile picker now draws each visible tile from the client minimap (`MinimapRenderer.GetTileTexture`) instead of the WDL heightmap preview, so donor selection is recognizable by roads, buildings, and biome.
- Added `PhaseChunkMask` (16×16) to placements and composition; both terrain adapters filter excluded chunks before merge, including exact-grid rotation/mirror via inverse chunk-slot mapping. Masks persist as hex in the layer project.
- The picker exposes a 16×16 chunk grid (All/None/Invert) and whole-map placement at an arbitrary tile offset. Archaeology > Cartography adds a `Viewport chunk edit` mode that toggles the donor chunk under a 3D terrain click.
- Verification: focused `PhaseLayerProjectFileTests` + footer tests 33/33 pass; viewer compile 0 errors; UI registry 166/166 clean. Operator visual proof remains owed.

## 2026-11-05 — Spec 294 Phases 3–4 manual curation and persistence

- Added curation-only placement snapshots and explicit PM4 hull relations, manual position/yaw controls, client Asset Catalog placement, duplicate-safe pair creation, explicit replace/remove actions, and save/reload to a separate per-map curation JSON.
- Persisted authored donor-group transforms through `PhaseLayerProjectFile`; round-trip tests also verify manual placement/association JSON and that a source-map sentinel remains unchanged.
- Verification: focused Spec 294 tests 22 passed; `Verify-UiReference.ps1` 0 errors/0 warnings; full `WowViewer.slnx` build exit 0 (266 warnings, 0 errors). Runtime/visual/save-reopen proof remains operator-owned.
- Receipt: [`phase3-4-manual-curation-persistence-2026-11-05.md`](../specs/294-spatial-curation/evidence/phase3-4-manual-curation-persistence-2026-11-05.md).

## 2026-11-05 — Spec 294 Phase 2 group transform UI

- Added an owned WorkbenchPanels service for arbitrary-angle/translation entry, cloned coordinate preview, validation display, explicit accept/cancel, stale-preview protection, and bounded per-layer undo/redo. Existing quarter-turn rotation remains available.
- Registered the group-transform section/actions and regenerated the UI reference. Focused tests: 13 passed; `Verify-UiReference.ps1`: 0 errors/0 warnings; CrossPlatform viewer build: 0 errors (2 NU1902 package warnings).
- Receipt: [`phase2-group-transform-ui-2026-11-05.md`](../specs/294-spatial-curation/evidence/phase2-group-transform-ui-2026-11-05.md). Real-map preview/accept/cancel/undo/redo proof remains operator-owned. Next: Phase 3 manual object placement and explicit PM4 hull pairing.

## 2026-11-05 — Spec 294 Phase 1 core transform

- Implemented the UI-free donor-group transform preview, angle normalization, centroid/custom-pivot fallback, integer-grid snapping, and structured validation for bounds, collisions, invalid placements, and gaps.
- Added 13 focused unit cases; `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~DonorGroupTransformServiceTests" --nologo` exited 0 (13 passed, 0 failed, 0 skipped).
- Receipt with changed-file list and requirement→evidence table: [`phase1-core-transform-2026-11-05.md`](../specs/294-spatial-curation/evidence/phase1-core-transform-2026-11-05.md). Next: Spec 294 Phase 2 UI; operator runtime proof remains owed.

## 2026-11-05 — Monthly spec and memory-bank consolidation

- Audited every registered active epic/spec task ledger against its receipt references and current tree. Reset 37 unsupported checks across Specs 259, 265, 266, 277, 280, and 288; implementation history is retained and reasons are recorded in the cleanup report.
- Archived the pre-consolidation active context and October progress ledger; current handoff now points to the status router, owning specs, and operator-owned proof only.
- Reconciled the cleanup ledger and missing Spec 293 registry entry; indexed dated memory-bank archives.
- Verification: cleanup inventory/path/receipt audit commands and results are recorded in [`cleanup-2026-11-05.md`](../specs/251-epic-viewer-ux-and-code-health/evidence/cleanup-2026-11-05.md).
