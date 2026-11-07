# Tasks 299 — All-Tile PM4 Rule Propagation, Async Progress Flow & LK ADT Export

Receipt: `evidence/receipt-spec299.md`

## Phase 1 — Map-Wide PM4 Enumerator & Unloaded Tile Hull Parser (FR-001)

- [x] T001 [Service] Create `Pm4MapWidePropagationService.cs` in `WoWViewer.Workbench.Services.Pm4Workbench`.
- [x] T002 [Service] Implement whole-map PM4 tile discovery from `IDataSource.GetFileList(".pm4")` resolving tile coordinates and total file count.
- [x] T003 [Service] Implement background parsing of unopened/unloaded PM4 tiles (`Pm4File.Read` + hull extraction) to generate valid 3D candidate bounding boxes without requiring viewport residency.

## Phase 2 — Background Async Execution & Progress Telemetry (FR-002)

- [x] T004 [Service] Implement thread-safe progress state in `Pm4MapWidePropagationService` (`IsRunning`, `ProcessedTiles`, `TotalTiles`, `CurrentTileName`, `Phase`, `PlacementsAdded`, `CancellationTokenSource`).
- [x] T005 [UI] Update `Pm4WorkbenchSidebarService.cs` to render an interactive `ImGui.ProgressBar`, phase and tile readouts, and `[ Cancel ]` button when background propagation is active.
- [x] T006 [UI] Replace the synchronous `Apply All Rules Across Map` button with an invocation of `StartMapWidePropagationAsync`.

## Phase 3 — Cross-Build & Donor Missing Data Harvest (FR-003)

- [x] T007 [Service] Implement donor and cross-build placement scanning: check active donor phase layers in `worldScene.Layers`, donor ADT placements, and `Pm4PriorKnowledgeDatabase` for unoccupied PM4 hulls lacking direct human-taught pairings.
- [x] T008 [Service] Synthesize resolved placements for unmatched hulls with authentic world transforms and unique IDs.

## Phase 4 — Automated LK ADT Export Pipeline (FR-004)

- [x] T009 [Export] Implement automatic LK ADT export at the conclusion of map-wide propagation, targeting `output/patched_adts/<mapName>/`.
- [x] T010 [Export] For tiles with existing base ADTs, apply `AdtPlacementEditor` with additions/moves; for blank/unauthored tiles, generate monolithic LK ADTs via `BlankAdtFactory` and `LkAdtWriter`.

## Phase 5 — Verification, UI Reference & Receipts (FR-005, SC-001..SC-005)

- [x] T011 [Tests] Add unit test suite in `Pm4MapWidePropagationTests.cs` verifying tile enumeration, progress reporting, and placement generation.
- [x] T012 [Verification] Run full solution build (`dotnet build WoWViewer.csproj -c Debug -m:1`) and verify 0 errors.
- [x] T013 [Verification] Run unit tests and verify 100% pass.
- [x] T014 [Verification] Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
- [x] T015 [Receipt] Record verification commands, exit status, and criterion-to-evidence matrix in `evidence/receipt-spec299.md`.

