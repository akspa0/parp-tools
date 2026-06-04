# Tasks: Viewer Shell Usability

**Input**: `wow-viewer/specs/044-viewer-shell-usability/spec.md`

## Phase 1: Shell Activation and Discoverability

**Goal**: Turn the dormant dockable shell into the default viewer shell and make map discovery visible after client load.

- [x] T001 [US1] Render the dockspace host from `ViewerApp.DrawUI()` before shell panels are drawn.
- [x] T002 [US1] Default `UseDockspaceUi` to `true`, persist it correctly, and upgrade old layout settings into the dockable shell path.
- [x] T003 [US1] Disable fixed sidebar splitters while dockable shell mode is active.
- [x] T004 [US2] Auto-open `World Maps` when discovered map count transitions from `0` to `>0`.
- [x] T005 [US1] Keep `View > Reset Shell Layout` aligned with the dockable shell default.

## Phase 2: Menu Declutter

**Goal**: Keep old data/conversion tools reachable without presenting them as primary file-load actions.

- [x] T006 [US3] Remove `Open MK Dataset...` and `Open Zarr Dataset...` from the `File` menu.
- [x] T007 [US3] Add a dedicated `Tools > Offline Data / Conversion` submenu for MK/Zarr, dataset build, training, texture transfer, and map/WMO conversion utilities.
- [x] T008 [US1] Expose a `View > Dockable Shell Panels` toggle so the shell mode is controllable and persisted.

## Phase 3: Deferred Cursor Diagnostic

**Goal**: Track the cursor-model requirement as a separate runtime slice instead of silently dropping it.

- [ ] T009 [US4] Identify client-era cursor asset ownership and version routing for MDX-era versus M2-era clients.
- [ ] T010 [US4] Implement a model-backed cursor render path that follows the game-era asset selection rules.
- [ ] T011 [US4] Add focused proof that a model-render failure also suppresses the cursor model, making the regression visually obvious.

## Validation

- [x] T012 Run `dotnet build i:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug`.
- [x] T013 Update `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md` with the landed shell behavior plus the deferred cursor-model gap.
