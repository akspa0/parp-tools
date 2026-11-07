# Spec 299 — All-Tile PM4 Rule Propagation, Async Progress Flow, Donor/Cross-Build Missing Data Harvest & LK ADT Export

## 1. Problem Statement

1. **Camera-Window Tile Limitation**: The PM4 propagation action (`ApplyAllTaughtPairingsAcrossAllTiles`) currently only evaluates tiles loaded into the camera window (`worldScene.Pm4Overlay._pm4TileObjects`, typically ~86 tiles out of 616 for `development`). Unopened map tiles remain unpopulated and ignored.
2. **Main Thread Blocking & UI Freezes**: Executing multi-tile pairing propagation synchronously on the ImGui render thread freezes the entire application for tens of seconds, causing OS hitching and preventing user feedback or cancellation.
3. **Missing Automated LK ADT Materialization**: After rules are applied, users must navigate secondary menus to export patched ADTs. There is no automated, end-to-end pipeline that carries whole-map rule propagation directly into exportable Wrath / 3.3.5 LK ADTs ready for Noggit, client, or tool inspection.
4. **Unfilled PM4 Hulls Without Donor Harvest**: Many PM4 collision hulls on unauthored tiles do not match any of the currently taught pairing rules. Placed models from donor layers, other maps, or historical build datasets are not automatically harvested to fill in these missing meshes during map-wide propagation.

## 2. User Requirements

- **FR-001 (Map-Wide All-Tile PM4 Evaluation)**:
  - Discover all candidate PM4 files available for the active map (e.g. all 616 PM4 files for `development`).
  - For loaded tiles, evaluate their resident `Pm4OverlayObject` collision hulls.
  - For unopened/unloaded tiles, parse PM4 files from the data source on-the-fly (`Pm4File.Read` + `Pm4OverlayGeometry`) without requiring camera-window viewport residency.
- **FR-002 (Asynchronous Non-Blocking Execution with Live Progress Bar)**:
  - Run the whole-map propagation on a background thread (`Task.Run`) to keep the UI at full interactive framerate.
  - Provide a live ImGui progress bar showing:
    - Overall completion percentage (`ProcessedTiles / TotalTiles`).
    - Current tile coordinate (`development_XX_YY`).
    - Live counters of evaluated hulls and spawned placements.
    - An immediate `[ Cancel ]` button to gracefully abort the operation and preserve progress.
- **FR-003 (Cross-Build & Donor Harvest for Missing Data Fill-In)**:
  - Before or during propagation, scan available donor sources (active donor layers in `worldScene.Layers`, donor ADTs in data source/archive, and persistent prior knowledge / Rosetta databases).
  - Automatically harvest associations and synthesize placements for unoccupied PM4 hulls that lack direct user-taught rules, ensuring complete zone restoration.
- **FR-004 (Automatic LK ADT Materialization & Export)**:
  - Upon completion of map-wide propagation, automatically export all populated and patched tiles as monolithic Wrath / 3.3.5 LK ADTs into `output/patched_adts/<mapName>/`.
  - For tiles with existing base ADTs, preserve terrain geometry, alpha maps, and textures via `AdtPlacementEditor`, adding new model placements.
  - For blank/unauthored tiles, synthesize monolithic LK ADTs via `BlankAdtFactory` and `LkAdtWriter` with default ground terrain so models are physically loadable and verifiable in external ADT inspectors and clients.
- **FR-005 (UI Integration & Architecture Compliance)**:
  - Wire the whole-map action into `Pm4WorkbenchSidebarService` and `ExportWorkbenchService`.
  - Follow AGENTS.md §10 (God-Class Freeze): isolate all propagation logic into a dedicated service `Pm4MapWidePropagationService` without expanding `Pm4DoodadDuplicationService` beyond the 2,000 line budget.
  - Validate with full build (0 errors), unit tests, and `Verify-UiReference.ps1`.

## 3. Success Criteria

- **SC-001**: Clicking the whole-map propagation button initiates an async background task across all 616 tiles without freezing the UI or dropping frame presentation.
- **SC-002**: Live progress bar accurately displays percentage, tile count, current tile name, and placement counter, with cancellation stopping the task cleanly.
- **SC-003**: Unoccupied PM4 hulls without user-taught rules are populated from available donor layers, donor ADTs, or prior knowledge.
- **SC-004**: Patched LK ADTs (`.adt`) are automatically generated in `output/patched_adts/<mapName>/` and contain valid `MDDF`/`MODF` placement chunks matching the PM4 hulls.
- **SC-005**: Full solution builds with 0 errors, unit tests pass 100%, and UI reference verification succeeds.

