# Spec 197 Tasks: UI Workspace Profiles, Editor Mode Integration, PM4 Mouse Inspection, Multi-Client Map Staging & MoP 5.0.1 ADT Pipeline

## Phase 1: Workspace Profile Switcher & Editor Tab Integration
- [ ] **T101**: Define `ViewerWorkspaceMode` (`Viewer`, `Editor`, `Archaeology`) in `ViewerApp.cs`.
- [ ] **T102**: Add the Top-Bar Mode Switcher button group (`[Viewer]`, `[Editor]`, `[Archaeology]`) in `DrawMenuBar()` / `ViewerApp.cs`.
- [ ] **T103**: Expose the missing `Editor` tab button (`DrawTopTabButton(WorkbenchTab.Editor, "Editor")`) in `ViewerApp_Sidebars.cs`.
- [ ] **T104**: Filter right-sidebar tabs (`Quick`, `Inspect`, `Scene`, `Utilities`, `Editor`, `Archaeology`) according to active `ViewerWorkspaceMode`.
- [ ] **T105**: Test profile switching across Viewer, Editor, and Archaeology modes.

## Phase 2: Menu Audit & "MK Dataset" Legacy Purge
- [ ] **T106**: Audit all items in `DrawMainMenu()` and remove dead "MK Dataset" / `MkDatasetHarvester` references.
- [ ] **T107**: Consolidate ML and offline export options under `Tools > Offline Data / Conversion`.
- [ ] **T108**: Verify clean compilation and zero leftover references to `MkDataset`.

## Phase 3: Direct PM4 Viewport Mouse Raycast & Selection
- [ ] **T109**: Implement bounding-box and triangle raycast helper for PM4 geometry in `WorldScene.cs`.
- [ ] **T110**: Wire PM4 candidates into `TryHandleSceneClickSelection` in `ViewerApp_ClickSelection.cs`.
- [ ] **T111**: Enable Inspector synchronization when a PM4 object is picked via viewport click.
- [ ] **T112**: Test mouse picking of PM4 objects, surfaces, and collision bounding boxes.

## Phase 4: Multi-Client Map Staging & Restoration Library
- [ ] **T113**: Create `MultiClientMapStagingService.cs` supporting multiple simultaneous `IDataSource` client mounts.
- [ ] **T114**: Implement cross-client chunk slice extraction and staging into a unified restoration target.
- [ ] **T115**: Write unit tests for multi-client mounting and chunk copying in `MultiClientMapStagingTests.cs`.

## Phase 5: Ghidra Analysis & 4.3.4–5.1 MoP ADT Pipeline
- [ ] **T116**: Connect to Ghidra session (`localhost:8080`) on `WoW.exe` 5.0.1.15464.
- [ ] **T117**: Decompile and extract `CMapChunk`, `CMapTile`, `MHID`, `MDID`, `MCXH`, and WMO terrain seam blending routines.
- [ ] **T118**: Implement `MopAdtChunkParser.cs` in `WowViewer.Core.IO`.
- [ ] **T119**: Wire 4.3.4/5.0.1 split ADTs, height texture blending, and WMO blending in `StandardTerrainAdapter` and `TerrainRenderer`.
- [ ] **T120**: Write unit tests for `MopAdtChunkParser` (100% green).
