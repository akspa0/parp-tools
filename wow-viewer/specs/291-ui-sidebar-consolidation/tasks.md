# Task Checklist: Spec 291 — Dual Sidebar Docking, Destination Integration & UI Harmonization

## Phase 1: Dual Dock Core Architecture (`UiRegion.LeftDock`)

- [x] **T001**: Add `UiRegion.LeftDock` to `UiRegion` enum and compute its bounds and chat displacement in `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs`.
- [x] **T002**: Define `LeftPanel` enum (`Workspace, Files, Maps, Assets, Taxi, Capture, Diagnostics, Options, Layers`) in `WorkbenchTab.cs`.
- [x] **T003**: Add `_leftDockOpen`, `_leftDockWidth`, and `ActiveLeftPanel` state and layout persistence to `ViewerUiManager.cs`.
- [x] **T004**: Implement `DrawLeftDock()` with right-edge interactive splitter in `ViewerUiManager.cs`.
- [x] **T005**: Wire left action bar icons (`ActionWorkspace`, `ActionFiles`, `ActionMaps`, `ActionAssets`, `ActionTaxi`, `ActionCapture`, `ActionDiagnostics`) and `ActionOptions` to toggle `UiRegion.LeftDock` in `ViewerActionBarsService.cs`.
- [x] **T006**: Dispatch `DrawLeftDockContent(LeftPanel)` to existing host providers (`Workspace`, `Files`, `Maps`, `Assets`, `Taxi`, `Capture`, `Diagnostics`, `Options`, `Layers`) in `WorkbenchPanelsService.cs`.
- [x] **T007**: Retire floating `DrawToolWindow` calls for left tools in `ViewerActionBarsService.cs`.
- [x] **Gate A (Phase 1 Verification)**:
  - Solution builds cleanly with 0 errors. Left rail icons toggle the Left Dock without opening floating windows. (Receipt: `evidence/final-receipt.md`)

---

## Phase 2: Destination Integration into Sidebars

- [x] **T008**: Remove floating `ActionWindow_destination` from `ViewerActionBarsService.cs`.
- [x] **T009**: Integrate Editor workbench rendering into `UiRegion.RightDock` when `ActiveTopTab == WorkbenchTab.Editor`.
- [x] **T010**: Integrate Archaeology workbench rendering into `UiRegion.RightDock` when `ActiveTopTab == WorkbenchTab.Archaeology`.
- [x] **T011**: Remove the redundant `Pages` button from `ViewerFooterLayout.cs`, `ViewerActionBarsService.cs`, and `ViewerUiManager.cs`.
- [x] **T012**: Ensure switching destinations (`Viewer`, `Editor`, `Archaeology`) via bottom bar smoothly updates the docked workbench without closing the sidebar or resetting widths.
- [x] **Gate B (Phase 2 Verification)**:
  - Editor and Archaeology workbenches render directly inside `UiRegion.RightDock`. No floating lower-left window appears. (Receipt: `evidence/final-receipt.md`)

---

## Phase 3: Export & Navigation Deduplication

- [x] **T013**: Streamline `ViewerPanel.World` from 4 to 3 pages (`["Objects", "Placements", "LOD & Budget"]`), replacing page 3 ("Exports") with a direct link button to `ViewerPanel.Exports`.
- [x] **T014**: In `ViewerPanel.Inspector` Model Info, replace duplicate model export buttons with a link to `ViewerPanel.Exports`.
- [x] **T015**: Expand `ExportWorkbenchService.cs` to incorporate Terrain File Operations (Alpha, Heightmap 257, MCCV exports and imports) and DAT export controls under "World & Terrain Geometry".
- [x] **T016**: Remove duplicate `DrawCapturePanelContent()` embed from Archaeology Page 3 (Playback).
- [x] **T017**: Remove vestigial `Inspect PM4 in Viewer` button from Archaeology header.
- [x] **Gate C (Phase 3 Verification)**:
  - All export options are unified inside `ViewerPanel.Exports`. Duplicate exports in World, Inspector, and Editor are eliminated. (Receipt: `evidence/final-receipt.md`)

---

## Phase 4: Design System Cleanup & Registry Sync

- [x] **T018**: Standardize styling in `AssetCatalogView.cs` (replace hardcoded navy/gold border with `UiTokens`).
- [x] **T019**: Standardize styling in `ArchaeologyPanelService.PhaseLayers.cs` (replace hardcoded orange text with `UiTokens.Brass`).
- [x] **T020**: Standardize button styling in `EditorPanelsService.cs` (replace hardcoded blue buttons with `SharedUiWidgets.ActionButton`).
- [x] **T021**: Update `wow-viewer/eng/ui/ui-controls.json` with updated control paths and removed redundant controls (`Pages`, duplicate exports).
- [x] **T022**: Run `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
- [x] **T023**: Run full solution build `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore` with 0 errors.
- [x] **Gate D (Phase 4 Verification)**:
  - Receipt written to `specs/291-ui-sidebar-consolidation/evidence/final-receipt.md`.
