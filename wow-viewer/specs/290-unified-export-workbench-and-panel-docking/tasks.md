# Task Checklist: Spec 290 — Unified Export Workbench & Panel Docking

## Phase 1: Service Architecture & UI Manager Wiring

- [x] **T001**: Add `ViewerPanel.Exports` enum value to `WoWViewer/Workbench/WorkbenchTab.cs`.
- [x] **T002**: Create `ExportWorkbenchService.cs` in `WoWViewer/Workbench/Services/Export/` implementing `DrawExportPanelContent()`.
- [x] **T003**: Expose `ExportWorkbenchService ExportWorkbench { get; }` on `IViewerAppHost` and instantiate in `ViewerApp`.
- [x] **T004**: Add `IconKind.Export` and `ActionExports` icon button in `ViewerActionBarsService.DrawRightBar()`.
- [x] **T005**: Wire `case ViewerPanel.Exports:` dispatch into `WorkbenchPanelsService.cs`.
- [x] **Gate A (Phase 1 Verification)**:
  - Project builds cleanly with 0 errors.

---

## Phase 2: Consolidating PM4, World, and Dataset Exports

- [x] **T006**: Implement PM4 Navigation & Collision Exports section in `ExportWorkbenchService.cs` (Objects JSON, OBJ Set, LLM Bundle, Visible Overlay Report, Zarr v60 archive).
- [x] **T007**: Implement World & Terrain Exports section in `ExportWorkbenchService.cs`.
- [x] **T008**: Implement Model & Asset Exports section in `ExportWorkbenchService.cs`.
- [x] **T009**: Implement Archaeology & Datasets section in `ExportWorkbenchService.cs`.
- [x] **T010**: Add "Open Export Workbench" button in `Pm4WorkbenchSidebarService.cs` and replace redundant inline exports.
- [x] **Gate B (Phase 2 Verification)**:
  - All export actions function properly from the new workbench without regressions.

---

## Phase 3: Bottom Destination Tool Window Repositioning

- [x] **T011**: Update `ViewerActionBarsService.DrawToolWindows()` destination window position from top-right to lower-left quadrant adjacent to chat.
- [x] **T012**: Verify that opening `Editor` or `Archaeology` does not overlap or obscure `UiRegion.RightDock`.
- [x] **Gate C (Phase 3 Verification)**:
  - Destination window displays cleanly beside chat without obscuring right-dock panels.

---

## Phase 4: UI Reference Catalog & Regression Verification

- [x] **T013**: Register new controls in `wow-viewer/eng/ui/ui-controls.json` and `New-UiControlRegistry.ps1`.
- [x] **T014**: Run `Verify-UiReference.ps1` and verify 0 errors, 0 warnings.
- [x] **T015**: Run full solution build: `dotnet build WoWViewer.CrossPlatform.csproj -c Debug --no-dependencies --no-restore /p:GenerateAppHost=false /nodeReuse:false /m:1`.
- [x] **Gate D (Phase 4 Verification)**:
  - Receipt written to `specs/290-unified-export-workbench-and-panel-docking/evidence/final-receipt.md`.
