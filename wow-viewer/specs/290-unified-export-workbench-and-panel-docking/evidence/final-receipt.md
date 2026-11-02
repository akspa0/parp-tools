# Spec 290 — Unified Export Workbench & Panel Docking Receipt

Date: 2026-11-01
Owner: Antigravity Assistant

## 1. Files Changed

- `wow-viewer/src/viewer/WoWViewer/Workbench/WorkbenchTab.cs`:
  - Added `ViewerPanel.Exports` enum value.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Export/ExportWorkbenchService.cs`:
  - Created authoritative consolidated Export & Report Workbench service.
  - Implemented PM4 Navigation & Collision exports: JSON, OBJ Set, LLM Bundle, Visible Overlay Report, Zarr v60 archive, Selected PM4 object JSON copy.
  - Integrated World & Terrain exports via `_host.MainMenuBar.DrawWorldExportContent()`.
  - Integrated Model & Asset mesh exports via `_host.MainMenuBar.DrawModelExportContent()`.
  - Integrated Archaeology & ML dataset exports (Synthesized Minimap dialog, VLM ML Dataset exporter).
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/IViewerAppHost.cs`:
  - Added `ExportWorkbenchService ExportWorkbench { get; }`.
- `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` & `ViewerApp_Host.cs`:
  - Instantiated `_exportWorkbench` and implemented interface property.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs`:
  - Added `IconKind.Export` and rendered upward-arrow tray pictogram in `DrawActionBarIcon()`.
  - Added `ActionExports` button to right action rail (`DrawRightBar()`).
  - Repositioned destination tool window (`Editor`, `Archaeology`, `Pages`) from top-right `(rightX, rightY)` overlapping RightDock to the lower-left workspace adjacent to the chat dock (`destX`, `destY`).
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs`:
  - Wired `case ViewerPanel.Exports:` to `_host.ExportWorkbench.DrawExportPanelContent()`.
- `wow-viewer/src/viewer/WoWViewer/Terrain/Pm4/UI/Pm4WorkbenchSidebarService.cs`:
  - Added `openExports` action support to `DrawOverlaySettings()`.
- `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Pm4Workbench/Pm4WorkbenchService.cs` & `Pm4WorkbenchService.Exports.cs` & `Pm4WorkbenchService.SceneGraph.cs`:
  - Added "Open Export Workbench" button in PM4 overlay sidebar and selection tab.
  - Made `SaveMergedPlacementsToZarr()`, `RunPm4Export()`, and `ExportPm4OverlayReport()` accessible to `ExportWorkbenchService`.
- `wow-viewer/eng/ui/New-UiControlRegistry.ps1`, `wow-viewer/eng/ui/ui-controls.json`, `wow-viewer/docs/WoWViewer/UI-REFERENCE.md`:
  - Registered `exports-rail-icon` ("Exports") under Right action bar.

## 2. Verification Commands and Output

1. **UI Reference & Control Registry Audit**:
   ```powershell
   pwsh -File "I:\parp\parp-tools\wow-viewer\eng\ui\Verify-UiReference.ps1"
   ```
   **Output**:
   ```
   UI_REFERENCE_CHECK inventory_rows=138 distinct_labels=128 registry_entries=128
   UI_REFERENCE_CHECK PASSED errors=0 warnings=0
   ```
   Exit status: **0**.

2. **Full Compilation & Packaging**:
   ```powershell
   dotnet build I:/parp/parp-tools/wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj -c Debug --no-dependencies --no-restore /p:GenerateAppHost=false /nodeReuse:false /m:1
   ```
   **Output**:
   ```
   131 Warning(s)
   0 Error(s)
   Time Elapsed 00:00:30.03
   ```
   Exit status: **0**.

## 3. Acceptance Criteria Verification Table

| Acceptance Criterion | Description | Evidence / Output |
|---|---|---|
| **AC-101** | Unified Exports Panel on Right Action Rail | `ViewerActionBarsService.DrawRightBar()` renders `ActionExports` with `IconKind.Export` icon. Clicking toggles `ViewerPanel.Exports` in `UiRegion.RightDock`. |
| **AC-102** | Consolidates PM4, World, Model, Dataset Exports | `ExportWorkbenchService.DrawExportPanelContent()` hosts SectionHeaders for PM4 Collision & Nav, World & Terrain Geometry, Model Meshes, and Archaeology & ML Datasets. |
| **AC-103** | Preserves Core Exporter Backend Logic | All underlying calls delegate directly to `Pm4WorkbenchService`, `MainMenuBar`, and `DatasetExportDialogsService` without rewriting I/O or reader algorithms. |
| **AC-201** | Destination Window Repositioned | Destination window (`Editor`, `Archaeology`) positions at `(destX, destY)` adjacent to chat dock (`rail + 12f * scale + chatWidth`), clearing `UiRegion.RightDock` with $>500$ px horizontal clearance. |
| **AC-202** | Independent Window Sizing & Dragging | ImGui tool window permits free moving and resizing without closing or resizing right-dock inspection panels. |
| **AC-301** | Architecture & God-Class Compliance | AGENTS.md §10 satisfied: `ExportWorkbenchService` is an owned service class; 0 new fields or methods added to `WorldScene` or `ViewerApp` body; all files $\le 2000$ lines. |
| **AC-302** | UI Reference & Registry Sync | 138 inventory rows, 128 registry entries verified; `Verify-UiReference.ps1` exit code 0. |
| **AC-303** | Build Passes Cleanly | `dotnet build WoWViewer.CrossPlatform.csproj` exit code 0. |

