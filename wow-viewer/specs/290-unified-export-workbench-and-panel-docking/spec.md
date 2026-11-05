# Feature Specification: Spec 290 — Unified Export Workbench & Panel Docking

**Owner**: Epic 251 (Viewer UX, Shell & Code Health)  
**Origin**: Operator prompt (2026-11-01) regarding:
1. Fragmented export workflows: PM4 model and reporting outputs (OBJ sets, JSONs, LLM evidence bundles, overlay summaries, Zarr v60 archives, selected object OBJs) are scattered across multiple sub-menus, sidebars, and dialogs.
2. Building a dedicated **Export & Reports** workbench panel on the right sidebar, moving all export and report actions from across the UI into one clean, authoritative home.
3. Repositioning the bottom-bar destination windows (`Editor`, `Archaeology`) so they pop up in the lower portion of the screen (adjacent to the chat frame) instead of spawning at the top-right and obscuring right-dock inspection panels (PM4, Inspector, World).

**Status**: Draft / Specified (2026-11-01)

---

## 1. Executive Summary & Problem Analysis

### 1.1 Fragmented Export Surfaces (Violating Spec 223 FR-3)
Currently, export and report actions are fragmented across multiple disconnected locations:
- In `Pm4WorkbenchSidebarService.cs`: "Export Report" button near the top.
- In `Pm4WorkbenchService.cs`: "Selected PM4 Object Details & Exports" and "PM4 Exports" collapsible sections with 5 separate buttons (`Export Objects JSON...`, `Export OBJ Set...`, `Export LLM Evidence Bundle...`, `Export Visible Overlay Report`, `Save Merged Placements to Zarr v60`).
- In `AssetCatalogView.cs` and `MainMenuBarService.Content.cs`: Model glTF/OBJ exports.
- In `NavigatorPanelService.Sidebars.cs`: World / Map exports.
- In `ArchaeologyPanelService.cs`: Synthesized minimap export.
- In `DatasetExportDialogsService.cs`: VLM and ML dataset exports.

Per AGENTS.md §11 and Spec 223 FR-3, every data surface must have exactly one authoritative home. An operator inspecting PM4 navmeshes or preparing client assets currently has to hunt across tabs, sub-tabs, and bottom drawers to find output tools.

### 1.2 Destination Tool Window Screen Collision
In `ViewerActionBarsService.cs` (lines 146–150):
```csharp
float rightX = Math.Max(rail, display.X - rail - 430f * scale - 12f * scale);
float rightY = top + mapHeight + 12f * scale;
if (_host.ActiveTopTab != WorkbenchTab.Viewer)
    DrawToolWindow("destination", _host.ActiveTopTab.ToString(), new(rightX, rightY), new(500f, 620f),
        DrawDestinationContent);
```
When the user clicks `Editor` or `Archaeology` on the bottom toolbar, the resulting tool window is placed at `(rightX, rightY)`—the exact screen coordinates occupied by the right dock (`UiRegion.RightDock`). This completely covers and obscures whichever right-dock inspector was active (`PM4`, `Inspector`, `World`).

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Dedicated Unified Export Workbench Panel (Priority: P1)
As a technical artist or reverse engineer exporting PM4 navmesh models, world geometry, and placement ledgers, I want a single dedicated **Exports & Reports** panel on the right sidebar rail so that all export functionality is cleanly organized in one place.

- **AC-101**: A new `Exports` icon button is added to the right action bar (`ViewerActionBarsService.DrawRightBar`) alongside `Inspector`, `World`, `Gossip`, and `PM4`.
- **AC-102**: Clicking `Exports` opens `ViewerPanel.Exports` in the right dock (`UiRegion.RightDock`) with an owned service class `ExportWorkbenchService.cs` conforming to AGENTS.md §10.
- **AC-103**: The panel organizes all exports into categorized, collapsible `SharedUiWidgets.SectionHeader` groups:
  1. **PM4 Navigation & Collision Exports**:
     - `Export Objects JSON...`: Exports all loaded PM4 objects across tiles to JSON.
     - `Export OBJ Set...`: Exports complete tile OBJ models and discrete object OBJ files with manifest.
     - `Export LLM Evidence Bundle...`: Exports JSON, Markdown, and topological SVG diagrams.
     - `Export Visible Overlay Report`: Exports text diagnostics and surface summaries.
     - `Save Merged Placements to Zarr v60`: Archives resident ADT and magnetized PM4 placements.
     - *When an object is selected*: `Export Selected Object OBJ` and `Copy Object JSON`.
  2. **World & Terrain Exports**:
     - Map / ADT terrain mesh export.
     - World model and doodad placement tables (`MODF` / `MDDF`).
  3. **Model & Asset Exports**:
     - Selected M2/WMO 3D model export.
  4. **Archaeology & Dataset Synthesis**:
     - Synthesized minimap export dialog.
     - VLM / Terrain ML dataset export.
- **AC-104**: Redundant export buttons inside the PM4 inspection sidebar (`Pm4WorkbenchSidebarService.cs` and `Pm4WorkbenchService.cs`) are streamlined, linking directly to the authoritative Exports dock.

### User Story 2: Non-Colliding Bottom Destination Windows (Priority: P1)
As a user working with the editor or archaeology tools alongside 3D scene inspection, I want destination tool windows (`Editor`, `Archaeology`) anchored in the lower portion of the screen so that the right sidebar remains unobscured.

- **AC-201**: The destination tool window (`Editor`, `Archaeology`) is re-anchored to the lower-left / lower-center screen quadrant above the bottom action bar and adjacent to the chat frame (`UiRegion.Chat`).
- **AC-202**: Opening `Editor` or `Archaeology` does not obscure, overlap, or close the right dock (`PM4`, `Exports`, `Inspector`, `World`). Both surfaces can be viewed and operated simultaneously.
- **AC-203**: Window coordinates and bounds dynamically adapt to display resolution and UI scale without clipping off-screen.

### User Story 3: UI Standardization & Verification (Priority: P2)
As a maintainer of the viewer UI system, I want all new controls and layouts registered in the UI control catalog so that `Verify-UiReference.ps1` passes with zero errors and zero warnings.

- **AC-301**: All new UI buttons and controls in `ExportWorkbenchService` are registered in `eng/ui/ui-controls.json`.
- **AC-302**: `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` passes with 0 errors and 0 warnings.
- **AC-303**: All new source files remain $\le 2,000$ lines.

### Documentation correction — 2026-11-05

AC-301's inventory path is corrected to the existing canonical `eng/ui/ui-controls.json`; the former `ui-reference-inventory.json` path does not exist. This corrects the registry filename only and does not change the accepted UI behavior or scope.
