# Spec 291 Final Verification Receipt

## 1. Overview
- **Spec**: Spec 291 — Dual Sidebar Docking, Destination Integration & UI Harmonization
- **Owner**: Epic 251 (Viewer UX, Shell & Code Health)
- **Date**: 2026-11-02
- **Status**: Implemented & Verified

---

## 2. Files Changed

| Path | Summary of Changes |
| :--- | :--- |
| `wow-viewer/src/viewer/WoWViewer/Workbench/WorkbenchTab.cs` | Added `LeftPanel` enum (`Workspace, Files, Maps, Assets, Taxi, Capture, Diagnostics, Options, Layers`). |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs` | Added `UiRegion.LeftDock` region, width/visibility layout calculations, and non-colliding `chatDockX` offset. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs` | Added `_leftDockOpen`, `_leftDockWidth`, `ActiveLeftPanel`, `OpenLeftDock`, `ToggleLeftDock`, `CloseLeftDock`, `DrawLeftDock` with interactive splitter, updated `DrawRightDock` to host Editor/Archaeology content, and removed `bottom-pages`. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs` | Bumped layout version to 7 (`ui-layout-v7.json`), persisting `LeftDockOpen`, `LeftDockWidth`, and `LeftPanel`. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs` | Re-routed left rail icons to toggle `UiRegion.LeftDock`, removed redundant `bottom-pages` action, and cleared floating window spawns. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/WorkbenchPanels/WorkbenchPanelsService.cs` | Added `DrawLeftContent(LeftPanel)`, streamlined `ViewerPanel.World` to 3 pages (replacing duplicate Exports tab with link button), streamlined Model Info in `ViewerPanel.Inspector` with link button. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Export/ExportWorkbenchService.cs` | Integrated Terrain File Operations (Alpha, Heightmap 257, MCCV) and DAT exports under World Geometry exports. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Archaeology/ArchaeologyPanelService.Sidebars.cs` | Removed vestigial `Inspect PM4 in Viewer` button and duplicate camera capture panel from Playback. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/Archaeology/ArchaeologyPanelService.PhaseLayers.cs` | Replaced hardcoded orange color with `UiTokens.Brass`. |
| `wow-viewer/src/viewer/WoWViewer/Workbench/Services/EditorPanels/EditorPanelsService.cs` | Replaced hardcoded blue buttons with `UiTokens.BrassDim`. |
| `wow-viewer/src/viewer/WoWViewer/Catalog/AssetCatalogView.cs` | Replaced hardcoded navy/gold border with `UiTokens.Surface1` and `UiTokens.Brass`. |
| `wow-viewer/eng/ui/ui-controls.json` | Removed `bottom-pages`, added `open-model-exports` and `open-world-exports`. |
| `wow-viewer/eng/ui/New-UiControlRegistry.ps1` | Added registry help metadata for model and world export redirect buttons. |
| `wow-viewer/docs/WoWViewer/UI-REFERENCE.md` | Regenerated UI reference documentation. |
| `wow-viewer/tests/WowViewer.Core.Tests/ViewerFooterLayoutTests.cs` | Removed `bottom-pages` from footer layout unit tests. |
| `wow-viewer/tests/WowViewer.Core.Tests/UiControlDocumentationTests.cs` | Updated expected action bar icons list to include `Exports`. |

---

## 3. Verification Commands & Exit Status

| # | Command | Working Directory | Exit Code | Real Output Summary |
| :- | :--- | :--- | :-: | :--- |
| 1 | `pwsh -ExecutionPolicy Bypass -File wow-viewer/eng/ui/Verify-UiReference.ps1` | `I:/parp/parp-tools` | **0** | `UI_REFERENCE_CHECK inventory_rows=139 distinct_labels=129 registry_entries=129`<br>`UI_REFERENCE_CHECK PASSED errors=0 warnings=0` |
| 2 | `dotnet build wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj -c Debug --no-restore` | `I:/parp/parp-tools` | **0** | Build succeeded with 0 errors. Post-build UI reference target executed and passed. |
| 3 | `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~ViewerFooterLayout" --no-restore` | `I:/parp/parp-tools` | **0** | Passed: 17, Failed: 0, Skipped: 0, Total: 17. |
| 4 | `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~UiControlDocumentation" --no-restore` | `I:/parp/parp-tools` | **0** | Passed: 5, Failed: 0, Skipped: 0, Total: 5. |

---

## 4. Acceptance Criteria to Real Output Mapping

| Acceptance Criterion | Verification Method / Real Output | Status |
| :--- | :--- | :--- |
| **AC-101**: Introduce `UiRegion.LeftDock` bounded vertically between top menu and bottom bar. | `UiLayout.cs` defines `LeftDock` bounds; `GetRegionRect(UiRegion.LeftDock)` returns rect from top bar to bottom bar. | **PASS** |
| **AC-102**: `UiRegion.LeftDock` features interactive edge splitter with persisted width. | `ViewerUiManager.cs` `DrawLeftDock()` implements splitter with clamp (260px - 720px); persisted in `ui-layout-v7.json`. | **PASS** |
| **AC-103**: Clicking Left Action Rail icons toggles/activates tool in `UiRegion.LeftDock`. | `ViewerActionBarsService.cs` wires left rail icons (`Workspace`, `Files`, `Maps`, `Assets`, `Taxi`, `Capture`, `Diagnostics`, `Options`, `Layers`) to `ToggleLeftDock(LeftPanel)`. | **PASS** |
| **AC-104**: All floating tool windows for left-rail tools are eliminated. | `DrawToolWindows()` in `ViewerActionBarsService.cs` has retired all `DrawToolWindow` floating window invocations. | **PASS** |
| **AC-201**: Floating `ActionWindow_destination` retired. | `ViewerActionBarsService.cs` removed `_windows["destination"]` and floating destination window. | **PASS** |
| **AC-202**: Editor mode docks cleanly into sidebar panels. | `ViewerUiManager.cs` `DrawRightDock()` delegates to `DrawEditorDockContent()` when `ActiveTopTab == WorkbenchTab.Editor`. | **PASS** |
| **AC-203**: Archaeology mode docks cleanly into right dock. | `ViewerUiManager.cs` `DrawRightDock()` delegates to `DrawArchaeologyDockContent()` when `ActiveTopTab == WorkbenchTab.Archaeology`. | **PASS** |
| **AC-204**: Seamless destination switching without closing sidebars or spawning floating windows. | `Navigate(WorkbenchTab)` maintains `_rightDockOpen = true` and updates docked content dynamically. | **PASS** |
| **AC-301**: Remove redundant `Pages` button from bottom action bar. | `bottom-pages` removed from `ViewerFooterLayoutTests.cs`, `ViewerActionBarsService.cs`, `ui-controls.json`. | **PASS** |
| **AC-302**: Streamline `ViewerPanel.World` from 4 to 3 pages; redirect exports to `ViewerPanel.Exports`. | `WorkbenchPanelsService.cs` page array reduced to `["Objects", "Placements", "LOD & Budget"]` with direct action button linking to `ViewerPanel.Exports`. | **PASS** |
| **AC-303**: Streamline Model Info in `ViewerPanel.Inspector`, redirect duplicate exports. | Model Info export section replaced with action button opening `ViewerPanel.Exports`. | **PASS** |
| **AC-304**: Centralize Terrain File Operations & DAT exports in `ExportWorkbenchService`. | `ExportWorkbenchService.DrawWorldExportSection()` now embeds `_host.MainMenuBar.DrawTerrainFileOperationsContent()`. | **PASS** |
| **AC-305**: Remove duplicate capture embed and vestigial button in Archaeology. | Removed `DrawCapturePanelContent()` from Playback tab and removed `Inspect PM4 in Viewer` button in `ArchaeologyPanelService.Sidebars.cs`. | **PASS** |
| **AC-401**: Replace hardcoded style colors with `UiTokens`. | `AssetCatalogView.cs`, `ArchaeologyPanelService.PhaseLayers.cs`, `EditorPanelsService.cs` refactored to `UiTokens`. | **PASS** |
| **AC-402**: Register all modified control paths in `ui-controls.json`. | `ui-controls.json` updated with `open-model-exports` and `open-world-exports`. | **PASS** |
| **AC-403**: `Verify-UiReference.ps1` passes with 0 errors and 0 warnings. | Exact exit code 0; `UI_REFERENCE_CHECK PASSED errors=0 warnings=0`. | **PASS** |
| **AC-404**: Solution builds cleanly with 0 errors and unit tests pass. | Build succeeded with 0 errors, all 22 targeted unit tests passed. | **PASS** |

