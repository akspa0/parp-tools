# Feature Specification: Spec 291 — Dual Sidebar Docking, Destination Integration & UI Harmonization

**Owner**: Epic 251 (Viewer UX, Shell & Code Health)  
**Origin**: Operator prompt (2026-11-01) requesting a comprehensive UI audit, deduplication, and unification of all panels, windows, options, editor, and archaeology tools into the left and right sidebar panels.  
**Status**: Draft / Specified (2026-11-01)

---

## 1. Executive Summary & Problem Analysis

Over recent iterations (Specs 280, 283, 286, 288, 289, 290), the viewer UI has made substantial progress:
- The right-hand side has a cohesive **Right Dock** (`UiRegion.RightDock`) hosting `Inspector`, `World`, `Gossip`, `PM4`, and `Exports`.
- The top menu is streamlined to File/View/Help.
- The footer has an adaptive layout policy with auto-scaling metrics.

However, the UI remains roughly "80% of the way there" due to several architectural asymmetries and vestigial patterns:

### 1.1 Left vs. Right Rail Asymmetry (Floating Windows vs. Docked Sidebars)
- The **Right Rail** toggles a docked sidebar panel under the minimap (`UiRegion.RightDock`) that never occludes the 3D scene haphazardly.
- The **Left Rail**, in contrast, spawns **floating, independent tool windows** (`ActionWindow_workspace`, `ActionWindow_files`, `ActionWindow_maps`, `ActionWindow_assets`, `ActionWindow_taxi`, `ActionWindow_capture`, `ActionWindow_diagnostics`) that drift across the viewport, overlapping the scene, minimap, and chat.
- `Options` and `Terrain Layers` also spawn as free-floating popup windows rather than docking.

### 1.2 Destination Window Isolation (Editor & Archaeology Floating in Lower-Left)
- Clicking `Editor` or `Archaeology` on the bottom toolbar sets `_rightDockOpen = false` and opens a floating tool window (`ActionWindow_destination`) positioned in the lower-left workspace.
- This creates a jarring paradigm: Viewer tools live in docked sidebars, but Editor and Archaeology tools live in a floating dialog over the terrain and chat.

### 1.3 Bottom Action Bar & Navigation Redundancy
- The bottom bar contains `Viewer`, `Editor`, `Archaeology`, `Spawns`, `Pages`, and `Options`.
- `Pages` is redundant because clicking the destination button (`Editor` or `Archaeology`) already opens its pages.
- `Spawns` opens a standalone modal dialog, whereas `Maps` (left rail) already provides spawn point discovery and `Gossip` / `Simulation` handles population streaming.
- Archaeology contains vestigial cross-destination buttons (`Inspect PM4 in Viewer`) and embeds the entire camera `Capture` panel inside its Playback page, duplicating the Left Rail's dedicated `Capture` panel.

### 1.4 Fragmented Export Surfaces
- Despite Spec 290 creating `ExportWorkbenchService` (`ViewerPanel.Exports`), redundant export interfaces remain active:
  1. `World` panel still has a 4th tab dedicated to "Exports".
  2. `Inspector` panel (Model Info) still embeds duplicate GLB model export buttons.
  3. `Editor > Data I/O` still embeds duplicate Terrain File Operations (Alpha, Heightmap, MCCV).
  4. `Maps` sidebar still embeds DAT map export controls.

### 1.5 Bespoke Styling Remaining
- Several classes still employ manual `PushStyleColor` overrides (e.g. hardcoded navy/gold backgrounds in `AssetCatalogView`, hardcoded orange text in `ArchaeologyPanelService.PhaseLayers`, blue buttons in `EditorPanelsService`) rather than standard `UiTokens` and `SharedUiWidgets`.

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Unified Dual Dock Architecture (`UiRegion.LeftDock` & `UiRegion.RightDock`) (Priority: P1)
As an operator interacting with client files, maps, assets, capture, and diagnostics, I want left-rail tools to dock into an anchored, resizable left sidebar panel (`UiRegion.LeftDock`) so that tools never float freely over the 3D scene.

- **AC-101**: Introduce `UiRegion.LeftDock` in `UiLayout.cs` positioned between `UiRegion.LeftSidebar` (the rail) and the 3D scene, bounded vertically between the top menu bar and bottom action bar.
- **AC-102**: `UiRegion.LeftDock` features an interactive edge splitter enabling smooth width resizing (persisted between sessions, min 260px, max 720px).
- **AC-103**: Clicking any icon on the Left Action Rail (`Workspace`, `File Browser`, `Maps`, `Assets`, `Taxi`, `Capture`, `Diagnostics`) or `Options` / `Terrain Layers` toggles or activates the tool in `UiRegion.LeftDock`.
- **AC-104**: All floating tool windows (`DrawToolWindow`) for left-rail tools are eliminated in favor of docked presentation.

### User Story 2: Destination Integration into Sidebars (Priority: P1)
As an operator switching between Viewer, Editor, and Archaeology modes, I want Editor and Archaeology workbenches hosted cleanly within the docked sidebars rather than spawning a detached floating window in the lower-left quadrant.

- **AC-201**: The floating `ActionWindow_destination` is retired.
- **AC-202**: In `Editor` mode, editor tools dock cleanly into the sidebar panels:
  - Left Dock: Project Output, New Map Creation, Data Import.
  - Right Dock: Placement Authoring, Rosetta Object Library, Population, Terrain Lab, Converters.
- **AC-203**: In `Archaeology` mode, archaeology tools dock cleanly into the right dock (Weak Signal & Stratigraphy, UniqueId Timeline, Layers & Provenance, Playback, Cartography).
- **AC-204**: Switching destinations via bottom bar seamlessly switches the docked workbench views without closing sidebars or spawning floating windows.

### User Story 3: Deduplication & Navigation Streamlining (Priority: P2)
As a user navigating the interface, I want redundant controls, duplicate export menus, and vestigial navigation buttons eliminated so that every operation has exactly one obvious home.

- **AC-301**: Remove the redundant `Pages` button from the bottom action bar.
- **AC-302**: Remove the duplicate "Exports" tab from `ViewerPanel.World` (streamlining World to 3 clean pages: Objects, Placements, LOD & Budget) and redirect export requests to `ViewerPanel.Exports`.
- **AC-303**: Streamline Model Info in `ViewerPanel.Inspector`, replacing duplicate export buttons with a link to `ViewerPanel.Exports`.
- **AC-304**: Centralize Terrain Alpha/Heightmap/MCCV exports and DAT exports into `ExportWorkbenchService` as the single authoritative export hub.
- **AC-305**: Remove the duplicate `DrawCapturePanelContent()` embed from Archaeology Page 3, and remove the vestigial `Inspect PM4 in Viewer` button from Archaeology.

### User Story 4: Design System Consistency & Registry Sync (Priority: P2)
As a maintainer, I want all panels to use `SharedUiWidgets` primitives and `UiTokens`, and all controls registered in `ui-controls.json` so that `Verify-UiReference.ps1` passes with zero errors and zero warnings.

- **AC-401**: Replace hardcoded style colors in `AssetCatalogView.cs`, `ArchaeologyPanelService.PhaseLayers.cs`, and `EditorPanelsService.cs` with `UiTokens`.
- **AC-402**: Register all modified control paths in `eng/ui/ui-controls.json`.
- **AC-403**: `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` passes with 0 errors and 0 warnings.
- **AC-404**: Solution builds cleanly with 0 errors and all unit tests pass.

