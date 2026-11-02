# Implementation Plan: Spec 291 — Dual Sidebar Docking, Destination Integration & UI Harmonization

## 1. Architectural Strategy & Design System Boundaries

### 1.1 Dual Dock Layout Architecture (`UiRegion.LeftDock` & `UiRegion.RightDock`)
In `UiLayout.cs`:
- Introduce `UiRegion.LeftDock` into `UiRegion` enum:
  ```csharp
  internal enum UiRegion { Scene, TopBar, BottomBar, StatusBar, LeftSidebar, LeftDock, RightSidebar, RightDock, Options, Minimap, Chat }
  ```
- Geometry:
  - Position: `X = lw` (immediately to the right of the left icon rail), `Y = top`.
  - Dimensions: `Width = leftDockWidth * scale`, `Height = sceneHeight`.
  - Chat layout adjustment: When `LeftDock` is open, `chatDockX = lw + leftDockWidth + 10f * scale` ensuring chat never collides with or hides under the left dock.
- Splitter & Resizing:
  - A right-edge grip splitter on `LeftDock` allows the user to drag to resize between 260px and 720px.
  - Persisted in layout config as `left_dock_width` and `left_dock_open`.

### 1.2 Left Dock Panel Management (`LeftPanel`)
In `ViewerUiManager.cs` / `WorkbenchTab.cs`:
- Define `LeftPanel` enum:
  ```csharp
  public enum LeftPanel { Workspace, Files, Maps, Assets, Taxi, Capture, Diagnostics, Options, Layers }
  ```
- `ViewerUiManager` tracks `_leftDockOpen`, `_leftDockWidth`, and `LeftPanel ActiveLeftPanel`.
- Clicking left action bar icons toggles `ActiveLeftPanel` and opens `_leftDockOpen`.
- The icon button highlights as active when the dock is open and that panel is displayed.

### 1.3 Destination Integration into Sidebar Docks
Currently, `Editor` and `Archaeology` are forced into a floating popup in the lower-left corner (`ActionWindow_destination`). Under Spec 291:
1. When switching to `WorkbenchTab.Editor`:
   - `RightDock` automatically displays the Editor workbench (`Placement & Objects`, `Terrain Tools`, `Converters`).
   - `LeftDock` (or Editor Left panel) hosts Project Output and New Map Creator.
2. When switching to `WorkbenchTab.Archaeology`:
   - `RightDock` automatically displays the Archaeology workbench (`Weak Signal & Stratigraphy`, `UniqueId Timeline`, `Layers & Provenance`, `Playback`, `Cartography`).
3. Switching to `WorkbenchTab.Viewer`:
   - `RightDock` returns to the active `ViewerPanel` (`Inspector`, `World`, `Gossip`, `PM4`, `Exports`).
4. Floating `ActionWindow_destination` is completely removed.

### 1.4 Deduplication & Cleanup
1. **Bottom Action Bar**:
   - Delete `Pages` button from `ViewerFooterLayout.cs` and `ViewerActionBarsService.DrawBottomActions()`.
   - The destination buttons (`Viewer`, `Editor`, `Archaeology`) directly switch the active workspace and its docked panels.
2. **Export Consolidation**:
   - `ViewerPanel.World`: Streamline pages from 4 to 3 (`["Objects", "Placements", "LOD & Budget"]`). Remove page 3 ("Exports"). Add a button: "Open Export Workbench" (`_host.UiManager.OpenViewer(ViewerPanel.Exports)`).
   - `ViewerPanel.Inspector`: Remove redundant model export section; add button to open Export Workbench.
   - `ExportWorkbenchService`: Add Terrain Alpha/Heightmap/MCCV exports to the "World & Terrain Geometry" section.
   - `Archaeology`: Remove vestigial `Inspect PM4 in Viewer` button and remove duplicate `DrawCapturePanelContent()` inside Archaeology Playback.

### 1.5 Styling Harmonization
- Replace hardcoded color literals in `AssetCatalogView.cs`, `ArchaeologyPanelService.PhaseLayers.cs`, and `EditorPanelsService.cs` with `UiTokens` (`UiTokens.Brass`, `UiTokens.Surface1`, `UiTokens.Border`, etc.).

---

## 2. Phased Implementation Roadmap

### Phase 1: Dual Dock Core Architecture (`UiRegion.LeftDock`)
- Add `UiRegion.LeftDock` to `UiLayout.cs` and calculate rect with non-colliding chat positioning.
- Add `LeftPanel` enum, `_leftDockOpen`, and `_leftDockWidth` persistence to `ViewerUiManager.cs`.
- Render `UiRegion.LeftDock` with right-edge interactive splitter in `ViewerUiManager.cs`.
- Wire Left Action Bar icons to toggle `UiRegion.LeftDock` instead of opening floating tool windows.
- Retire floating `DrawToolWindow` calls for Workspace, Files, Maps, Assets, Taxi, Capture, Diagnostics, Options, and Layers.

### Phase 2: Destination Integration into Sidebars
- Remove floating `ActionWindow_destination`.
- Wire `WorkbenchTab.Editor` and `WorkbenchTab.Archaeology` to render directly inside `RightDock` (and LeftDock as appropriate) with unified section headers.
- Remove redundant `Pages` button from bottom action bar.

### Phase 3: Export & Navigation Deduplication
- Streamline `ViewerPanel.World` to 3 pages, removing duplicate Exports tab.
- Streamline `ViewerPanel.Inspector` Model Info, removing duplicate export section.
- Expand `ExportWorkbenchService` with Terrain Alpha/Heightmap/MCCV exports.
- Remove vestigial `Inspect PM4 in Viewer` button and duplicate Capture panel in Archaeology.

### Phase 4: Styling Cleanup & Registry Sync
- Replace remaining hardcoded colors with `UiTokens`.
- Update `ui-controls.json` via registry updater script.
- Verify `pwsh -File wow-viewer/eng/ui/Verify-UiReference.ps1` passes with 0 errors and 0 warnings.
- Build solution and verify all tests pass.

