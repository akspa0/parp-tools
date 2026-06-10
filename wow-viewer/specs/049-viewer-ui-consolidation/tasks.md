# Tasks: 049 Viewer UI Consolidation

## Phase 1: Audit & Map

- [ ] T001 [P] List every panel/window rendered in `DrawUI()` (~line 1523 of ViewerApp.cs). Write a panel inventory with: name, show flag, content method, category (Scene/PM4/Terrain/Utilities), and whether it's docked, floating, or sidebar.

- [ ] T002 [P] Audit `DrawUnifiedToolSidebar()` (ViewerApp_Sidebars.cs ~line 791) and its four content methods. Trace every piece of content in: `DrawUnifiedViewerSettingsSidebarContent`, `DrawUnifiedSelectionSidebarContent`, `DrawUnifiedWorldToolsSidebarContent`, `DrawViewerDiagnosticsSidebarContent`. Map each to where it will go after consolidation.

## Phase 2: Categorized Tools Menu

- [ ] T003 Rewrite the Tools menu (`ImGui.BeginMenu("Tools")` at ViewerApp.cs ~line 1774) into four categorized groups: Scene, PM4, Terrain, Utilities. Each group separated by `ImGui.Separator()`. Each `MenuItem` toggles the correct `_show*` flag. Disable items when prerequisites aren't loaded.

The menu should contain:
**Scene**: Scene Inspector, PM4 Workbench, PM4 Info
**PM4**: PM4 Object Match, PM4 Correlation, PM4 Alignment
**Terrain**: Terrain Tools, Terrain Controls, Chunk Clipboard, Terrain Analysis, MCNK Explorer, Weak Signal Amplifier
**Utilities**: UniqueId Archaeology, Taxi Panel, Minimap, Log Viewer, Perf, Render Quality, Capture Automation, Asset Catalog

## Phase 3: Floating Window Extraction

- [ ] T004 [P] Extract PM4 Correlation from the PM4 Workbench's Correlation tab into its own floating window (`_showPm4CorrelationWindow`). The PM4 Workbench retains its Overlay + Selection tabs. The Correlation tab calls `DrawPm4CorrelationInspectorContent()`.

- [ ] T005 [P] Ensure every floating window uses `ImGui.Begin("Title", ref _showFlag)` for sticky close-button behavior. Audit existing windows: Log Viewer, Perf, Render Quality, Terrain Tools, Chunk Clipboard, Terrain Analysis, MCNK Explorer, Capture Automation, PM4 Alignment, PM4 Object Match, PM4 Correlation, UniqueId Archaeology, Taxi Panel, Weak Signal Amplifier.

## Phase 4: Sidebar Consolidation

- [ ] T006 [P] Remove `DrawUnifiedToolSidebar()` (ViewerApp_Sidebars.cs ~line 791). Its four content methods' contents are now accessible from individual tool windows or the Scene Inspector tabs. Remove `DrawRightSidebarSection`, `DrawUnifiedViewerSettingsSidebarContent`, `DrawUnifiedSelectionSidebarContent`, `DrawUnifiedWorldToolsSidebarContent`, `DrawViewerDiagnosticsSidebarContent`.

- [ ] T007 [P] Verify all content from the removed sidebar sections is still accessible. Specifically:
  - Theme settings → available in Viewer Settings (keep inline or move to dedicated window)
  - Camera controls → keep in Scene Inspector or Navigator
  - UniqueId Archaeology → now its own window (T001 already done)
  - Terrain controls adjustment → accessible via Scene Inspector Terrain tab or Terrain Controls panel
  - Model info → Scene Inspector Model tab
  - World objects → Scene Inspector World tab
  - PM4 Workbench → Scene Inspector PM4 tab or standalone PM4 Workbench
  - Utility toggles → in Tools menu

- [ ] T008 [P] Remove duplicate content rendering. Search for every content method being called from multiple places and ensure each is called from exactly one visible host.

## Phase 4b: Sticky & Dockable Panels

- [ ] T011 [P] Verify the dockspace system (`_useDockspaceUi`) allows users to drag-and-group any shell panel together. Panels that share a category should default to being tab-grouped in the same dock node. Users can ungroup/rearrange freely. No panel auto-closes when clicking elsewhere — only the X button or menu toggle closes it.

- [ ] T012 [P] For floating windows (not docked shell panels), ensure they all use `ImGui.Begin("Title", ref _showFlag)` so they have a title bar with close button and stay open when clicking the scene. Position and size are saved to `imgui.ini` automatically.

## Phase 5: Sticky Behavior & Polish

- [ ] T009 [P] Audit all floating windows: verify `ImGuiCond.FirstUseEver` for size, verify `ref _showFlag` is respected when closed with X button, verify `imgui.ini` saves/restores positions.

- [ ] T010 [P] Update `wow-viewer/docs/architecture/viewer-ui-panels-reference.md` with the final panel inventory, categories, and show flags.

## Notes

- No content methods should be deleted, only moved or re-hosted
- The Scene Inspector panel already exists and calls the same content methods by tab
- Phase 4 (sidebar removal) is the riskiest — verify thoroughly before committing
