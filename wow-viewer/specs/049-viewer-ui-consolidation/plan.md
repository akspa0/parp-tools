# Plan: 049 Viewer UI Consolidation

## Phases

### Phase 1: Audit & Map (estimated: 30 min)
- List every panel/window currently rendered in `DrawUI()` (ViewerApp.cs ~line 1523)
- Categorize each into Scene/PM4/Terrain/Utilities
- Identify duplicate rendering (same content in sidebar + floating window)
- Map each content method to its category

### Phase 2: Categorized Tools Menu (estimated: 1 hr)
- Rewrite the Tools menu (`ImGui.BeginMenu("Tools")` at ViewerApp.cs line 1774) into four categorized sections
- Each `MenuItem` toggles the correct `_show*` flag
- Disable items when their prerequisite isn't loaded (e.g., no terrain → terrain tools disabled)

### Phase 3: Floating Window Extraction (estimated: 1 hr)
- Ensure every tool that should be a floating window has:
  - A `_show*` boolean field
  - An `ImGui.Begin("Title", ref _showFlag)` guarded render call in `DrawUI()`
  - A Tools menu entry
- New windows to add: PM4 Correlation (split from PM4 Workbench tab), PM4 Alignment
- Already done: UniqueId Archaeology, Taxi Panel, Weak Signal Amplifier

### Phase 4: Sidebar Consolidation (estimated: 2 hr)
- The right sidebar's `DrawUnifiedToolSidebar()` currently has 4 collapsible sections
- Replace these with the Scene Inspector tabbed panel (already exists as `SceneInspector` shell panel)
- Remove `DrawUnifiedToolSidebar()` — its content is now in Scene Inspector tabs
- Verify all content from `DrawUnifiedViewerSettingsSidebarContent`, `DrawUnifiedSelectionSidebarContent`, `DrawUnifiedWorldToolsSidebarContent`, `DrawViewerDiagnosticsSidebarContent` is accessible from the new panel layout

### Phase 5: Remove Duplicates (estimated: 30 min)
- Ensure no content method is called from two different visible panels
- The utility popup (`DrawViewerDiagnosticsSidebarContent`) rendered in old sidebar must move to Tools menu or individual tool windows

### Phase 6: Sticky Behavior Audit (estimated: 30 min)
- Verify every floating window uses `ImGui.Begin("Title", ref _showFlag)` (not `ImGuiCond.Once`)
- Verify close button (X) sets the flag to false properly
- Verify `imgui.ini` saves positions
