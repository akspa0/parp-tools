# Spec 071: Left/Right Sidebar Split + Model Viewer Mode

**Branch**: `071-left-right-sidebar-split`
**Status**: Draft
**Owner**: wow-viewer (viewer shell)
**Builds on**: 069 (single Workbench panel, headless content variants)
**Replaces**: 069 workbench-on-right-side-only layout

## Context

069 landed a single Workbench popout on the right side of the master window. User feedback:

1. **File browser / World Maps list should be in a separate LEFT sidebar**, not in the right workbench. Loading a map is a different mental activity from inspecting it.
2. **No useful model inspection panels in the workbench.** Loading a model shows nothing useful — no model info, no animation list, no play/pause/stop button. Need a **Model Viewer mode**.
3. **All popups/panels should be tabs.** No floating windows. Log viewer, perf, render quality, capture automation, taxi, archeology range/layers/playback/capture, asset catalog, runtime stats — all should be tabs in the right workbench, not floating windows.

## User Scenarios

### US1 — Two-side layout (P1)
**Given** viewer is open,
**When** user looks at the UI,
**Then** they see:
- **Left sidebar** (resizable, ~360px default): file browser + world maps list + workspace bars (open folder/file)
- **Center**: 3D viewport (full size, no chrome overlap)
- **Right sidebar** (resizable, ~480px default): the Workbench with all tool tabs
- **Menu bar** at top
- **Status bar** at bottom

**Acceptance**: `DrawLeftSidebar` (new), `DrawWorkbenchPopout` (renamed to `DrawRightSidebar`), 3D viewport rect calculation accounts for both sidebars.

### US2 — File browser / World Maps always visible (P1)
**Given** user has loaded a game folder,
**When** they look at the left sidebar,
**Then** they see:
- **Source** section: Open Game Folder, Open File, current source path/status
- **File Browser** section: file tree, search filter, .mdx/.wmo/.m2/.blp/.wdt filter
- **World Maps** section: discovered maps list with Load/Spawn buttons

**Acceptance**: Reuses `DrawWorkspaceBarsPanelContent` + `DrawFileBrowserContent` + `DrawMapDiscoveryContent`. No nested windows.

### US3 — Model Viewer mode (P1)
**Given** user loaded a model (M2/MDX/WMO),
**When** they look at the right workbench,
**Then** they see a "Model" top tab with sub-tabs:
- **Info**: model path, type, vertex/triangle count, materials, textures
- **Animations**: animation list (sequences), play/pause/stop buttons, frame slider, speed
- **Actions**: Frame Model, Auto-frame toggle, WMO doodad set selector (if WMO)
- **LOD**: level-of-detail controls (if available)

**Acceptance**: New `ModelViewerTopTab` in workbench. Reuses `DrawModelInfoContent` + `DrawSelectedSqlGameObjectAnimationControls` + MdxRenderer/M2Renderer.Animation APIs.

### US4 — All popups are tabs (P1)
**Given** user clicks any menu item that previously opened a floating window (Log Viewer, Perf, Render Quality, Capture Automation, Taxi, Asset Catalog),
**When** the action triggers,
**Then** instead of opening a floating window, the right workbench switches to the matching tab.

**Acceptance**: Tools menu items call `OpenWorkbenchTab(WorkbenchTab.X)` instead of setting `_show*Window` flags. Legacy `_show*Window` flags still work (legacy mode).

### US5 — Three top categories in workbench (P1)
**Given** user looks at the workbench top tab bar,
**When** they switch tabs,
**Then** they see 3 top-level categories (not 6):
- **Model** — for inspecting individual models (M2/MDX/WMO)
- **World** — for working with the loaded world map (placements, tiles, overlays, source)
- **Tools** — for archeology, capture automation, PM4, terrain, log, perf, etc.

**Acceptance**: 3 top tabs replaces current 6. Each has 4-8 sub-tabs.

## Functional Requirements

### FR-001: Left sidebar
- New `DrawLeftSidebar()` method called in DrawUI before workbench
- Position: x=0, y=topOffset, width=_leftSidebarWidth, height=viewport_height
- Width persisted as `_leftSidebarWidth` (already exists, default ~360px)
- Content: `DrawWorkspaceBarsPanelContent` + `DrawFileBrowserContent` + `DrawMapDiscoveryContent`
- 3D viewport calculation: `x = _leftSidebarWidth`, `width = displayWidth - _leftSidebarWidth - _rightSidebarWidth`

### FR-002: Right sidebar (workbench)
- Rename `DrawWorkbenchPopout` → `DrawRightSidebar`
- Position: x=displayWidth - _rightSidebarWidth, y=topOffset, width=_rightSidebarWidth, height=viewport_height
- Width persisted as `_rightSidebarWidth` (already exists, default ~480px)
- 3D viewport calculation excludes this width too

### FR-003: Model Viewer mode
- New `TopTab.Model` (or sub-tab under Model top tab)
- Sub-tabs: Info / Animations / Actions / LOD
- Animations sub-tab reuses `DrawSelectedSqlGameObjectAnimationControls` + adds explicit Play/Pause/Stop buttons + frame slider
- Model currently loaded = `_renderer` (MdxRenderer or WmoRenderer or M2Renderer)
- Selected model = `_selectedObject` (when user clicks object in world)

### FR-004: Three top tabs (Model / World / Tools)
- Replace `TopTab` enum with 3 values: Model, World, Tools
- Each has its own sub-tab set
- Sub-tab sets are exhaustive: all current Tools menu items become Tools sub-tabs

### FR-005: Tools menu integration
- `Tools > Log Viewer` → `OpenWorkbenchTab(WorkbenchTab.Log)` (no _show*Window)
- Same for Perf, Render Quality, Capture Automation, Taxi, Asset Catalog, etc
- Legacy `_show*Window` flags preserved for users who toggle View > Legacy Windows

### FR-006: Per-tab persistence
- `_activeTopTab` + `_activeTopBottomTab` (per top tab, separate sub-tab index) persisted in ViewerSettings
- Resets when model changes (model-specific sub-tabs)

## Out of Scope

- Per-workbench native windows (deferred to 070)
- Model viewer LOD details (basic only for v1)
- Per-frame animation scrubbing (step buttons, not timeline)

## Success Criteria

1. Viewer launches with 2 sidebars + 3D viewport between them
2. Loading a model opens Model > Info tab with vertex/triangle/material info
3. Animations tab has Play/Pause/Stop + frame slider
4. All Tools menu items switch workbench tab, no floating windows
5. World > Source, World > Placements, World > Tiles, World > Overlays all work
6. Tools > Log, Perf, Render Quality, Capture Automation, etc. all become tabs

## Files to Create

| File | Purpose |
|------|---------|
| `src/viewer/WoWViewer/Workbench/WorkbenchTab.cs` | Top tab enum (Model/World/Tools) |
| `src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs` | Sub-tab routing helpers |

## Files to Modify

| File | Change |
|------|--------|
| `src/viewer/WoWViewer/ViewerApp.cs` | Add `DrawLeftSidebar()`, rename `DrawWorkbenchPopout` → `DrawRightSidebar`, fix 3D viewport rect |
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | New 3 top tabs, Model Viewer sub-tabs, route Tools menu items |
| `src/viewer/WoWViewer/ViewerApp_ClickSelection.cs` | Model Viewer mode hooks into click selection |

## Migration from 069

- TopTab enum shrinks from 6 to 3
- All sub-tab content methods preserved, just routed differently
- Tools menu items become tab switchers instead of `_show*Window = true`
- 3D viewport math updated to subtract both sidebar widths
- Model Viewer sub-tabs reuse existing M2/MDX animation APIs

## Phases

1. **Phase A**: Add 3D viewport math (both sidebars)
2. **Phase B**: Add DrawLeftSidebar
3. **Phase C**: Rename DrawWorkbenchPopout → DrawRightSidebar, update 3D viewport calc
4. **Phase D**: TopTab → 3 values (Model/World/Tools), Tools menu integration
5. **Phase E**: Model Viewer mode (Info sub-tab)
6. **Phase F**: Model Viewer mode (Animations sub-tab with Play/Pause/Stop)
7. **Phase G**: Model Viewer mode (Actions, LOD sub-tabs)
8. **Phase H**: Memory bank + spec sync
