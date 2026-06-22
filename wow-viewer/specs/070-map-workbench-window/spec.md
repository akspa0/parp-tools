# Spec 070: Map Workbench Window — Per-Map Workspace with Native Feel

**Branch**: `070-map-workbench-window`
**Status**: Draft
**Owner**: wow-viewer (viewer shell)
**Builds on**: 069 (tab system data model + archeology playback)
**Replaces**: 069 tab layout (which never delivered native feel)

## Context

The 069 tab system approach was wrong. Tabs at top + bottom + popout windows + child regions = still a "Debug window" overlay on a master viewport. The user wants something that **feels native**, like a real tool (Visual Studio, Blender, Chrome tabs).

The right model: each loaded map = a **workbench**. A workbench is a window or docked panel that owns:
- Its own 3D viewport (no master viewport — the workbench IS the viewport)
- Its own tab UI (top tab bar inside the workbench)
- Its own sub-tab content (right side panel inside the workbench)
- Its own minimap (corner of the workbench)

The master window is just a **launcher**: file browser, map list, "new workbench" button. Once a workbench is open, the user works inside the workbench. The master becomes a thin launcher that can be minimized/hidden.

## User Scenarios

### US1 — Workbench is the workspace (P1)
**Given** a map is loaded,
**When** the user opens it,
**Then** a new workbench opens as a docked tab in the master window,
**And** the workbench contains: 3D viewport (full content area), top tab bar, side panel, minimap corner.

**Acceptance**: `MapWorkbench` class. Per-workbench state: terrain manager, world scene, camera, render state, tab UI state.

### US2 — Multiple workbenches as docked tabs (P1)
**Given** two maps are loaded,
**When** the user opens the second map,
**Then** the second workbench appears as a tab next to the first in the master window,
**And** the user can switch between them by clicking tabs,
**And** each workbench remembers its own camera position, tab state, archeology playback state.

**Acceptance**: `MapWorkbenchManager` tracks list of open workbenches. Master dockspace has a central tab strip for workbench tabs.

### US3 — Tear-off to native window (P2)
**Given** a workbench is open as a docked tab,
**When** the user drags the tab out of the master window or clicks a "pop out" button,
**Then** a new OS-level window opens containing that workbench,
**And** the workbench is removed from the master tab strip,
**And** the new window has full chrome (title bar, min/max, close).

**Acceptance**: `IWindow` from Silk.NET for each torn-off workbench. Shared GL context.

### US4 — Workbench internal layout (P1)
**Given** a workbench is focused,
**When** the user looks at it,
**Then** they see:
- **Top**: thin tab bar (Scene / World / Terrain / PM4 / Archeology / Utilities)
- **Center**: 3D viewport (full content area)
- **Right side**: collapsible side panel with sub-tab content
- **Bottom-right corner**: minimap (~200x200)
- **Bottom**: thin status bar

**Acceptance**: Workbench has its own `DrawUI` that draws all this. No master chrome inside workbench (just its own).

### US5 — Master = launcher only (P1)
**Given** the user starts the viewer,
**When** they look at the master window,
**Then** they see:
- **Left sidebar**: file browser, map list, "open workbench" button
- **Center**: docked tab strip of open workbenches (or empty state with prompt)
- **Right side**: optional inspector for selected object across workbenches
- **Menu bar**: File, View, Window (workbench list), Help

**Acceptance**: Master chrome is minimal. The user clicks a map in the sidebar → new workbench tab opens. User can hide the master sidebar to focus on workbench.

## Functional Requirements

### FR-001: MapWorkbench class
- `MapWorkbench(string mapPath, WorkbenchManager mgr)` constructor
- Owns: `TerrainManager` (or `VlmTerrainManager`), `WorldScene`, `Camera`, `Lighting`, tab UI state
- Has its own `Render()` method (3D scene) and `DrawUI()` method (ImGui overlay)
- Has lifecycle: `Initialize`, `Update`, `Render`, `Dispose`

### FR-002: Workbench internal layout
- Top tab bar: `TopTab` enum from 069
- Side panel: collapsible, contains active sub-tab content
- Minimap: bottom-right corner
- 3D viewport: full center area
- Status bar: thin bottom

### FR-003: WorkbenchManager
- `Dictionary<Guid, MapWorkbench> _workbenches`
- `OpenWorkbench(string mapPath)` creates new workbench
- `CloseWorkbench(Guid id)`
- `GetActiveWorkbench()` returns current focused workbench
- Routes input/clipboard between workbenches and master

### FR-004: Master docking
- Master has a dockspace (already enabled in 069)
- Central dock hosts workbench tabs
- Left sidebar hosts file browser + map list
- Right sidebar hosts shared inspector (optional)

### FR-005: Native multi-window (US3)
- `WorkbenchWindow(IWindow silkWindow, MapWorkbench bench)` wraps torn-off workbench
- Shares GL context with master
- Has its own ImGui context OR shares master context
- Renders workbench's 3D scene + tab UI
- Min/max/close buttons via `IWindow`

### FR-006: Per-workbench state
- Camera position
- Active top tab + sub-tab
- Archeology range/playback state
- Capture automation state
- All persisted to `viewer_settings.json` keyed by map path

### FR-007: Workbench chrome
- Each workbench has its own menu bar (compact): File > Close Workbench, View > Tools, etc
- Top tab bar with Scene/World/Terrain/PM4/Archeology/Utilities
- Side panel with sub-tab content (resizable, collapsible)
- Bottom-right minimap (always visible, draggable to move)
- Bottom: per-workbench status bar (chunk count, uniqueId range, etc)

## Out of Scope (Future)

- Per-workbench capture automation (capture is per-workbench already, but cross-workbench batch is future)
- Cross-workbench archeology comparison
- Workbench templates (save workbench state as a "preset")
- Sync camera between workbenches

## Success Criteria

1. User opens the viewer, sees a launcher with file browser + map list
2. Click a map → new workbench docked tab opens with 3D viewport + tab UI
3. Open a second map → second workbench tab
4. Switch between workbench tabs → each remembers its own state
5. Tear off a workbench → native OS window with same content
6. Each workbench feels like its own app, not a debug overlay

## Files to Create

| File | Purpose |
|------|---------|
| `src/viewer/WoWViewer/Workbench/MapWorkbench.cs` | Per-map workbench class |
| `src/viewer/WoWViewer/Workbench/WorkbenchManager.cs` | Manages list of workbenches |
| `src/viewer/WoWViewer/Workbench/WorkbenchWindow.cs` | Native window wrapper for torn-off workbench |
| `src/viewer/WoWViewer/Workbench/WorkbenchLayout.cs` | Layout calculation for workbench (top tab bar, side panel, minimap) |
| `src/viewer/WoWViewer/Workbench/WorkbenchSettings.cs` | Per-workbench persisted state |

## Files to Modify

| File | Change |
|------|--------|
| `src/viewer/WoWViewer/ViewerApp.cs` | Replace single-viewport render with workbench-aware render. Master chrome only. |
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | Move tab content methods into `MapWorkbench` class. Add master sidebar (file browser + map list) without side panels. |
| `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` | Minimap moves into workbench. Status bar moves into workbench. |

## Migration from 069

- Tab enums, sub-tab content methods, archeology playback logic all carry over
- Layout logic gets rewritten
- Master chrome gets simplified

## Open Questions

1. Should torn-off workbench windows be fully independent processes or shared-process with the master?
2. Should workbench state be persisted per-map-path or per-build-version?
3. How do workbenches handle the "no map loaded" state (just terrain manager, no world scene)?
4. Should capture automation capture the whole master or just one workbench?

## Notes

- Each workbench has its own state, but share the GL context + ImGui context for now (full multi-context is much more complex)
- Per-workbench minimap is its own `MinimapRenderer` instance
- Per-workbench camera is a separate `Camera` instance
- Shared between workbenches: data source, asset catalog, theme settings
