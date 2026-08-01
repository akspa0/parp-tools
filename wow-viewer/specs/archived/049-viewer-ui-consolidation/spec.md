# Spec 049: Viewer UI Consolidation

**Status**: In Progress | **Priority**: P1 | **Owner**: WoWViewer

**Related specs**:
- `044-viewer-shell-usability` — foundation layer; landed the dockable shell host (`DrawDockspaceHost`), `UseDockspaceUi` default + persistence, fixed-sidebar splitter suppression, `World Maps` auto-open on client load, and `Tools > Offline Data / Conversion` menu declutter. 049 builds the categorized Tools menu, floating window sticky behavior, and right sidebar consolidation on top of that foundation. US4 (cursor-as-model) in 044 remains deferred P2.

## Problem

The viewer UI has grown organically over 10 months of development. Panels are scattered across a right sidebar with collapsible sections, separate docked panels, and floating windows — all with no consistent organizational principle. Related tools are in different places (e.g., Taxi Panel was buried inside World Objects, Weak Signal inside Terrain Controls). Users waste time hunting for controls.

## User Stories

### US1: Tool Categorization (P1)
As a user, I want every panel/window categorized into one of four groups — **Scene**, **PM4**, **Terrain**, **Utilities** — so I know where to find things by the type of work I'm doing.

### US2: Tools Menu (P1)
As a user, I want a single Tools menu that lists every available tool with its open/closed state, so I can discover and toggle features without memorizing locations.

### US3: Tool Windows Stay Open (P1)
As a user, I want every tool window to be "sticky" — it stays where I put it and doesn't disappear when I click elsewhere. Position saved in `imgui.ini`.

### US4: No Duplicate Content (P2)
As a user, I want the same panel content to appear in exactly one place — no sidebar section AND floating window showing the same thing.

## Panel Categories

### Scene Group — inspecting the current 3D scene
| Panel | Source Content | Type |
|-------|---------------|------|
| **Inspector** | Current Selection panel + World Objects panel + Model Info + Runtime Stats | Docked panel (right) |
| **PM4 Workbench** | PM4 overlay + selection + correlation | Docked panel (right) |
| **PM4 Info** | Selected-object details + raw MSLK/MSHD + export buttons | Docked panel (right) |

### Terrain Group — editing terrain data
| Panel | Source Content | Type |
|-------|---------------|------|
| **Terrain Tools** | Tile/chunk selection grid, overlay toggles, export scope | Floating window |
| **Terrain Controls** | Time of day, fog, WDL, wireframe, layout mode | Docked panel (right) |
| **Chunk Clipboard** | Copy/paste chunks, heightmap save | Floating window |
| **Terrain Analysis** | Terrain analysis reports | Floating window |
| **MCNK Explorer** | MCNK chunk explorer | Floating window |
| **Weak Signal Amplifier** | Weak-signal terrain restore controls | Floating window |

### PM4 Group — PM4 decoding tools
| Panel | Source Content | Type |
|-------|---------------|------|
| **PM4 Workbench** | Overlay tab (color/split/legend) + Selection tab (matches/graph) + Correlation tab | Docked panel |
| **PM4 Info** | Selected object details, raw MSLK/MSHD, export buttons | Docked panel |
| **PM4 Object Match** | Per-object match window | Floating window |
| **PM4 Correlation** | WMO correlation inspector | Floating window |
| **PM4 Alignment** | Precise object alignment controls | Floating window |

### Utilities Group — general tools
| Panel | Source Content | Type |
|-------|---------------|------|
| **Navigator** | World overview, file browser, map discovery | Docked panel (left) |
| **Minimap** | Tile minimap | Floating window |
| **Log Viewer** | Debug log output | Floating window |
| **Perf** | Performance stats | Floating window |
| **Render Quality** | Render quality settings | Floating window |
| **UniqueId Archaeology** | UniqueId layer analysis | Floating window |
| **Taxi Panel** | Taxi path visualization and ride camera | Floating window |
| **Capture Automation** | Screenshot automation | Floating window |
| **Asset Catalog** | Browse/load game assets | Floating window |

## Functional Requirements

### FR-001: Category-Aware Layout
- Each panel/window is assigned to one of four categories
- The Tools menu groups items by category with separators
- The right sidebar's default dock layout organizes panels within their category

### FR-002: Tools Menu
- Every toggleable panel appears in the Tools menu
- Checkbox shows current open/closed state
- Categories separated by `ImGui.Separator()`
- Disabled items shown for unavailable tools (e.g., "Terrain Tools" disabled when no terrain loaded)

### FR-003: Floating Window Persistence
- All floating windows use `ImGuiCond.FirstUseEver` for size
- All floating windows use `ref _showFlag` for lifetime management
- Close button (X) sets flag to false
- Position/size saved in `imgui.ini` automatically

### FR-004: No Duplicate Rendering
- Each content method is rendered in exactly one place
- No panel renders the same content as another visible panel
- The old collapsible-section right sidebar (`DrawUnifiedToolSidebar`) is replaced by individual floating/docked panels

### FR-005: Menu Bar Updates
- The View menu keeps scene-relevant toggles (Left Sidebar, Right Sidebar, Minimap)
- The Tools menu gets all tool toggles
- Remove stale menu items that point to panels that no longer exist as separate entities

## Success Criteria

1. Tools menu lists all 15+ panels in categorized groups with working toggles
2. Each panel content renders in exactly one place
3. Floating windows stay where placed across restarts
4. No content from old sidebar sections is lost
5. Right sidebar panels are properly categorized by work type (Scene/PM4/Terrain/Utilities)

## Assumptions

- ImGui's window lifetime (`ref bool`) is sufficient for sticky behavior
- The existing content methods (`DrawSelectionPanelContent`, etc.) can be called directly from their new panel hosts
- `imgui.ini` persistence works for all floating windows
- The shell panel system (`ShellPanelId`, `DrawDockedShellPanelsForLane`) remains the docked-panel infrastructure
