# Spec 080: WoW-Like UI Consolidation — Frame-Based Navigation, Zero Duplication

**Status**: Draft
**Owner**: wow-viewer (viewer shell)
**Builds on**: 069 (tab system), 071 (sidebar split), 077 (bar layout)
**Supersedes**: 049 (UI consolidation — incomplete), 073 (surface revamp)

## Context

The viewer UI has grown organically through 79+ specs. The current state has:

- **7 levels of nested tabs**: Top tabs (Model/World/Tools) → Tools sub-tabs (Quick/Archeology/PM4/Terrain/Utilities) → sub-sub-tabs (Layers/Clipboard/Analysis/etc.)
- **3+ places for the same toggle**: Layer visibility, grid lines, surface overlays are in Bottom Bar, World > Overlays, Tools > Terrain > Layers simultaneously
- **Ephemeral windows** (Taxi) that disappear on click-away
- **Left sidebar** crammed with unrelated content: file browser + map discovery + minimap + world overview
- **Right sidebar (workbench)** has no spatial persistence — everything is nested behind tabs, no way to see two things at once
- **No WoW-like design language**: chaotic information density, no visual hierarchy

The goal is to make the UI feel like World of Warcraft's interface: clean, consistent, frame-based, with zero duplication. Each function has one home. Windows stay where you put them. Common toggles are on a persistent action bar.

## User Scenarios

### US1 — Bottom action bar is the single source of truth for toggles (P1)

**Given** the viewer is open with a terrain-backed world loaded,
**When** the user wants to toggle grid lines, overlays, or layer visibility,
**Then** they should find ALL of these controls on one persistent bottom action bar.
**And** no duplicate controls should exist in any sub-tab.

**Acceptance**:
- Bottom bar contains: grid toggles (chunks/tiles/cells), surface overlays (alpha/shadow/MCCV/contours), layer visibility (Base/L1/L2/L3/Holes), liquid toggle, split `Terrain WF` and `M2/WMO WF` toggles
- `M2/WMO WF` renders visible object wireframes as an overlay over normal solid object rendering; it must not enable hover-only reveal or hide non-hovered objects
- World > Overlays sub-tab: removed (no grid/overlay/layer controls)
- Tools > Terrain > Layers sub-tab: removed (no layer/overlay/grid controls)
- Settings window: removed (fog defaults moved to a proper setting, render quality removed from here)
- All layer/overlay/grid toggles have exactly one home

### US2 — Named frames replace nested tab hierarchy (P1)

**Given** the user opens the Tools menu,
**When** they select "World", "Terrain", "PM4", "Archeology", "Model", or "Utilities",
**Then** a proper named window opens (like WoW's Character/Spellbook frames).
**And** the window stays open at the position the user placed it.
**And** it does NOT disappear when clicking the 3D viewport.
**And** it has a title bar, close button, and is resizable.
**And** the window remembers its position across the session.

**Acceptance**:
- `DrawWorldFrame()` → tabbed internal frame: Source/Placements/Tiles/Overlays/SelectionTools
- `DrawTerrainFrame()` → Layers/Clipboard/Analysis/MCNK/WeakSignal/Export
- `DrawPm4Frame()` → Overlay/Selection/Correlation/Info/Match/Alignment
- `DrawArcheologyFrame()` → Range/Layers/Playback/Capture
- `DrawModelFrame()` → Info/Animations/Actions/LOD
- `DrawUtilitiesFrame()` → Minimap/Log/Perf/CaptureAutomation/Taxi/AssetCatalog
- Each frame has `_showWorldFrame`, `_showTerrainFrame`, etc. bools persisted in settings
- No more right sidebar (workbench) — replaced by individually-addressable frames
- No more left sidebar — file browser becomes a dialog or frame

### US3 — Settings is a proper modal/game menu (P1)

**Given** the user clicks File → Settings or presses Escape,
**Then** they see a proper settings window with categories:
- Display: render quality, texture filtering, MSAA, backface culling
- Fog: global fog start/end defaults; the Lighting surface owns any active world override
- Interface: tab UI toggle, minimap visibility, theme
- Camera: speed, FOV defaults

**Acceptance**:
- `DrawSettingsWindow()` opens as a modal or persistent window
- Settings window is NOT called from Tools > Quick or any sub-tab
- Fog defaults from settings apply globally, are persisted in viewer_settings.json

### US4 — Minimap lives in its own frame or screen corner (P1)

**Given** the user has a world loaded,
**When** they click Tools → Minimap or press M,
**Then** the minimap appears as a proper frame in the top-right corner of the screen.
**And** it stays there.
**And** it can be resized.
**And** all minimap interactions (teleport, drag, fullscreen) work.

**Acceptance**:
- M key toggles fullscreen minimap (existing behavior preserved)
- Tools → Minimap opens a positionable frame in top-right corner
- Minimap frame remembers its position across the session
- Minimap is NOT in any sub-tab

### US5 — Taxi panel stays open (P1)

**Given** the user opens the Taxi panel from Tools → Utilities → Taxi,
**When** they click on the 3D viewport or another frame,
**Then** the Taxi panel stays open.
**And** they can interact with it until they close it.

**Acceptance**:
- DrawTaxiContent uses a proper ImGui window with ref bool for close tracking
- Window flag includes `ImGuiWindowFlags.NoDocking` to prevent collapse
- No popup behavior (BeginPopup removed if present)

### US6 — Zero dead controls (P2)

**Given** the user opens any frame or panel,
**When** they look at each control,
**Then** every visible button/checkbox/slider has a working function.
**And** any control that is not yet wired is hidden, not shown as a dead button.

**Acceptance**:
- Every `Draw*Content` method in Viewers has been audited for dead controls
- Controls that are planned but not implemented are gated behind `_featureEnabled` flags
- Controls that can't work in the current state show a disabled state with tooltip

### US7 — File/World loading as a dialog (P2)

**Given** the user wants to open a game folder or load a map,
**When** they click File → Open Game Folder or File → Open Map,
**Then** they see a dialog (not a sidebar section).
**And** the dialog has: folder picker, file browser, and map list.
**And** once done, the dialog closes.

**Acceptance**:
- Left sidebar removed. `DrawLeftSidebar()` gutted.
- File browser and map discovery moved into a proper dialog window
- Dialog can be reopened from File menu
- `DrawFileBrowserContent()` and `DrawMapDiscoveryContent()` only called from the dialog

## Functional Requirements

### FR-001 to FR-010: Frame System
- FR-001: Replace right sidebar (workbench) with individually-addressable named frames
- FR-002: Each frame has `_show<Name>Frame` bool, persisted in ViewerSettings
- FR-003: Each frame uses `ImGui.Begin("<Name>", ref _show<Name>Frame, ImGuiWindowFlags.None)` for stable windowing
- FR-004: Frames remember their position within a session (ImGui auto-handles this with .ini)
- FR-005: Frames appear in Tools menu as toggleable items
- FR-006: Hotkeys: Ctrl+W = World, Ctrl+T = Terrain, Ctrl+P = PM4, Ctrl+A = Archeology, Ctrl+M = Model, Ctrl+U = Utilities
- FR-007: `DrawRightSidebar()` removed when `_useTabUi=true`
- FR-008: `DrawLeftSidebar()` removed when `_useTabUi=true`
- FR-009: Frame content methods (`DrawWorldFrame()`, etc.) are extracted from current sidebar sub-tab content
- FR-010: Each frame has internal tabs (like WoW's spellbook has tabs for each school)

### FR-011 to FR-015: Action Bar
- FR-011: Bottom bar is the single action bar for all scene toggles
- FR-012: Bottom bar contains: Chunks, Tiles, Cells, Alpha, Shadows, MCCV, Contours, Base, L1, L2, L3, Holes, Liquid, WL*, Wireframe
- FR-013: Bottom bar height: ~40px, full window width
- FR-014: Above bottom bar: status bar with camera coordinates on the left and a compact runtime line on the right containing FPS, AreaName, CPU frame time, tile/chunk residency, WMO/MDX visibility, and pending asset loads
- FR-015: Top toolbar (`DrawToolbar()`) only shows investigation mode buttons when no terrain is loaded

### FR-016 to FR-020: Duplicate Removal
- FR-016: World > Overlays sub-tab removed entirely (all controls in bottom bar)
- FR-017: Tools > Terrain > Layers sub-tab removed (all controls in bottom bar)
- FR-018: Tools > Quick sub-tab removed (camera/fog/layer controls now in frames)
- FR-019: World Overview minimap in left sidebar removed (minimap is its own frame)
- FR-020: Export controls only in Terrain frame > Export sub-tab (not in World > Tiles)

### FR-021 to FR-025: Settings
- FR-021: Settings window is a single modal/persistent panel
- FR-022: Settings contains: render quality (texture filtering, MSAA, culling), fog defaults (start/end), interface (theme, tab UI toggle), camera defaults
- FR-023: Settings is opened from File → Settings or a gear icon on the bottom bar
- FR-024: `_showSettingsWindow` bool persists in ViewerSettings
- FR-025: Fog defaults apply on load when no active world override is selected; the Lighting surface
  exposes the active start/end range, its source, and reset-to-lighting behavior

### FR-026 to FR-030: Window Behavior
- FR-026: All frames use `ImGuiWindowFlags.NoDocking` to prevent docking collapse
- FR-027: All frames use `ImGui.Begin(..., ref _showFlag)` with proper close button
- FR-028: Taxi window: debug and fix the ephemeral behavior (ensure it's a Begin/End window with ref bool)
- FR-029: ImGui .ini saves window positions automatically
- FR-030: On next launch, frames restore to their last position

## Out of Scope

- Replacing ImGui with Avalonia/MAUI (deferred per 060)
- New minimap rendering features
- Archeology algorithm changes
- Adding new tool functionality (pure UI reorganization)
- Theme overhaul (only structural changes, not visual theming)
- Hotkey customization (hardcoded default hotkeys only)

## Success Criteria

1. Every grid/overlay/layer toggle exists in exactly ONE place (the bottom bar)
2. No content method is called from more than one dispatch path
3. Opening Tools → any named frame opens a stable, positionable window
4. Taxi panel: open it, click viewport, still there
5. `_showLeftSidebar` and `_showRightSidebar` are false for new sessions
6. No dead buttons, no "not implemented" states
7. Build: 0 errors, no new unused-method warnings; every remaining warning is inventoried and assigned a release disposition

## Files to Change

| File | Change |
|------|--------|
| `src/viewer/WoWViewer/ViewerApp.cs` | Remove DrawLeftSidebar/DrawRightSidebar calls, add frame draw calls, add frame bools to ViewerSettings, add hotkey handling |
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | Remove DrawLeftSidebar/DrawRightSidebar, remove DrawWorldOverlaysSubTab, remove DrawTerrainLayersSubTab, remove DrawQuickControlsContent, refactor content into Draw*Frame methods |
| `src/viewer/WoWViewer/ViewerApp_Settings.cs` | Expand settings to cover render quality + fog + interface + camera |
| `src/viewer/WoWViewer/ViewerApp_RenderQuality.cs` | Remove DrawRenderQualityWindow wrapper (content preserved for settings) |
| `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` | Add DrawMinimapFrame method, consistent with frame system |
| `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` | Fix DrawTaxiContent to use proper window if needed |
| `src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs` | Remove or gut (replaced by frame system) |
| `wow-viewer/memory-bank/activeContext.md` | Update after implementation |
| `wow-viewer/memory-bank/progress.md` | Update after implementation |

## Phases

1. **Phase A**: Remove all duplicates from sub-tabs (World>Overlays, Terrain>Layers, Tools>Quick). Bottom bar becomes single source of truth for toggles.
2. **Phase B**: Refactor named frames. Extract each frame from the right sidebar. Add `_show*Frame` flags. Wire to Tools menu.
3. **Phase C**: Remove sidebars. Gut left sidebar, move file browser to dialog. Remove right sidebar/workbench.
4. **Phase D**: Settings consolidation. Expand Settings window, remove separate Render Quality window.
5. **Phase E**: Taxi fix + window behavior audit. Ensure every frame is stable.
6. **Phase F**: Memory bank + spec sync.

## Notes

- Phase A alone eliminates the "why is this toggle in 3 places" problem
- Phase B alone makes the UI feel like WoW (named frames instead of nested tabs in a sidebar)
- Phases A+B can be done together as they're interdependent (removing duplicates requires deciding WHERE the control lives)
- Phase C is the biggest change (removing sidebars entirely) and should be validated carefully
- After Phase C, the UI is: menu bar + 3D viewport + bottom action bar + status bar + named frames + fullscreen minimap. This matches WoW's layout closely.
