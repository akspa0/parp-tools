# Spec 069: Viewer UI Overhaul — Top/Bottom Tab Bands, Archeology Window, Playback, World+Terrain Merge

**Branch**: `069-viewer-ui-overhaul`
**Status**: Draft
**Owner**: wow-viewer (viewer shell)
**Builds on**: 044 (shell usability), 049 (UI consolidation), 060 (UI cleanup)

## Context

Viewer UI grew organic. 12+ shell panels + 15 floating windows. User wants structural rebuild:

1. **Top tabs + bottom tabs** instead of left/right sidebars. Two horizontal bands, each with its own tab set. Sidebars removed.
2. **6 top-level tabs**: Scene | World | Terrain | PM4 | Archeology | Utilities.
3. **Archeology as first-class top-level tab** with uniqueId filtering, range sliders, sticky persistence, playback.
4. **World + Terrain merged** into single "World" tab with sub-tabs for placements, tiles, overlays.
5. **Minimap** lives in Utilities tab, resizable, shares code with fullscreen.

## User Scenarios

### US1 — Cells overlay (P1, already done)
8x8 green grid per chunk at 66.666 world units. Verified working.

### US2 — Top tab bar + bottom tab bar layout (P1)
**Given** user has viewer open with world loaded,
**When** they look at the UI,
**Then** they see:
- Menu bar (top, existing)
- **Top tab bar** below menu: [Scene] [World] [Terrain] [PM4] [Archeology] [Utilities]
- 3D viewport (center, full available space between bars)
- **Bottom tab bar** above status bar: context-sensitive tabs for active top tab
- Status bar (bottom, existing)
**And** no left/right sidebars. Maximized viewport.

**Acceptance**: `DrawUI()` renders top tab bar before viewport, bottom tab bar after. Sidebar draw calls removed. Layout state in `ViewerApp` fields, not `ShellPanelId`.

### US3 — Top tab = Scene (P1)
**Given** Scene tab active,
**When** user looks at bottom tabs,
**Then** sees: [Selection] [Camera] [Settings] [Themes]
**And** selecting a bottom tab shows its content panel above the bottom bar.

**Acceptance**: `DrawSceneTabContent()` method. Bottom tabs vary by top tab.

### US4 — Top tab = World (P1)
**Given** World tab active,
**When** user looks at bottom tabs,
**Then** sees: [Placements] [Tiles] [Overlays] [Selection Tools]
**And** Placements = MDDF/MODF/WMO list (was World Objects).
**And** Tiles = tile/chunk selection grid (was Terrain Tools).
**And** Overlays = chunk/tile/cell grid toggles + alpha/shadow/contour.

**Acceptance**: `DrawWorldTabContent()` with sub-tab dispatch. Old `DrawWorldObjectsContentCore` + `DrawTerrainToolsContent` content merged here.

### US5 — Top tab = Terrain (P1)
**Given** Terrain tab active,
**When** user looks at bottom tabs,
**Then** sees: [Layers] [Clipboard] [Analysis] [MCNK] [Weak Signal] [Export]
**And** Layers = base/L1/L2/L3/holes toggles.
**And** Clipboard = chunk copy/paste.
**And** Analysis = terrain analysis reports.

**Acceptance**: `DrawTerrainTabContent()`. Old `_showChunkClipboardWindow`, `_showTerrainAnalysisWindow`, `_showMcnkExplorerWindow`, `_showWeakSignalWindow` become sub-tabs.

### US6 — Top tab = PM4 (P1)
**Given** PM4 tab active,
**When** user looks at bottom tabs,
**Then** sees: [Overlay] [Selection] [Correlation] [Info] [Match] [Alignment]
**And** same content as old PM4 Workbench tabs + PM4 floating windows, now consolidated.

**Acceptance**: `DrawPm4TabContent()`. Old PM4 floating windows become sub-tabs.

### US7 — Top tab = Archeology (P1) — uniqueId work
**Given** Archeology tab active,
**When** user looks at bottom tabs,
**Then** sees: [Range] [Layers] [Playback] [Capture]
**And** Range = visible range min/max sliders, scope selector (Per Map / Camera Tile).
**And** Layers = detected archeology layers table.
**And** Playback = Play/Pause/Stop, speed slider, loop checkbox.
**And** Capture = "Apply to next capture" / "Apply to video recording" checkboxes.

**Acceptance**: `DrawArcheologyTabContent()`. Old `_showUniqueIdArchaeologyWindow` content lives here. World tab no longer has uniqueId controls.

### US8 — Top tab = Utilities (P1)
**Given** Utilities tab active,
**When** user looks at bottom tabs,
**Then** sees: [Minimap] [Log] [Perf] [Render Quality] [Taxi] [Capture Automation] [Asset Catalog] [Runtime Stats]
**And** Minimap = interactive minimap surface, resizable.

**Acceptance**: `DrawUtilitiesTabContent()`. All old floating windows become sub-tabs.

### US9 — Sticky archeology range (P1)
**Given** user sets Visible Range Start = 1234, End = 5678,
**When** they close+reopen viewer,
**Then** range restored from `viewer_settings.json`.

**Acceptance**: `_archeologyMinUniqueId`, `_archeologyMaxUniqueId` fields. Save in `SaveViewerSettings()`. Load in `LoadViewerSettings()`.

### US10 — Archeology playback (P1)
**Given** user clicks Play in Archeology > Playback sub-tab,
**When** playback active,
**Then** `Visible Range End` animates from current min to max at user-set speed,
**And** `_worldScene.UniqueIdFilterMax` updates per frame,
**And** user can pause/resume/stop.
**And** touching slider pauses playback (per user choice).

**Acceptance**: `DrawArcheologyPlaybackContent()`. `_archeologyPlaybackActive`, `_archeologyPlaybackSpeed`, `_archeologyPlaybackLoop` fields. Per-frame update in `Update()`.

### US11 — Capture automation + archeology playback (P1)
**Given** user enables "Apply playback to next capture" in Archeology > Capture sub-tab,
**When** capture sequence runs,
**Then** playback runs once per capture frame, end advances per shot.
**And** for video recording, playback runs at recording start, plays to end or until video ends.
**And** recording uses real-time playback speed (per user choice).

**Acceptance**: `ArcheologyPlaybackConfig { Enabled, Speed, Loop, FrameStep }` on `PendingCaptureRequest`. `VideoRecordingConfig.ArcheologyPlayback` field. `DrawArcheologyCaptureContent()`.

### US12 — Minimap as resizable sub-tab (P1)
**Given** Utilities > Minimap sub-tab active,
**When** user resizes the panel (or window),
**Then** internal minimap texture scales to fit,
**And** all functionality (teleport, drag, click, show loaded tiles) works identically to fullscreen mode.

**Acceptance**: `DrawInteractiveMinimapSurface(interactionId, cursorPos, mapSize, ...)` already takes size param. Minimap sub-tab calls it with sub-tab's content region size. Fullscreen mode (`_fullscreenMinimap`) calls it with full viewport size. Same code path.

## Functional Requirements

### Layout (FR-001 to FR-007)
- FR-001: Remove `_useDockspaceUi`, `ShellPanelId` enum, `DrawDockedShellPanelsForLane`, `DrawLeftSidebar`, `DrawRightSidebar` from active code paths.
- FR-002: New `TopTab` enum: `Scene | World | Terrain | PM4 | Archeology | Utilities`.
- FR-003: New `BottomTab` enum (per top tab, e.g., `WorldBottomTab { Placements, Tiles, Overlays, SelectionTools }`).
- FR-004: `_activeTopTab`, `_activeBottomTab` fields. Persisted in `viewer_settings.json`.
- FR-005: `DrawTopTabBar()` renders tab buttons. `DrawBottomTabBar()` renders context-sensitive buttons. `DrawMainViewport()` clears and renders 3D scene between bars.
- FR-006: All `ShellPanelDefinition`, `_savedShellPanelLayouts`, dock persistence code removed (kept as dead code for one release, then deleted).
- FR-007: Menu bar still has File / View / Tools / Help. View menu simplified (Hide UI Chrome only). Tools menu removed (replaced by tabs).

### Top tab dispatch (FR-008 to FR-013)
- FR-008: `DrawTopTabContent(TopTab tab)` dispatches to per-tab `Draw*TabContent()`.
- FR-009: `DrawBottomTabBar(TopTab activeTop)` renders only valid bottom tabs for active top.
- FR-010: `DrawBottomTabContent(TopTab activeTop, BottomTab tab)` dispatches to per-sub-tab `Draw*SubTabContent()`.
- FR-011: `DrawSceneTabContent()` → `DrawSceneSubTab_X()` for each Scene sub-tab.
- FR-012: `DrawWorldTabContent()` → `DrawWorldSubTab_X()` (merges old World Objects + Terrain Tools).
- FR-013: `DrawTerrainTabContent()` → `DrawTerrainSubTab_X()` (Layers/Clipboard/Analysis/MCNK/Weak Signal/Export).

### PM4 tab (FR-014 to FR-016)
- FR-014: `DrawPm4TabContent()` → `DrawPm4SubTab_X()` (Overlay/Selection/Correlation/Info/Match/Alignment).
- FR-015: Old `_showPm4*Window` flags removed, replaced by sub-tab state.
- FR-016: `ViewerApp_Pm4Utilities.cs` content methods refactored to be sub-tab content.

### Archeology tab (FR-017 to FR-024)
- FR-017: `DrawArcheologyTabContent()` → `DrawArcheologySubTab_Range/Layers/Playback/Capture`.
- FR-018: `_showUniqueIdArchaeologyWindow` flag removed. `_activeBottomTab == ArcheologyBottomTab.Range` shows range sub-tab.
- FR-019: `_archeologyMinUniqueId`, `_archeologyMaxUniqueId` fields, persisted.
- FR-020: `_archeologyPlaybackActive`, `_archeologyPlaybackSpeed` (units/sec), `_archeologyPlaybackLoop` fields.
- FR-021: Per-frame `UpdateArcheologyPlayback(double deltaTime)` advances `_worldScene.UniqueIdFilterMax` when active.
- FR-022: Slider interaction pauses playback (sets `_archeologyPlaybackActive = false`).
- FR-023: `ArcheologyPlaybackConfig` class with `Enabled`, `Speed`, `Loop`, `FrameStep` (for captures).
- FR-024: `PendingCaptureRequest.ArcheologyPlayback` field, applied per-shot.

### Utilities tab (FR-025 to FR-028)
- FR-025: `DrawUtilitiesTabContent()` → sub-tabs Minimap/Log/Perf/Render Quality/Taxi/Capture Automation/Asset Catalog/Runtime Stats.
- FR-026: `_showMinimapWindow`, `_showLogViewer`, `_showPerfWindow`, `_showRenderQualityWindow`, `_showTaxiWindow`, `_showCaptureAutomationWindow` flags removed.
- FR-027: `DrawUtilitiesSubTab_Minimap()` calls `DrawInteractiveMinimapSurface` with sub-tab content size.
- FR-028: `_fullscreenMinimap` still exists, calls same surface fn with full viewport size. M key toggles.

### Capture automation (FR-029 to FR-032)
- FR-029: `VideoRecordingConfig.ArcheologyPlayback` field (bool).
- FR-030: `VideoRecordingConfig.ArcheologyPlaybackSpeed` field (float).
- FR-031: When recording starts and `ArcheologyPlayback == true`, kick off playback at `ArcheologyPlaybackSpeed` real-time, advance each frame.
- FR-032: When recording ends, stop playback, restore previous `UniqueIdFilterMax`.

### Minimap (FR-033 to FR-035)
- FR-033: `DrawInteractiveMinimapSurface(string, Vector2, float, ...)` already accepts size. Verify works at small (300px) and large (fullscreen 1920px) sizes.
- FR-034: Sub-tab content region size = available content area in tab panel.
- FR-035: Resize: sub-tab content fills its tab area, minimap scales with it.

## Out of Scope

- Replacing ImGui with Avalonia/MAUI (deferred per 060).
- Adding new archeology analysis algorithms (existing detection preserved).
- New minimap export formats (PNG already supported).
- WMO minimap tools (separate `WowViewer.Tool.WmoMinimap`).
- Liquid minimap export.

## Success Criteria

1. Viewer launches with top + bottom tab bars. No sidebars visible.
2. All 6 top tabs work, each with 3-8 sub-tabs.
3. All old floating windows accessible as sub-tabs.
4. World + Terrain merged: placements/tiles/overlays all under World tab.
5. Archeology tab has range/layers/playback/capture sub-tabs.
6. Playback advances `Visible Range End` at user-set speed.
7. Touching slider pauses playback.
8. Video recording with `ArcheologyPlayback = true` captures full playback at real-time.
9. Range min/max + playback speed sticky across sessions.
10. Minimap resizable, same code path for sub-tab and fullscreen.
11. No regression: cells grid, PM4 tools, chunk editing all still work.

## Files to Change (major)

| File | Change |
|------|--------|
| `src/viewer/WoWViewer/ViewerApp.cs` | Add TopTab/BottomTab enums, _activeTopTab/_activeBottomTab fields, remove ShellPanelId, DrawTopTabBar, DrawBottomTabBar, DrawMainViewport |
| `src/viewer/WoWViewer/ViewerApp_Sidebars.cs` | Rename to ViewerApp_TabSystem.cs. Replace sidebar draw methods with tab draw methods. |
| `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs` | Refactor minimap surface fn to be called from both Utilities sub-tab and fullscreen. |
| `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` | Add ArcheologyPlaybackConfig to PendingCaptureRequest + VideoRecordingConfig. |
| `src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs` | Refactor PM4 content methods to be sub-tab content. |
| `src/viewer/WoWViewer/ViewerApp_TerrainAnalysis.cs` | Move to Terrain sub-tab content. |
| `src/viewer/WoWViewer/ViewerApp_Workspaces.cs` | Add archeology sticky settings load/save. |
| `src/viewer/WoWViewer/ViewerApp_Themes.cs` | Move to Settings sub-tab under Scene. |
| `docs/architecture/viewer-ui-panels-reference.md` | Rewrite with new tab structure. |

## Open Questions (resolved in this draft)

1. ~~Archeology window type~~ → Sub-tab in Utilities removed, Archeology is its own top tab.
2. ~~Playback interaction on slider touch~~ → Pause.
3. ~~Video recording playback speed~~ → Real-time.
4. ~~World+Terrain merge depth~~ → Full merge with sub-tabs (Placements/Tiles/Overlays).
5. ~~Tab layout~~ → Top tabs + bottom tabs.
6. ~~Tab count~~ → 6 top tabs (Scene/World/Terrain/PM4/Archeology/Utilities).

## Notes

- Cells overlay (US1) verified working. No code change needed for it.
- 049 tasks.md has many unchecked items; this spec supersedes 049 by going further (tab system vs sidebar).
- 060 marked complete; this spec extends with tab-based layout.
- Existing `ShellPanelId` system remains in code as dead/disabled for one release to ease rollback. Final removal in 070.
- Capture automation playback = "real-time" per user choice. FPS × playback_speed = range advance per frame.
