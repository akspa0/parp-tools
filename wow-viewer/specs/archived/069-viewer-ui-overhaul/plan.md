# Plan: 069 Viewer UI Overhaul

**Branch**: `069-viewer-ui-overhaul`
**Generated from**: `spec.md`
**Phases**: 8 small, independently-validatable phases. Each phase = one PR.

## Phase 1: Tab System Foundation (no behavior change yet)

Add top/bottom tab enums and fields. Don't remove sidebars yet — both systems run side by side behind a flag. Old shell panels still work.

**Files**:
- `src/viewer/WoWViewer/ViewerApp.cs`: add `TopTab`, `WorldBottomTab`, `TerrainBottomTab`, `Pm4BottomTab`, `ArcheologyBottomTab`, `UtilitiesBottomTab`, `SceneBottomTab` enums. Add `_useTabUi` bool (default false), `_activeTopTab`, `_activeBottomTab` fields.
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: add `DrawTopTabBar()`, `DrawBottomTabBar()`, `DrawMainViewport()`, `DrawTopTabContent()`, `DrawBottomTabContent()` no-op stubs.

**Done when**: `_useTabUi = true` shows empty tab bars + empty main viewport, sidebars disabled. Viewer still launches, no regression.

## Phase 2: Scene + Utilities tab (low risk, isolated)

Move existing sidebar content into Scene tab. Move existing floating windows into Utilities tab.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: implement `DrawSceneTabContent()` → sub-tabs Selection/Camera/Settings/Themes. Implement `DrawUtilitiesTabContent()` → sub-tabs for each old floating window.
- `src/viewer/WoWViewer/ViewerApp_Themes.cs`: add `DrawSceneSubTab_Themes()` wrapper.
- `src/viewer/WoWViewer/ViewerApp_Workspaces.cs`: add `DrawSceneSubTab_Settings()` wrapper.

**Done when**: With `_useTabUi = true`, Scene tab and Utilities tab work. Old sidebars disabled. All previous functionality accessible via tabs.

## Phase 3: World + Terrain merge (medium risk)

Consolidate World Objects + Terrain Tools into single World tab with sub-tabs.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: add `DrawWorldTabContent()` → sub-tabs Placements/Tiles/Overlays/SelectionTools.
- `src/viewer/WoWViewer/ViewerApp_TerrainAnalysis.cs`: refactor `DrawTerrainToolsContent()` to `DrawWorldSubTab_Tiles()`.
- Remove `ShellPanelId.WorldObjects` from enum (keep as `[Obsolete]` alias for one release).

**Done when**: World tab shows all 4 sub-tabs with same content as old World Objects + Terrain Tools merged. Old `_showTerrainToolsWindow` deprecated.

## Phase 4: Terrain + PM4 tabs (low risk, mostly refactor)

Move remaining terrain tools and PM4 content into dedicated tabs.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: `DrawTerrainTabContent()` → Layers/Clipboard/Analysis/MCNK/Weak Signal/Export sub-tabs.
- `src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs`: refactor content methods to `DrawPm4SubTab_X()`.
- Remove all `_show*Window` flags for: ChunkClipboard, TerrainAnalysis, McnkExplorer, WeakSignal, Pm4Alignment, Pm4ObjectMatch, Pm4WmoCorrelation.

**Done when**: Terrain and PM4 tabs work. Old floating windows disabled. All previous functionality accessible.

## Phase 5: Archeology tab foundation

New top tab with range/layers sub-tabs. No playback yet.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: `DrawArcheologyTabContent()` → Range/Layers/Capture sub-tabs.
- Move `DrawUniqueIdArchaeologyContent()` body into `DrawArcheologySubTab_Range()`.
- Add `DrawArcheologySubTab_Layers()` and `DrawArcheologySubTab_Capture()`.
- Remove uniqueId controls from `DrawWorldObjectsContentCore()` (already in Phase 3 World tab).

**Done when**: Archeology tab has 3 sub-tabs (Range/Layers/Capture) with all old uniqueId functionality.

## Phase 6: Sticky archeology settings

Persist range + scope across sessions.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_Workspaces.cs` (or settings file): add `_archeologyMinUniqueId`, `_archeologyMaxUniqueId`, `_archeologyScopeIndex` to settings load/save.
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: load settings on init, save on change.

**Done when**: User sets range, closes viewer, reopens, range restored.

## Phase 7: Archeology playback (animation + capture integration)

Add playback controls, per-frame update, capture integration.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_Sidebars.cs`: add `DrawArcheologySubTab_Playback()`. Add `_archeologyPlaybackActive`, `_archeologyPlaybackSpeed`, `_archeologyPlaybackLoop` fields.
- `src/viewer/WoWViewer/ViewerApp.cs`: add `UpdateArcheologyPlayback(double deltaTime)` called in `Update()`.
- `src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs`: add `ArcheologyPlaybackConfig` class. Add `ArcheologyPlayback` field to `PendingCaptureRequest`. Add `VideoRecordingConfig.ArcheologyPlayback` + `ArcheologyPlaybackSpeed`.

**Done when**: Play/Pause/Stop work. Slider advances at user-set speed. Capture automation runs playback per shot. Video recording runs playback at real-time.

## Phase 8: Minimap refactor + final cleanup

Make minimap a sub-tab in Utilities. Remove dead code. Update docs.

**Files**:
- `src/viewer/WoWViewer/ViewerApp_MinimapAndStatus.cs`: refactor `DrawInteractiveMinimapSurface` to accept explicit size, called from both Utilities > Minimap sub-tab and fullscreen mode.
- `src/viewer/WoWViewer/ViewerApp.cs`: remove `ShellPanelId` enum, `ShellPanelDefinition` records, dock state, sidebar methods. Keep `ShellPanelId` as `[Obsolete]` for one release.
- `docs/architecture/viewer-ui-panels-reference.md`: rewrite with tab system.

**Done when**: Minimap resizable in Utilities tab, same code path as fullscreen. All old floating window flags removed. All sidebar code removed. Viewer fully tab-based.

## Validation per Phase

Each phase must pass:
- `dotnet build wow-viewer/WowViewer.slnx -c Debug` (compile only — skip tests)
- Manual smoke: viewer launches, tabs render, no crash
- No regression: cells grid, PM4 tools, capture automation all work
- If phase touches archeology: range/layers/playback all work as spec'd

## Out-of-Phase Work (Future Specs)

- Avalonia/MAUI migration (per 060)
- New archeology analysis algorithms
- Liquid minimap export
- WMO minimap tools integration into viewer
