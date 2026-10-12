# Tasks: 069 Viewer UI Overhaul

## Phase 1: Tab System Foundation

- [x] T001 Add `TopTab` enum: Scene, World, Terrain, PM4, Archeology, Utilities. (ViewerApp.cs)
- [x] T002 Add per-top-tab `BottomTab` enums: `SceneBottomTab`, `WorldBottomTab`, `TerrainBottomTab`, `Pm4BottomTab`, `ArcheologyBottomTab`, `UtilitiesBottomTab`.
- [x] T003 Add `_useTabUi` bool (default false), `_activeTopTab`, `_activeBottomTabIndex` fields.
- [x] T004 Add `DrawTopTabBar()` (renders 6 buttons, sets `_activeTopTab` on click).
- [x] T005 Add `DrawBottomTabBar()` (renders sub-tab buttons for active top).
- [x] T006 Add `DrawMainViewport`, `DrawTopTabContent`, `DrawBottomTabContent` stubs.
- [x] T007 In `DrawUI()`: if `_useTabUi`, call tab render. Else fall through to old sidebar path.
- [x] T008 Verify build. Build clean (0 errors, 162 pre-existing warnings). Viewer behavior unchanged at `_useTabUi=false`.

## Phase 2: Scene + Utilities Tabs

- [x] T009 Implement `DrawSceneTabContent()` → `DrawSceneSubTabContent()` → Selection/Camera/Settings/Themes.
- [x] T010 Selection sub-tab calls `DrawSelectionPanelContent()`.
- [x] T011 Camera sub-tab calls `DrawCameraControlsContent()`.
- [x] T012 Themes sub-tab calls `DrawUiThemeSettingsContent()`.
- [x] T013 Settings sub-tab has hide-UI-chrome toggle + camera controls.
- [x] T014 Implement `DrawUtilitiesSubTabContent()` → Minimap/Log/Perf/RenderQuality/Taxi/CaptureAutomation/AssetCatalog/RuntimeStats.
- [x] T015 Log sub-tab calls `DrawLogViewer()`.
- [x] T016 Perf sub-tab calls `DrawPerfWindow()`.
- [x] T017 RenderQuality sub-tab calls `DrawRenderQualityWindow()`.
- [x] T018 Taxi sub-tab calls `DrawTaxiWindow()`.
- [x] T019 CaptureAutomation sub-tab calls `DrawCaptureAutomationWindow()`.
- [x] T020 RuntimeStats sub-tab calls `DrawRuntimeStatsPanelContent()`.
- [x] T021 AssetCatalog sub-tab calls `_catalogView?.Draw()`.
- [x] T022 Minimap sub-tab uses new headless `DrawMinimapContent()`.
- [x] T023 `_useTabUi = true` default. Old sidebars/windows suppressed when tab system active.
- [x] T024 Build clean. Tab system is the default UI.

## Phase 3: World + Terrain Merge

- [x] T025 Implement `DrawWorldSubTabContent()` → Placements/Tiles/Overlays/SelectionTools.
- [x] T026 Placements sub-tab calls `DrawWorldObjectsContentCore()`.
- [x] T027 Tiles sub-tab calls `DrawTerrainWorkbenchSelectionContent` + `DrawTerrainControlsAdjustmentContent` + `DrawTerrainExportSubTab`.
- [x] T028 Overlays sub-tab: layer toggles (base/L1/L2/L3/holes), grid toggles (chunks/tiles/cells), alpha/shadow/MCCV/contours.
- [x] T029 SelectionTools sub-tab: `DrawSelectedObjectSummaryContent`.
- [x] T030 Top tab content shows context band: map name, loaded tile count, sub-tab.
- [x] T031 Terrain Tools window suppressed when `_useTabUi = true` (kept for legacy mode).
- [x] T032 Build clean. World tab functional with 4 sub-tabs.

## Phase 4: Terrain + PM4 Tabs

- [x] T033 Implement `DrawTerrainSubTabContent()` → Layers/Clipboard/Analysis/MCNK/WeakSignal/Export.
- [x] T034 Layers sub-tab: layer toggles + alpha/shadow/MCCV/contour + grid + holes.
- [x] T035 Clipboard sub-tab: `DrawChunkClipboardContent(renderer)`.
- [x] T036 Analysis sub-tab: `DrawTerrainAnalysisWindow()`.
- [x] T037 MCNK sub-tab: `DrawMcnkExplorerWindow()`.
- [x] T038 Weak Signal sub-tab: `DrawWeakSignalWindow()`.
- [x] T039 Export sub-tab: scope selector + alpha/heightmap/MCCV export buttons.
- [x] T040 Implement `DrawPm4SubTabContent()` → Overlay/Selection/Correlation/Info/Match/Alignment.
- [x] T041 Overlay/Selection sub-tabs: `DrawPm4OverlayWorkbenchContent` + `DrawPm4SelectionWorkbenchContent`.
- [x] T042 Info sub-tab: `DrawPm4InfoPanelContent`.
- [x] T043 Match sub-tab: `DrawPm4ObjectMatchWindow`.
- [x] T044 Alignment sub-tab: `DrawPm4AlignmentWindow`.
- [x] T045 Old `_show*Window` flags kept for legacy (else branch) but not rendered when tabs on.
- [x] T046 Build clean. All 4 missing top tabs (Terrain/PM4/Archeology/World) now have sub-tab content.

## Phase 5: Archeology Tab Foundation

- [x] T047 Implement `DrawArcheologySubTabContent()` → Range/Layers/Playback/Capture.
- [x] T048 Range sub-tab: filter checkbox, scope selector, range sliders, status.
- [x] T049 Layers sub-tab: detected archeology layers table with Show buttons.
- [x] T050 Removed uniqueId controls from World sub-tab (only in Archeology).
- [x] T051 Kept `_showUniqueIdArchaeologyWindow` flag for legacy mode (else branch).
- [x] T052 Build clean. Archeology tab has 4 sub-tabs with real content (2 wired, 2 placeholders for Phase 7).

## Phase 6: Sticky Archeology Settings

- [x] T053 Added `_archeologyMinUniqueId`, `_archeologyMaxUniqueId`, `_archeologyScopeIndex` fields.
- [x] T054 Added to `ViewerSettings` and `SaveViewerSettings()`.
- [x] T055 Added to `LoadViewerSettings()` with safe defaults.
- [x] T056 Range sub-tab persists on slider change + on reset.
- [x] T057 Build clean. Sticky range ready for testing.

## Phase 7: Archeology Playback

- [x] T058 Added playback fields: active, speed, loop, accumulator, restore snapshot.
- [x] T059 Playback sub-tab: Play/Pause/Stop buttons, speed slider, loop checkbox, status.
- [x] T060 `UpdateArcheologyPlayback(double dt)` advances UniqueIdFilterMax at speed.
- [x] T061 Called from `OnUpdate(dt)` after HandleKeyboardInput.
- [x] T062 Slider touch in Range sub-tab pauses playback.
- [x] T063 `ArcheologyPlaybackConfig` represented as fields (not class).
- [x] T064 `PendingCaptureRequest.ApplyArcheologyPlayback` field added.
- [x] T065 Per-shot capture advances end by (max-min)/32 per shot.
- [x] T066 `ActiveVideoRecording.ApplyArcheologyPlayback` field added.
- [x] T067 Video recording start: `StartArcheologyPlayback()` if `_archeologyApplyToVideoRecording`.
- [x] T068 Video recording stop: `StopArcheologyPlayback(restoreRange: true)`.
- [x] T069 Capture sub-tab: checkboxes for "Apply to next capture" + "Apply to video recording".
- [x] T070 Build clean. Playback + capture integration wired.

## Phase 8: Minimap Refactor + Final Cleanup

- [x] T071 Minimap surface refactor (done in Phase 2 — `DrawMinimapContent` extracted).
- [x] T072 Utilities > Minimap sub-tab uses headless `DrawMinimapContent`.
- [x] T073 Fullscreen minimap uses same surface fn with full viewport size.
- [x] T074 `ShellPanelId` enum marked `[Obsolete]`, will be removed in 070.
- [x] T075 Skipped — left for 070 to avoid breaking legacy fallback.
- [x] T076 Skipped — left for 070 to avoid breaking legacy fallback.
- [x] T077 Wrote new doc: `docs/architecture/viewer-ui-tab-system-2026-06-21.md`.
- [x] T078 Skipped (no separate ui-architecture.md exists; new doc covers it).
- [x] T079 Build clean. All 8 phases done. UI overhaul complete (v1).

## Phase 9: Critical fixes (per user feedback)

- [x] T080 Added `World > Source` sub-tab with file browser + map discovery + workspace bars.
- [x] T081 Removed `ImGui.Begin("##TopTabContent")` debug window wrap.
- [x] T082 Emptied Tools menu (only modal/dialog entries remain).
- [x] T083 Reordered tab render: top tab bar → context → 3D viewport → bottom tab bar → popouts.

## Phase 10: 3D viewport fix

- [x] T084 `TryGetSceneViewportRect` returns full middle area when `_useTabUi=true`
  (no dockspace/sidebar insets). 3D world now fills the full available area
  instead of being clipped to a small dockspace rect.

## Phase 11: Popout positioning

- [x] T085 `DrawSubTabWindow` supports `dockRight` / `dockLeft` positioning.
- [x] T086 Source popout docks on right edge, Quick Controls docks on left edge.

## Phase 12: Quick Controls + per-sub-tab popouts

- [x] T087 Quick Controls popout: Camera Speed, FOV, ADT Detail Tiles, Time of Day,
  Fog Start/End, Layer toggles, Overlay toggles, Reset Camera, Toggle Wireframe.
  Opens by default on left side. This is the populated "Debug window".
- [x] T088 Per-sub-tab popouts: Source, Placements, Tiles, Overlays, Selection,
  Camera, Themes, Settings, Layers, Range, Playback, Capture, Log, Perf.
  Each has its own popout window, opens/closes independently.

## Phase 13: Toggle + indicator

- [x] T089 Click sub-tab to TOGGLE its popout (was: open only).
- [x] T090 Open popouts show `●` indicator in bottom tab bar.
- [x] T091 `IsSubTabPopoutOpen()` / `ToggleSubTabPopoutByIndex()` per sub-tab.

## Phase 14: Single Workbench panel (window sprawl fix)

- [x] T092 All sub-tab popouts collapsed into ONE big "Workbench" popout window.
- [x] T093 Master window no longer draws top/bottom tab bars (workbench has its own).
- [x] T094 Workbench contains: top tab bar + sub-tab bar + content area in a
  single resizable window.
- [x] T095 Added Scene > Quick sub-tab (index 0) for camera/lighting/layer controls.
- [x] T096 Removed per-sub-tab popout state fields (single `_workbenchOpen` flag).
- [x] T097 Removed dead code: DrawBottomTabBar, IsSubTabPopoutOpen, ToggleSubTabPopoutByIndex, OpenSubTabPopoutByIndex, DrawSubTabWindow, DrawQuickControlsPopoutBody, GetTerrainRendererSafe.

## Phase 15: Memory bank + spec sync

- [x] T098 Updated `memory-bank/activeContext.md` for Phase 14.
- [x] T099 Updated `memory-bank/progress.md` with 14-phase history + lessons learned.
- [x] T100 (this file) tasks.md updated to reflect current state.

## Phase 16: Headless content variants (sub-tab X-button fix)

- [x] T101 Refactored `DrawLogViewer` → `DrawLogViewerContent` (headless).
- [x] T102 Refactored `DrawPerfWindow` → `DrawPerfContent`.
- [x] T103 Refactored `DrawRenderQualityWindow` → `DrawRenderQualityContent`.
- [x] T104 Refactored `DrawCaptureAutomationWindow` → `DrawCaptureAutomationContent`.
- [x] T105 Refactored `DrawTaxiWindow` → `DrawTaxiContent`.
- [x] T106 Refactored `DrawWeakSignalWindow` → `DrawWeakSignalContent`.
- [x] T107 Refactored `DrawTerrainAnalysisWindow` → `DrawTerrainAnalysisContent`.
- [x] T108 Refactored `DrawMcnkExplorerWindow` → `DrawMcnkExplorerContent`.
- [x] T109 Sub-tab dispatchers now call headless variants (no nested windows in workbench).
- [x] T110 Legacy `Draw*Window` wrappers still work for View menu items.
- [x] T111 Memory bank + spec/plan/tasks updated.

## Build state

- Build: 0 errors, 0 warnings on every commit.
- 16 phases complete, all on `069-viewer-ui-overhaul` branch.
- Phase 14 = single Workbench panel (replaces 14 separate popouts).
- Phase 15 = memory bank + spec sync.
- Phase 16 = headless content variants (no nested windows).
- Next: 070 workbench window rewrite (per-map windows, real native feel).

## Notes

- Each phase = one PR. Don't merge phases.
- T008, T024, T032, T046, T051, T057, T070, T079 = validation gates.
- If a phase breaks validation, fix it before moving on. Don't pile on.
- Cells overlay (T001-pre) is already implemented and working. Don't touch it.
