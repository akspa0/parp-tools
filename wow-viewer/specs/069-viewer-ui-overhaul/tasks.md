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

- [ ] T033 Implement `DrawTerrainTabContent()` → Layers/Clipboard/Analysis/MCNK/Weak Signal/Export.
- [ ] T034 Move `DrawTerrainLayersContent` into `DrawTerrainSubTab_Layers()`.
- [ ] T035 Move `_showChunkClipboardWindow` content into `DrawTerrainSubTab_Clipboard()`.
- [ ] T036 Move `_showTerrainAnalysisWindow` content into `DrawTerrainSubTab_Analysis()`.
- [ ] T037 Move `_showMcnkExplorerWindow` content into `DrawTerrainSubTab_MCNK()`.
- [ ] T038 Move `_showWeakSignalWindow` content into `DrawTerrainSubTab_WeakSignal()`.
- [ ] T039 New `DrawTerrainSubTab_Export()`: export scope, format, output dir.
- [ ] T040 Implement `DrawPm4TabContent()` → Overlay/Selection/Correlation/Info/Match/Alignment.
- [ ] T041 Move PM4 Workbench content into `DrawPm4SubTab_Overlay/Selection/Correlation()`.
- [ ] T042 Move `_showPm4Info` content into `DrawPm4SubTab_Info()`.
- [ ] T043 Move `_showPm4ObjectMatchWindow` content into `DrawPm4SubTab_Match()`.
- [ ] T044 Move `_showPm4AlignmentWindow` content into `DrawPm4SubTab_Alignment()`.
- [ ] T045 Remove all old `_show*Window` flags for above panels.
- [ ] T046 Verify build, manual smoke.

## Phase 5: Archeology Tab Foundation

- [ ] T047 Implement `DrawArcheologyTabContent()` → Range/Layers/Capture/Playback sub-tabs.
- [ ] T048 Move `DrawUniqueIdArchaeologyContent` body into `DrawArcheologySubTab_Range()`.
- [ ] T049 New `DrawArcheologySubTab_Layers()`: detected archeology layers table.
- [ ] T050 Remove uniqueId controls from `DrawWorldSubTab_Placements()`.
- [ ] T051 Remove `_showUniqueIdArchaeologyWindow` flag.
- [ ] T052 Verify build, manual smoke.

## Phase 6: Sticky Archeology Settings

- [ ] T053 Add `_archeologyMinUniqueId`, `_archeologyMaxUniqueId`, `_archeologyScopeIndex` fields.
- [ ] T054 Add to settings save: `SaveViewerSettings()` writes archeology section.
- [ ] T055 Add to settings load: `LoadViewerSettings()` reads archeology section.
- [ ] T056 On change in `DrawArcheologySubTab_Range()`, persist immediately.
- [ ] T057 Verify: set range, restart viewer, range restored.

## Phase 7: Archeology Playback

- [ ] T058 Add `_archeologyPlaybackActive`, `_archeologyPlaybackSpeed` (units/sec), `_archeologyPlaybackLoop` fields.
- [ ] T059 Implement `DrawArcheologySubTab_Playback()`: Play/Pause/Stop buttons, speed slider, loop checkbox.
- [ ] T060 Add `UpdateArcheologyPlayback(double deltaTime)`: advance `_worldScene.UniqueIdFilterMax` at speed.
- [ ] T061 Call `UpdateArcheologyPlayback` in `Update()` when `_worldScene != null`.
- [ ] T062 On slider touch in `DrawArcheologySubTab_Range()`, set `_archeologyPlaybackActive = false` (pause).
- [ ] T063 Add `ArcheologyPlaybackConfig` class: `Enabled`, `Speed`, `Loop`, `FrameStep`.
- [ ] T064 Add `PendingCaptureRequest.ArcheologyPlayback` field.
- [ ] T065 In capture sequence: if `ArcheologyPlayback.Enabled`, advance end per shot at `FrameStep`.
- [ ] T066 Add `VideoRecordingConfig.ArcheologyPlayback` (bool) + `ArcheologyPlaybackSpeed` (float).
- [ ] T067 On video recording start: if `ArcheologyPlayback == true`, kick off playback at real-time speed.
- [ ] T068 On video recording end: stop playback, restore previous `UniqueIdFilterMax`.
- [ ] T069 New `DrawArcheologySubTab_Capture()`: "Apply to next capture" + "Apply to video recording" checkboxes + speed input.
- [ ] T070 Verify: playback works, capture integration works, video recording with playback captures progression.

## Phase 8: Minimap Refactor + Final Cleanup

- [ ] T071 Refactor `DrawInteractiveMinimapSurface` to accept explicit `(width, height)`. Already does — verify.
- [ ] T072 `DrawUtilitiesSubTab_Minimap()`: get content region size, call surface fn.
- [ ] T073 Fullscreen minimap: use full viewport size, same surface fn.
- [ ] T074 Remove `ShellPanelId` enum, `ShellPanelDefinition` records, dock state, sidebar methods. Keep `[Obsolete]` for one release.
- [ ] T075 Remove `_leftSidebarWidth`, `_rightSidebarWidth`, `_pendingRightSidebarSection`.
- [ ] T076 Remove `DrawLeftSidebar`, `DrawRightSidebar` methods.
- [ ] T077 Update `docs/architecture/viewer-ui-panels-reference.md` with tab system.
- [ ] T078 Update `docs/architecture/viewer-ui-architecture.md` with new layout description.
- [ ] T079 Final build + smoke test. All tabs work, no crashes, all sub-tabs render.

## Notes

- Each phase = one PR. Don't merge phases.
- T008, T024, T032, T046, T051, T057, T070, T079 = validation gates.
- If a phase breaks validation, fix it before moving on. Don't pile on.
- Cells overlay (T001-pre) is already implemented and working. Don't touch it.
