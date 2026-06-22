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

- [ ] T009 Implement `DrawSceneTabContent()` → `DrawSceneSubTab_Selection()`, `DrawSceneSubTab_Camera()`, `DrawSceneSubTab_Settings()`, `DrawSceneSubTab_Themes()`.
- [ ] T010 Move content from `DrawUnifiedSelectionSidebarContent` into `DrawSceneSubTab_Selection()`.
- [ ] T011 Move `DrawCameraControlsContent` into `DrawSceneSubTab_Camera()`.
- [ ] T012 Move `DrawUiThemeSettingsContent` (ViewerApp_Themes.cs) into `DrawSceneSubTab_Themes()`.
- [ ] T013 Move workspace/target/save summary into `DrawSceneSubTab_Settings()`.
- [ ] T014 Implement `DrawUtilitiesTabContent()` → sub-tabs Minimap/Log/Perf/Render Quality/Taxi/Capture Automation/Asset Catalog/Runtime Stats.
- [ ] T015 Move `_showLogViewer` flag content into `DrawUtilitiesSubTab_Log()`.
- [ ] T016 Move `_showPerfWindow` content into `DrawUtilitiesSubTab_Perf()`.
- [ ] T017 Move `_showRenderQualityWindow` content into `DrawUtilitiesSubTab_RenderQuality()`.
- [ ] T018 Move `_showTaxiWindow` content into `DrawUtilitiesSubTab_Taxi()`.
- [ ] T019 Move `_showCaptureAutomationWindow` content into `DrawUtilitiesSubTab_CaptureAutomation()`.
- [ ] T020 Move runtime stats draw into `DrawUtilitiesSubTab_RuntimeStats()`.
- [ ] T021 Move asset catalog/file browser into `DrawUtilitiesSubTab_AssetCatalog()`.
- [ ] T022 Minimap sub-tab: call `DrawInteractiveMinimapSurface` with sub-tab content size.
- [ ] T023 Set `_useTabUi = true` default. Remove old sidebar `_showFlag` wiring for Utilities.
- [ ] T024 Verify build, manual smoke.

## Phase 3: World + Terrain Merge

- [ ] T025 Implement `DrawWorldTabContent()` → Placements/Tiles/Overlays/SelectionTools sub-tabs.
- [ ] T026 Move `DrawWorldObjectsContentCore` body into `DrawWorldSubTab_Placements()`.
- [ ] T027 Move `DrawTerrainToolsContent` body into `DrawWorldSubTab_Tiles()`.
- [ ] T028 New `DrawWorldSubTab_Overlays()`: chunk/tile/cell grid toggles, alpha mask, shadow, MCCV, contours.
- [ ] T029 New `DrawWorldSubTab_SelectionTools()`: click selection, frame, asset path actions.
- [ ] T030 Remove `ShellPanelId.WorldObjects` from active use. Mark `[Obsolete]`.
- [ ] T031 Remove `_showTerrainToolsWindow` flag. Move content to sub-tab.
- [ ] T032 Verify build, manual smoke.

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
