# Tasks: 060 Viewer UI Cleanup + ImGui Migration Notes

## Phase 1: Migration Note (doc-only)

- [x] T001 Write `wow-viewer/docs/architecture/ui-migration-options-2026-06-12.md` with rationale, options table, cost estimate, "no commitment" disclaimer

## Phase 2: Runtime Stats Dedup

- [ ] T002 Grep all call sites of `DrawRuntimeStatsPanelContent` in `ViewerApp_Sidebars.cs`

- [ ] T003 Remove Runtime Stats from `DrawNavigatorPanelContent` (left sidebar)

- [ ] T004 Remove "Stats" tab from `DrawUnifiedToolSidebar` Inspector tab bar

- [ ] T005 Remove "Stats" tab from `DrawSceneInspectorPanelContent` (deferred to Phase 5 if SceneInspector removed)

- [ ] T006 Remove trailing Runtime Stats block from `DrawTerrainControlsContent`

- [ ] T007 [P] Memory bank note about Runtime Stats location

## Phase 3: Status Bar Button Removal

- [ ] T008 In `DrawStatusBar()` at `ViewerApp_MinimapAndStatus.cs:237`, remove the "Actions" column with Copy Scene / Log Scene buttons

- [ ] T009 Reduce status bar `ImGui.BeginTable` to 2-3 columns (status, FPS, optional coords)

- [ ] T010 Verify Capture Automation window still has Copy/Log Current Scene Bookmark buttons at `ViewerApp_CaptureAutomation.cs:233` and `:333`

## Phase 4: Capture UI-Hide Default

- [ ] T011 In `PrepareNextCaptureRequest()` at `ViewerApp_CaptureAutomation.cs:564`, set `_hideUiChrome = true` for `includeUi: false` (default) requests

- [ ] T012 Restore `_hideUiChrome = false` after capture frame completes (in `CompleteCaptureIfReady`)

- [ ] T013 Apply same toggle per-frame for video recording in `CaptureVideoFrameIfNeeded`

- [ ] T014 Validate: video capture default produces UI-hidden frames; `with_ui: true` still captures chrome

## Phase 5: SceneInspector Dedup

- [ ] T015 Compare SceneInspector tab bar (`ViewerApp_Sidebars.cs:1051`) vs Inspector tab bar (`ViewerApp_Sidebars.cs:802`)

- [ ] T016 Decision: remove SceneInspector (if overlap > 50%) or restructure

- [ ] T017 [P] If removing: delete `ShellPanelId.SceneInspector`, related fields, dispatch cases, quadrant grouping, Tools menu entry

## Phase 6: Polish

- [ ] T018 [P] Full build: `dotnet build` clean

- [ ] T019 [P] Viewer smoke: launch, verify all panels render, capture works

- [ ] T020 [P] Memory bank update

## Dependencies

- Phase 1: none
- Phase 2: independent
- Phase 3: independent
- Phase 4: independent
- Phase 5: Phase 2 (must remove Stats from SceneInspector tabs first if keeping SceneInspector)
- Phase 6: depends on all
