# Tasks: 060 Viewer UI Cleanup + ImGui Migration Notes

## Phase 1: Migration Note (doc-only)

- [x] T001 Write `wow-viewer/docs/architecture/ui-migration-options-2026-06-12.md` (committed)

## Phase 2: Runtime Stats Dedup

- [x] T002 Grep all call sites — found 5 (SHIPPED `6f47f0e9`)
- [x] T003 Remove from Navigator sidebar (SHIPPED `6f47f0e9`)
- [x] T004 Remove "Stats" tab from Inspector tab bar (SHIPPED `6f47f0e9`)
- [x] T005 Remove "Stats" tab from SceneInspector tab bar (SHIPPED `6f47f0e9`)
- [x] T006 Remove trailing Runtime Stats block from Terrain Controls (SHIPPED `6f47f0e9`)
- [x] T007 Memory bank note

## Phase 3: Status Bar Button Removal

- [x] T008 Remove "Actions" column (Copy Scene / Log Scene buttons) from `DrawStatusBar()` (SHIPPED `db021b72`)
- [x] T009 Reduce status bar to 3 columns: Status, Coords, Meta (SHIPPED `db021b72`)
- [x] T010 Copy/Log Scene still available in Capture Automation window (verified — `ViewerApp_CaptureAutomation.cs:234, 237`)

## Phase 4: Capture UI-Hide Default

- [x] T011 Set `_hideUiChrome = true` in `PrepareNextCaptureRequest` for `IncludeUi: false` (SHIPPED `adeb48f1`)
- [x] T012 Restore `_hideUiChrome = false` in `CompleteCaptureIfReady` (SHIPPED `adeb48f1`)
- [x] T013 Apply same toggle in `TryStartCurrentViewVideoRecording` / `StopVideoRecording` (SHIPPED `adeb48f1`)
- [x] T014 `with_ui: true` flag still works (verified — `includeUi` param preserved)

## Phase 5: SceneInspector Dedup

- [x] T015 Compared tab bars — 75% overlap (SHIPPED `21f08716`)
- [x] T016 Decision: remove SceneInspector (SHIPPED `21f08716`)
- [x] T017 Removed enum entry, dock state, panel content, quadrant grouping, Tools menu, all related fields (SHIPPED `21f08716`)

## Phase 6: Polish

- [x] T018 Full build: `dotnet build` clean (0 errors)
- [x] T019 Runtime Stats now in 1 place (the dedicated panel at line 1014)
- [x] T020 Memory bank updated
