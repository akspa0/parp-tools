# Tasks: WoWViewer UI Overhaul

**Input**: Design documents from `wow-viewer/specs/145-wow-ui-overhaul/`
**Tests**: Build and focused source checks are required; real-client visual checks remain user-run.

## Phase 1: Setup and foundational context

- [x] T001 Record the current dirty-tree boundary and preserve unrelated changes in `wow-viewer/specs/145-wow-ui-overhaul/research.md`.
- [x] T002 Add the viewer keybinding context/catalog contract in `wow-viewer/src/viewer/WoWViewer/ViewerKeyBindings.cs`.
- [x] T003 Add active main/nested page context state to `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`.

## Phase 2: User Story 1 — Contextual shortcuts and help (P1)

**Independent test**: Help > Keyboard Shortcuts shows the active context; Capture authoring keys do nothing outside Capture.

- [x] T004 [US1] Register global, Model, World, Tools, Utilities, and Capture binding records in `wow-viewer/src/viewer/WoWViewer/ViewerKeyBindings.cs`.
- [x] T005 [US1] Gate `HandleCameraPathKeyboardInput` on the active Capture context in `wow-viewer/src/viewer/WoWViewer/ViewerApp_CameraPaths.cs`.
- [x] T006 [US1] Add the Help > Keyboard Shortcuts launcher and persistent closeable dialog in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`.
- [x] T007 [US1] Update active context when main and nested workbench pages are selected in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- [ ] T008 [US1] Add focused context/catalog tests in `wow-viewer/tests/WowViewer.Core.Tests/ViewerKeyBindingTests.cs` if the binding catalog is placed in a testable shared seam.

## Phase 3: User Story 2 — Stable WoW-like shell navigation (P1)

**Independent test**: Model, World, and Tools are selectable from a vertical rail at compact width without horizontal main-tab overflow.

- [x] T009 [US2] Replace the main workbench tab strip with a vertical selector rail and content child in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- [x] T010 [US2] Keep nested page selection context-aware and bounded in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs` and `wow-viewer/src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs`.
- [x] T011 [US2] Add visual selected/hovered rail styling without changing bottom-bar ownership in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.

## Phase 4: User Story 3 — Useful bounded navigator and log (P1)

**Independent test**: The map list remains reachable with the minimap open; long log entries wrap and scroll vertically.

- [x] T012 [US3] Bound the world overview/minimap region and place file/map discovery in independent scrollable regions in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- [x] T013 [US3] Remove horizontal log overflow and wrap retained entries in `wow-viewer/src/viewer/WoWViewer/ViewerApp_LogViewer.cs`.
- [x] T014 [US3] Add a narrow-width layout regression note/check to `wow-viewer/specs/145-wow-ui-overhaul/quickstart.md`.

## Phase 5: User Story 4 — Persistent utility windows and honest pages (P2)

**Independent test**: A promoted utility window remains after a viewport click and closes with its X; placeholder pages identify unavailable data.

- [ ] T015 [US4] Centralize promoted utility-window drawing so visibility is independent of the selected page in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`.
- [ ] T016 [US4] Audit LOD and other placeholder/dead-control surfaces in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs` and `wow-viewer/src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs`.
- [ ] T017 [US4] Add close-state checks for the migrated utility windows and update `wow-viewer/specs/145-wow-ui-overhaul/quickstart.md`.

## Phase 6: User Story 5 — Truthful release surface (P2)

**Independent test**: About, project metadata, viewer README, root README, and release notes agree on v0.5.2 and pending proof boundaries.

- [x] T018 [P] [US5] Align version metadata in `wow-viewer/src/viewer/WoWViewer/WoWViewer.csproj`, `wow-viewer/src/viewer/WoWViewer/WoWViewer.CrossPlatform.csproj`, and shared project identity constants.
- [x] T019 [P] [US5] Update `wow-viewer/README.md` and `README.md` with current support, UI, capture, M2/MDX, WDL/WL*, and proof boundaries.
- [x] T020 [US5] Update `wow-viewer/docs/releases/v0.5.2.md` only where its current claims contradict the implemented v0.5.2 surface.

## Phase 7: Polish and continuity

- [x] T021 Build `wow-viewer/WowViewer.slnx` in Debug and record the result without claiming visual runtime proof.
- [x] T022 Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` with completed slice, remaining phases, and user-run gates.
- [x] T023 Recheck Spec 080 inventory references and mark Spec 145 as its bounded follow-up in `wow-viewer/docs/architecture/viewer-ui-surface-inventory.md` if required.

## Dependencies

- T002-T003 precede T004-T007.
- T007 precedes T009-T011 so page clicks establish key context.
- T009-T013 can be implemented as separate bounded UI slices, but build validation is required before T015.
- T018-T020 are independent documentation/version work but must reflect the implemented state.
- T021-T023 are final validation/continuity tasks.

## MVP scope

The first user-testable MVP is T002-T014 plus T018-T019: context-aware Capture shortcuts, visual help, vertical main rail, bounded navigator, usable log, and truthful version/readme surface. Persistent-window migration and placeholder audit remain the next phase.
