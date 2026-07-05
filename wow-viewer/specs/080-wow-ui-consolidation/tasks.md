# Tasks: Spec 080 WoW UI Consolidation

**Input**: `spec.md`, `plan.md`

**Prerequisites**: Current implementation stays in `wow-viewer`; `gillijimproject_refactor` is reference-only.

**Tests**: Build validation is required for each source-changing phase. Manual viewer checks are required for UI behavior.

## Phase 1: Bottom Bar And WMO Inspection (P1)

**Goal**: Make the existing high-value inspection controls visible and independently usable.

**Independent Test**: Load a standalone WMO and a world map; verify WMO group overlays/names and terrain/object wireframes can be toggled without opening the right sidebar.

- [x] T001 [US1] Add standalone WMO group-name state in `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`.
- [x] T002 [US1] Add bottom-bar controls for standalone WMO wireframe, WMO group bounding boxes, and group names in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- [x] T003 [US1] Render all standalone WMO group names when enabled in `wow-viewer/src/viewer/WoWViewer/ViewerApp_WmoGroups.cs`.
- [x] T004 [US1] Split world wireframe setters into terrain and M2/WMO object controls in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`.
- [x] T005 [US1] Build-check with `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`.
- [ ] T006 [US1] Manual-check standalone WMO: all group names appear, group boxes toggle, hidden-group inclusion still works.
- [ ] T007 [US1] Manual-check world map: `Terrain WF` and `M2/WMO WF` can be mixed independently.

## Phase 2: Settings Surface (P1)

**Goal**: Make Settings real, reachable, and not a dead click target.

**Independent Test**: Click File -> Settings and the bottom-bar Settings launcher; both open the same persistent Settings window.

- [x] T008 [US3] Add a bottom-bar Settings launcher in `ViewerApp_Sidebars.cs`.
- [ ] T009 [US3] Confirm File -> Settings and any sidebar Settings button only set `_showSettingsWindow`.
- [ ] T010 [US3] Add Camera defaults to `ViewerApp_Settings.cs`.
- [ ] T011 [US3] Verify render quality, fog, interface, and camera settings save through the existing settings path.

## Phase 3: Right Sidebar Audit (P1)

**Goal**: Make the messy right sidebar actionable before deleting or moving it.

**Independent Test**: Every right-sidebar/workbench tab has one owner classification and no invisible duplicate ownership.

- [ ] T012 [US2] Audit `DrawRightSidebar()` and legacy `##LegacyRightSidebar` dispatch in `ViewerApp_Sidebars.cs`.
- [ ] T013 [US2] Classify all Model tab content as Info, Animations, Actions, LOD, or remove.
- [ ] T014 [US2] Classify all World tab content as Source, Placements, Tiles, LOD, Selection Tools, or remove.
- [ ] T015 [US2] Classify Tools content as PM4, Terrain, Archeology, Utilities, or remove.
- [ ] T016 [US6] Hide or disable dead controls with tooltips instead of showing fake working buttons.
- [ ] T017 [US2] Add `WorldBottomTab.Lod` and label it `LOD` in `Workbench/WorkbenchNavigator.cs`.

## Phase 4: Model And World Info Tabs (P2)

**Goal**: Put factual inspection panels where users expect them.

**Independent Test**: With a model loaded, model facts are under Model -> Info/LOD; with a world loaded, world facts are under World -> Info/LOD-related tabs.

- [ ] T018 [US2] Move standalone WMO group controls into the Model info tab while keeping bottom-bar toggles.
- [ ] T019 [US2] Move WMO/MDX placement details into World info/placement tabs without duplicating bottom-bar toggles.
- [ ] T020 [US2] Replace the placeholder Model LOD text with actual available model LOD/runtime facts or a disabled state.
- [ ] T021 [US2] Add World LOD tab content for WDL/world-distance state or explicitly disabled text with the missing data path.

## Phase 5: Named Frames (P2)

**Goal**: Replace the right sidebar with stable named frames after content ownership is clean.

**Independent Test**: Opening Model, World, Terrain, PM4, Archeology, Utilities, and Settings creates stable windows that stay open after clicking the viewport.

- [ ] T022 [US2] Extract `DrawModelFrame()` from current Model tab content.
- [ ] T023 [US2] Extract `DrawWorldFrame()` from current World tab content.
- [ ] T024 [US2] Extract `DrawTerrainFrame()`, `DrawPm4Frame()`, `DrawArcheologyFrame()`, and `DrawUtilitiesFrame()` from Tools content.
- [ ] T025 [US2] Route Tools menu entries to `_show*Frame` booleans.
- [ ] T026 [US2] Keep left sidebar enabled until frame migration is build-validated and manually checked.
- [ ] T027 [US2] Remove right-sidebar dispatch only after named frames pass validation.

## Phase 6: Documentation And Continuity (P1)

**Goal**: Keep later sessions from routing Spec 080 back to stale legacy work.

- [x] T028 [US6] Update `wow-viewer/memory-bank/activeContext.md` with the current 080 owner and phase state.
- [x] T029 [US6] Update `wow-viewer/memory-bank/progress.md` with completed source and proof status.
- [ ] T030 [US6] Update `spec.md` status once Phase 1 build/manual proof is complete.

## Dependencies

- Phase 1 can ship independently.
- Phase 2 depends only on current Settings window wiring.
- Phase 3 must finish before Phase 5 deletes or bypasses right-sidebar content.
- Phase 4 depends on Phase 3 classification.
- Phase 5 depends on Phase 3 and should not include left-sidebar removal.
- Phase 6 happens after each completed phase, not only at the end.

## Notes

- Do not modify `gillijimproject_refactor` for this feature.
- Do not move file/world loading out of the left sidebar in this slice.
- Treat World LOD as missing until the audit identifies the actual WDL/LOD facts to expose.
- Do not claim UI runtime proof from source edits alone.
