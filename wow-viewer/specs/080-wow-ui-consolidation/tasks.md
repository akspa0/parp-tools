# Tasks: Spec 080 WoW UI Consolidation

> **Release-convergence amendment (2026-07-11):** This task list records the
> earlier partial implementation. The canonical completion order is
> [`ui-release-convergence-plan.md`](ui-release-convergence-plan.md). Start
> with its Phase 0 inventory and route-integrity gate before resuming any
> unchecked sidebar/frame task below.

**Input**: `spec.md`, `plan.md`

**Prerequisites**: Current implementation stays in `wow-viewer`; `gillijimproject_refactor` is reference-only.

**Tests**: Build validation is required for each source-changing phase. Manual viewer checks are required for UI behavior.

## Phase 0R: Legacy Panel And Warning Disposition (P1)

**Goal**: Inventory and disposition old-panel migration debt before the next
GitHub release. Do not resolve it with blanket warning suppression.

**Independent Test**: The active viewer projects have a repeatable warning
report with every warning assigned to an active route, duplicate, retired
surface, or documented compatibility path.

- [ ] T031 [US6] Capture Debug and release-configuration warning inventories for `WowViewer.slnx` with project, warning ID, source location, and branch baseline.
- [ ] T032 [US2] Cross-reference warning locations with `docs/architecture/viewer-ui-surface-inventory.md` and classify old panel methods and dispatches.
- [ ] T033 [US2] Complete or remove active tab/sidebar routes whose warnings identify missing ownership; preserve legacy mode until its retirement gate passes.
- [ ] T034 [US2] Remove duplicate launchers and stale dispatch only after the replacement/retirement row is recorded in the surface inventory.
- [ ] T035 [US6] Resolve bounded active-viewer warning batches, starting with unused locals and dead methods, with a warning-delta check after each batch.
- [ ] T036 [US6] Add a release warning disposition report and explicit warning budget to the Spec 080 release proof package.
- [ ] T037 [US6] Verify the final GitHub release candidate in both supported UI modes and record restored, migrated, disabled, retired, and compatibility-only surfaces.

## Phase 0S: Runtime Status Strip (P1)

**Goal**: Keep the small set of high-value runtime facts visible without
opening the Runtime Stats tab.

**Independent Test**: With a world loaded, the lower status bar shows FPS,
AreaName, CPU frame time, tile/chunk counts, WMO/MDX visibility, and pending
asset loads in one compact right-aligned line.

- [x] T038 [US6] Move the compact runtime line from the bottom action bar to the lower status bar and reuse `WorldScene.LastRenderFrameStats` as its source.
- [ ] T039 [US6] Manually verify the line after world load, camera movement, tile streaming, and standalone terrain load; verify it does not overlap the coordinate text at supported window sizes.

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
- [x] T009 [US3] Confirm File -> Settings and any sidebar Settings button only set `_showSettingsWindow`.
- [x] T010 [US3] Add Camera defaults to `ViewerApp_Settings.cs`.
- [x] T011 [US3] Verify render quality, fog, interface, and camera settings save through the existing settings path.

## Phase 3: Right Sidebar Audit (P1)

**Goal**: Make the messy right sidebar actionable before deleting or moving it.

**Independent Test**: Every right-sidebar/workbench tab has one owner classification and no invisible duplicate ownership.

- [ ] T012 [US2] Audit `DrawRightSidebar()` and legacy `##LegacyRightSidebar` dispatch in `ViewerApp_Sidebars.cs`.
- [ ] T013 [US2] Classify all Model tab content as Info, Animations, Actions, LOD, or remove.
- [ ] T014 [US2] Classify all World tab content as Source, Placements, Tiles, LOD, Selection Tools, or remove.
- [ ] T015 [US2] Classify Tools content as PM4, Terrain, Archeology, Utilities, or remove.
- [ ] T016 [US6] Hide or disable dead controls with tooltips instead of showing fake working buttons.
- [x] T017 [US2] Add `WorldBottomTab.Lod` and label it `LOD` in `Workbench/WorkbenchNavigator.cs`.

## Phase 2A: Current Sidebar Information Architecture (P1)

**Goal**: Replace the implementation-history top row with five deliberate
destinations and give context facts one inline inspector owner.

**Independent Test**: In tabbed mode, the top row is exactly Quick, Inspect,
Scene, Utilities, Experimental. Each route opens a visible body, Inspect shows
available model/MCNK/PM4 context inline, and Experimental > Terrain Lab shows
tile/chunk targeting beside clipboard controls.

- [x] T040 [US2] Update `wow-viewer/src/viewer/WoWViewer/Workbench/WorkbenchTab.cs` and `Workbench/WorkbenchNavigator.cs` with the five canonical categories and page labels; retain compatibility enums only where existing callers require them.
- [x] T041 [US2] Route the tabbed workbench in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs` to Quick, Inspect, Scene, Utilities, and Experimental without rendering Model/World/Tools labels or duplicating Audio outside Utilities.
- [x] T042 [US2] Add `DrawUnifiedInspectorContent()` in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs` so selection/model, current ADT/MCNK, and selected PM4 summaries share one inline owner.
- [x] T043 [US2] Merge tile/chunk targeting and clipboard actions into the Experimental Terrain Lab route in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`, with no duplicate top-level tile or clipboard destination.
- [x] T044 [US2] Adapt existing menu, hotkey, capture, PM4, terrain, and model callers in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`, `ViewerApp.cs`, `ViewerApp_CameraPaths.cs`, and `ViewerKeyBindings.cs` to the compatibility mapping and new labels.
- [x] T045 [US2] Remove popup-only inspector navigation for the migrated context summaries and replace any required identity/coordinate reveal with inline text in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`.
- [x] T046 [US2] Update `wow-viewer/docs/architecture/viewer-ui-surface-inventory.md` with the five canonical tabbed destinations and the retired visible Model/World/Tools ownership rows.
- [x] T047 [US6] Run focused source checks and `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; record that visual/runtime/manual proof remains open. The full solution test command timed out; the focused core suite remains red on nine pre-existing failures unrelated to this UI slice.

## Phase 2B: Sidebar Entry-Point Convergence (P1)

**Goal**: Make every main Panels menu item land on the exact canonical page it names.

**Independent Test**: In tabbed mode, opening Log Viewer, Perf, Asset Catalog, or Taxi selects
Utilities and the matching utility page without a second navigation action.

- [x] T048 [US2] Add a typed `UtilitiesBottomTab` workbench adapter that selects Utilities and the requested utility page.
- [x] T049 [US2] Route the main Panels menu's Log Viewer, Perf, Asset Catalog, and Taxi entries through the typed adapter; preserve Capture/Camera Path routing through the existing Capture adapter.
- [ ] T050 [US6] Manually verify each Panels entry lands on the named page at normal and compact sidebar widths.

## Phase 2C: Utilities Ownership And Animation Restoration (P1)

**Goal**: Keep diagnostics and audio under one Utilities destination and restore
the existing MDX/M2 animation surface in the unified Inspect body.

**Independent Test**: The top row contains Utilities but not Audio; Utilities
has one page selector containing Audio; Inspect exposes animation controls for
a standalone model and a selected world MDX instance when available.

- [x] T051 [US2] Promote Utilities to a canonical top-level workbench destination and move Audio ownership under its page selector without changing playback policy.
- [x] T052 [US2] Restore the existing MDX/M2 animation controls inside the unified Inspect body for standalone and selected world-model contexts.
- [ ] T053 [US6] Manually verify Utilities -> Audio, Experimental pages, and Inspect animation controls at normal and compact sidebar widths.

## Phase 4: Model And World Info Tabs (P2)

**Goal**: Put factual inspection panels where users expect them.

**Independent Test**: With a model loaded, model facts are under Model -> Info/LOD; with a world loaded, world facts are under World -> Info/LOD-related tabs.

- [ ] T018 [US2] Move standalone WMO group controls into the Model info tab while keeping bottom-bar toggles.
- [ ] T019 [US2] Move WMO/MDX placement details into World info/placement tabs without duplicating bottom-bar toggles.
- [ ] T020 [US2] Replace the placeholder Model LOD text with actual available model LOD/runtime facts or a disabled state.
- [x] T021 [US2] Add World LOD tab content for WDL/world-distance state or explicitly disabled text with the missing data path.

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
- [x] T031 [US6] Replace compact-width horizontal workbench and capture tab strips with visible vertical rails so every page remains directly reachable.

## Dependencies

- Phase 1 can ship independently.
- Phase 2 depends only on current Settings window wiring.
- Phase 3 must finish before Phase 5 deletes or bypasses right-sidebar content.
- Phase 4 depends on Phase 3 classification.
- Phase 5 depends on Phase 3 and should not include left-sidebar removal.
- Phase 6 happens after each completed phase, not only at the end.
- Phase 2A depends on the Phase 3 route inventory but does not retire legacy
  dispatch. It must pass its source/build/manual gate before any old top-level
  label or content method is deleted.

## Notes

- Do not modify `gillijimproject_refactor` for this feature.
- Do not move file/world loading out of the left sidebar in this slice.
- Treat World LOD as missing until the audit identifies the actual WDL/LOD facts to expose.
- Do not claim UI runtime proof from source edits alone.
