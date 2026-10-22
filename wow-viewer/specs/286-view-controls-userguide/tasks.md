# Tasks: View Controls and Contextual Disclosure

**Input:** [specification](spec.md), [approved plan](plan.md), [research](research.md), [data model](data-model.md), [local contracts](contracts/contextual-view.md), [verification cards](quickstart.md).
**State:** Unchecked implementation handoff, 2026-10-22. No source, test, runtime or screenshot completion is asserted.
**Order:** U0 → U1 → U2 → U3 → U4 → U5; eight tasks per phase, 48 total.

## Scope, setup and task conventions

- Existing project/setup and design prerequisites are already available. Do not add initialization, package, backend or standalone infrastructure phases. U0 contains its own fit/disclosure foundation; U5 contains approved documentation/final verification work.
- `[US1]` covers View/readability and its operator-approved contextual-status/selection amendment (U0/U1/U3/U4); `[US2]` covers client-to-Maps (U2); `[US3]` covers the illustrated guide (U5). These labels do not invent additional user stories.
- Paths resolve from this spec directory through the links below; proposed new filenames identify bounded implementation destinations, not existing APIs. UI/presentation stays in the viewer; do not put ImGui or window ownership into Core.
- All tasks start unchecked. `[P]` means disjoint writes with no dependency on another unfinished task. Unmarked tasks execute in listed order. Write the phase's regression tests before implementation; capture the failing baseline when its test seam is available, then the passing result. Use the existing test-project source-link pattern for pure viewer policies; do not add a viewer/GPU boot dependency to headless tests.
- Preserve existing setting, session, selection, diagnostic and popup owners. No second picker, selection store, telemetry pipeline, profiler, reader changes, renderer-fidelity work, help engine, release bump, new dependency or client-data bundling.
- No new members in the app/scene god-classes or new app partial files. New behavior belongs in owned services receiving explicit inputs; existing scene integration is narrow delegation/replacement only. Keep new files below the approximately 2,000-line budget and do not grow oversized files.
- Use existing [shared UI primitives](../../src/viewer/WoWViewer/UI/SharedUiWidgets.cs) and manager-owned windows/input. Changed controls must retain one authoritative home in the [control inventory](../../eng/ui/ui-controls.json).

## Evidence rules and stop gates

Each phase has **separate source/test and operator-proof tasks**. A source receipt may close only its source/policy assertions; it must mark runtime, input, display and screenshot criteria pending until the operator supplies real proof. No phase advances until both tasks pass. If proof is unavailable, stop at that gate rather than implement the next phase.

Every completed task needs a receipt or an attributable section of its phase receipt under [evidence](evidence/), containing changed paths, exact commands actually run and exit statuses, plus a criterion → evidence table. Source-only rows identify their limited assertion; rows naming runtime/data/visual behavior require real output. Operator rows also record revision/viewer version, configured client build/root/fingerprint when applicable, display resolution/DPI/effective and user scale, reproduction steps, results and relative capture links. Record unrun commands as unrun, never as passing. Keep captures inside the repository or an operator-supplied output root; publish no private client paths in the guide.

The primary agent owns integration and same-pass spec/task/continuity synchronization. This generated checklist does not authorize branch changes, commits or executing client/GPU/capture runs. Operators own those runs and screenshots; agents prepare instructions and integrate reviewed artifacts only.

## U0 — Non-scrolling footer and FPS disclosure (immediate defect; US1 amendment)

**Goal:** Both bottom regions fit their actual client rectangles, retain reachable essential actions/FPS and disclose existing diagnostic detail without scrolling or clipping.
**Independent check:** Policy tests exercise narrow/1080p/1440p/4K geometry and long payloads; the operator confirms the actual footer and FPS hover in empty/loaded scenes (FR-014/015/018; SC-008/010).

- [x] T001 [US1] Add footer-fit/disclosure regressions in [ViewerFooterLayoutTests.cs](../../tests/WowViewer.Core.Tests/ViewerFooterLayoutTests.cs) and extend [ViewerUiLayoutTests.cs](../../tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs): narrow regions independent of global display width; 1920×1080, 2560×1440 and 3840×2160; default/custom effective scales; long coordinates/area/status labels and large pending counters; essential action/FPS rectangles contained, detail retained and tooltip rectangles bounded. Wire pure policy files into [the test project](../../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj) as introduced by T002/T003, without launching the viewer.
- [x] T002 [US1] Implement a pure measured fit policy in [ViewerFooterLayout.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerFooterLayout.cs), taking actual action/status region geometry, effective scale and measured content widths; reserve reachable FPS/audio/essential actions and compact or disclose secondary content rather than relying on hidden overflow or display-global cursor positioning.
- [x] T003 [US1] Add an immutable existing-counter projection in [FooterDiagnosticsSnapshot.cs](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/FooterDiagnosticsSnapshot.cs), populated from [the status provider](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs); retain the currently displayed pending, CPU, tiles/chunks, model/load/failure, audio, area and coordinate context without adding collectors or hover-time asset/DBC/disk I/O.
- [x] T004 [US1] Apply measured fitting and prohibit both horizontal/vertical scrolling and wheel scrolling for both footer regions in [ViewerUiManager.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs), [ViewerActionBarsService.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs) and [the status provider](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs); preserve all existing action routes and full-name tooltips using in-region compact presentation, not off-screen controls or scrollbar-flag suppression alone.
- [x] T005 [US1] Implement bounded FPS-hover disclosure using [FooterDiagnosticsPresenter.cs](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/FooterDiagnosticsPresenter.cs), delegated from [the status provider](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs); keep FPS reachable even without a loaded scene, make formerly clipped fields readable, and use [UiPopupRegistry.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/UiPopupRegistry.cs) only if interactive long detail is needed. Respect [ViewerUiManager.Input.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Input.cs) capture/scene-input ownership; no selection or camera actions through interactive disclosure.
- [x] T006 [US1] Register the fitted footer and FPS disclosure routes in [ui-controls.json](../../eng/ui/ui-controls.json), updating the affected generated [UI-REFERENCE.md](../../docs/WoWViewer/UI-REFERENCE.md) through the existing [reference updater](../../eng/ui/Update-UiReference.ps1) as required; retain one authoritative diagnostic/control home and no permanent debug sidebar.
- [x] T007 [US1] **Source/test gate:** run commands B, U0 and R below and record changed paths, test output, command exit statuses and FR-014/015/018 → source/policy evidence in [u0-source-tests.md](evidence/u0-source-tests.md). Check both footer no-scroll flags, actual-region calculations and preservation of every former diagnostic field; explicitly leave SC-008/010 runtime/display/input proof open for T008. Do not advance on a failed source check.

> **Receipt (2026-10-22):** T001–T007 complete. `dotnet test` U0 filter: 28/28 passed, exit 0. `Verify-UiReference.ps1`: 0 errors, 0 warnings, exit 0. Full-solution build compiled with no C# errors but the copy step was blocked by the operator's running viewer (PID 31760); the test project built clean with `--no-dependencies`. Registry unchanged because the FPS disclosure is a hover affordance, not a new control. SC-008/010 remain open for T008. See [u0-source-tests.md](evidence/u0-source-tests.md).
- [ ] T008 [US1] **Operator-only runtime gate:** prepare and have the operator execute the U0 card from [quickstart.md](quickstart.md), recording real empty/loaded-scene results in [u0-operator-proof.md](evidence/u0-operator-proof.md): narrow plus 1080p/1440p/4K, standard/representative custom scales, no footer scrollbars, reachable FPS/audio/actions, readable pending/debug detail inside client bounds and no tooltip/popup input leakage (SC-008/010). Missing display/input/capture evidence stays unchecked; stop here before U1.

## U0b — Docked right panel and collapsed chat (operator amendment 2026-10-22)

**Goal:** Inspector/World/Model/PM4 share one docked right panel below the minimap and above the bottom bars, resizable and persisted; the chat starts collapsed.
**Independent check:** Layout tests cover dock geometry, hidden state and width clamps; the operator confirms the dock, splitter and collapsed chat in a run (FR-019–021; SC-011).

- [x] T049 [US1] Add `UiRegion.RightDock` geometry in [UiLayout.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/UiLayout.cs): below the minimap, above the bottom bar, left of the icon rail; clamp the requested width. Extend [ViewerUiLayoutTests.cs](../../tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs) with geometry, hidden-state and clamp regressions.
- [x] T050 [US1] Move Inspector/World/Model/PM4 into the dock in [ViewerUiManager.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs) and [ViewerActionBarsService.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerActionBarsService.cs); remove the four floating viewer windows and keep the right rail as the selector.
- [x] T051 [US1] Add the left-edge splitter and persist the dock width/open state in [ViewerUiManager.Persistence.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs); bump the layout document to v6.
- [x] T052 [US1] Default the chat to collapsed in [ViewerUiManager.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs) and [ViewerUiManager.Persistence.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs); record the source/test gate in [right-dock-and-chat-2026-10-22.md](evidence/right-dock-and-chat-2026-10-22.md). Leave SC-011 runtime/visual proof operator-owned.

> **Receipt (2026-10-22):** T049–T052 complete. Solution build 0 errors; 35/35 focused tests; registry 0 errors/0 warnings. SC-011 remains open for operator proof. See [right-dock-and-chat-2026-10-22.md](evidence/right-dock-and-chat-2026-10-22.md).

## U1 — One scale/reset owner and View routes (US1, P1)

**Goal:** Existing presentation settings are discoverable through View and one preference owner provides predictable scale-only, layout-only and combined recovery.
**Independent check:** Policy/persistence/reset tests pass; the operator changes scale and uses all resets with a map loaded, opens tools and restarts (FR-001–006/013; SC-001–003).

- [ ] T009 [US1] Add [ViewPreferencePolicyTests.cs](../../tests/WowViewer.Core.Tests/ViewPreferencePolicyTests.cs) for finite normalization to 0.85–2.20/default 1.0, non-finite/legacy persisted values, one-time persisted correction, shared View/Settings values and reset truth-table effects; extend [ViewerUiLayoutTests.cs](../../tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs) for effective display × user scaling. Link pure policy files in [the test project](../../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj) when introduced; scene-retention and actual restart proof remain T016.
- [ ] T010 [US1] Implement [ViewPreferencePolicy.cs](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewPreferencePolicy.cs) and owned [ViewPreferencesService.cs](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewPreferencesService.cs) over the existing settings value; normalize user scale without a competing store and retain display-aware composition in [ViewerUiManager.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs).
- [ ] T011 [US1] Route load/save and Settings changes through the single preference owner in [ViewerSettingsService.cs](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs) and [SettingsWindowService.cs](../../src/viewer/WoWViewer/Workbench/Services/Settings/SettingsWindowService.cs); clamp old persisted outliers once, update existing open UI immediately and persist the same normalized value without changing unrelated settings.
- [ ] T012 [US1] Expose current user text scale, immediate adjustment and the existing camera/presentation/interface visual groups through View in [MainMenuBarService.cs](../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs), at most one submenu deep; delegate to current setting/action owners rather than copy settings or change camera/renderer behavior.
- [ ] T013 [US1] Add clearly named scale-only and combined reset actions through [ViewPreferencesService.cs](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewPreferencesService.cs) and [MainMenuBarService.cs](../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs); preserve existing layout-only semantics in [ViewerUiManager.Persistence.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs). All three preserve the active client/map; scale reset restores user 1.0, not physical display scale.
- [ ] T014 [US1] Register View/scale/three reset routes and their authoritative owners in [ui-controls.json](../../eng/ui/ui-controls.json), refreshing affected [UI-REFERENCE.md](../../docs/WoWViewer/UI-REFERENCE.md) with [Update-UiReference.ps1](../../eng/ui/Update-UiReference.ps1); preserve accepted icon bars, full-name tooltips and unrelated supported settings routes.
- [ ] T015 [US1] **Source/test gate:** run B, U1 and R, recording normalization, persistence/reset assertions and menu-route coverage in [u1-source-tests.md](evidence/u1-source-tests.md), with exact exits and FR-001–006/013 mapping. Source tests do not close loaded-scene retention, restart or SC-003 display appearance; leave those rows for T016.
- [ ] T016 [US1] **Operator-only runtime gate:** execute the U1 [verification card](quickstart.md) and record [u1-operator-proof.md](evidence/u1-operator-proof.md): all existing visual groups and resets within View plus one submenu; exact reset effects; loaded client/map retained; menus/open/new tools consistent; restart persistence; no clipping/overlap at standard/custom scales on 1080p/1440p/4K and display-aware sizing retained (SC-001–003). Stop before U2 until real proof passes.

## U2 — Maps once per successful activation (US2, P1)

**Goal:** A committed client activation and its current discovery outcome open the existing Maps window once, including successful-empty results, without stale or repeated opening.
**Independent check:** A deterministic state-transition suite covers every listed case; the operator verifies representative actual client load routes (FR-007–009; SC-004/005).

- [ ] T017 [US2] Add [MapsActivationPolicyTests.cs](../../tests/WowViewer.Core.Tests/MapsActivationPolicyTests.cs) covering first/replacement activations, Maps already open, close then ordinary refresh, zero maps, failure, cancellation, discovery error and out-of-order stale completion; assert monotonic committed generations, one consumption per current activation and no previous-client list presented as a new result. Link the pure policy in [the test project](../../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj) as introduced.
- [ ] T018 [US2] Audit folder, saved-client, startup, archive/installation and CASC activation/discovery paths, recording concrete success/refresh/failure seams in [u2-activation-route-audit.md](evidence/u2-activation-route-audit.md) using [DataSourceSessionService.cs](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs) and [CascAhdrSourceService.cs](../../src/viewer/WoWViewer/Workbench/Services/CascAhdr/CascAhdrSourceService.cs) as entry points; identify only existing route owners, not new load routes or reader changes.
- [ ] T019 [US2] Add session-owned committed monotonic activation identity/current discovery notifications in [DataSourceSessionService.cs](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs), routing successful CASC activation from [CascAhdrSourceService.cs](../../src/viewer/WoWViewer/Workbench/Services/CascAhdr/CascAhdrSourceService.cs) and other owners identified by T018 through the same transition. Do not infer activation from previous/new map counts or increment on refresh/fail/cancel.
- [ ] T020 [US2] Implement pure once-consumption/stale-generation policy in [MapsActivationPolicy.cs](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/MapsActivationPolicy.cs) and owned [MapsActivationCoordinator.cs](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/MapsActivationCoordinator.cs), consuming only the current committed activation's discovery result; keep window close distinct from a new activation and avoid frame-loop reopen logic.
- [ ] T021 [US2] Integrate the coordinator with existing Maps opening in [DataSourceSessionService.cs](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs) and [ViewerUiManager.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs); reuse/bring forward the existing window, replace prior-client discovery contents, show an honest successful-empty state and prevent failure/cancel/stale completion from opening old data as new.
- [ ] T022 [US2] Update existing Maps activation/open-route descriptions in [ui-controls.json](../../eng/ui/ui-controls.json) and affected [UI-REFERENCE.md](../../docs/WoWViewer/UI-REFERENCE.md), using [Update-UiReference.ps1](../../eng/ui/Update-UiReference.ps1); retain explicit open/close and no permanent Maps sidebar/new tab.
- [ ] T023 [US2] **Source/test gate:** run B, U2 and R, attach T018's complete route audit and exact exit statuses to [u2-source-tests.md](evidence/u2-source-tests.md), mapping FR-007–009 to transition output. Document modeled versus actual source-route coverage separately; SC-004/005 actual client behavior remains T024.
- [ ] T024 [US2] **Operator-only runtime gate:** execute the U2 [card](quickstart.md) with representative real supported routes and record route/build/fingerprint plus first/replacement/open/closed/refresh/fail/cancel/empty/stale outcomes in [u2-operator-proof.md](evidence/u2-operator-proof.md). Verify exactly one current Maps window per success, respected close and no false-success/stale list (SC-004/005); unexercised cases remain gaps. Stop before U3 until the gate passes.

## U3 — Contextual selected identity and hover (US1 amendment)

**Goal:** Existing selection/hover feeds a readable viewport-centered identity header and bounded contextual disclosure, with honest fallback and no competing selection state or fake health.
**Independent check:** Presentation/lifecycle geometry tests pass; the operator checks named/unnamed entity and non-entity examples, scales and input ownership (FR-016/018; identity/input portions of SC-009/010).

- [ ] T025 [US1] Add [SelectionPresentationTests.cs](../../tests/WowViewer.Core.Tests/SelectionPresentationTests.cs) for named/unnamed identity, honest asset fallback, replace/deselect/client-replacement clearing, long names, viewport-offset centering, display/scale bounds and hover context; ensure no health/progress fabrication. Link pure presentation/geometry files in [the test project](../../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj) when introduced.
- [ ] T026 [US1] Add immutable [SelectionPresentation.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionPresentation.cs), projected from existing selection/hover identity in [SceneHoverAndPickService.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs); preserve actual display name or existing asset identifier and lifecycle, with no new picker, selection database, hover I/O or made-up entity values.
- [ ] T027 [US1] Implement an owned [SelectionContextPresenter.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionContextPresenter.cs) with pure geometry in [SelectionContextLayout.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionContextLayout.cs): larger bright/bold readable identity at the top-middle of the scene viewport, bounded long-name disclosure and chrome/capture-aware placement using existing UI tokens/inspection primitives.
- [ ] T028 [US1] Delegate selected-header and appropriate object-hover rendering from [SceneHoverAndPickService.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs) to the presenter; replace the affected projected nameplate route instead of leaving duplicate identity surfaces. Clear stale presentation on deselection/client replacement without changing picking, movement or scene ownership.
- [ ] T029 [US1] Integrate overlay hit/capture behavior with [ViewerUiManager.Input.cs](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Input.cs) and [SelectionContextPresenter.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionContextPresenter.cs); keep contextual content inside the current client/viewport and prevent scene selection/camera input through UI while preserving ordinary unobstructed scene interaction.
- [ ] T030 [US1] Register the contextual identity/hover surfaces and their single authoritative selection owner in [ui-controls.json](../../eng/ui/ui-controls.json), refreshing affected [UI-REFERENCE.md](../../docs/WoWViewer/UI-REFERENCE.md) through [Update-UiReference.ps1](../../eng/ui/Update-UiReference.ps1); do not introduce a permanent inspection sidebar.
- [ ] T031 [US1] **Source/test gate:** run B, U3 and R, recording presentation/lifecycle/geometry output, changed paths and exact exits in [u3-source-tests.md](evidence/u3-source-tests.md). Map FR-016/018 and identity/input portions of SC-009/010 without closing actual appearance/input or the U4 category-color requirement.
- [ ] T032 [US1] **Operator-only runtime gate:** follow the U3 [card](quickstart.md) for named/unnamed entities and non-entities, long names, deselect/replace/client change and standard/custom display cases; record top-middle readable identity, honest fallback, no stale/HP values and bounded hover/no input leakage in [u3-operator-proof.md](evidence/u3-operator-proof.md). Category agreement stays open until U4; stop before U4 until U3's identity/input assertions pass.

## U4 — One category policy for header, hover and enabled bounds (US1 amendment)

**Goal:** Deterministic category resolution and one palette agree across contextual identity and existing enabled bounds, without fabricated semantics or altered geometry/selection behavior.
**Independent check:** Known-tree/WMO/NPC/unknown policy and agreement tests pass; the operator compares real header/hover/bounds and verifies unchanged bounds controls (FR-017/018; category portions of SC-009).

- [ ] T033 [US1] Add [SelectionCategoryPolicyTests.cs](../../tests/WowViewer.Core.Tests/SelectionCategoryPolicyTests.cs) for explicit semantic metadata precedence, bounded evidenced known-tree references, deterministic fallback, WMO/NPC/unknown cases and no inference that every M2/WMO is a tree; assert equal category/color input for header, hover and bounds while emphasis remains independent. Link pure policy files in [the test project](../../tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj) as introduced.
- [ ] T034 [US1] Inventory only already-available metadata/reference evidence used by current selection/bounds inputs, documenting any bounded known-tree mapping and explicit fallback reasons in [u4-category-evidence.md](evidence/u4-category-evidence.md); use [SceneHoverAndPickService.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs) and existing [WorldScene.cs](../../src/viewer/WoWViewer/Terrain/WorldScene.cs) bounds submission as seams. Do not harvest, classify with ML, change readers or treat file-family/path guesses as proven semantics.
- [ ] T035 [US1] Implement pure [SelectionCategoryPolicy.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionCategoryPolicy.cs) containing the shared resolver/category palette using T034's existing evidence; distinguish known object kind from unproven semantic category and retain explicit Unknown/fallback evidence. Keep color, readable category label and selected/hovered emphasis separate.
- [ ] T036 [US1] Feed the same resolved category/style into [SelectionPresentation.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionPresentation.cs) and [SelectionContextPresenter.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionContextPresenter.cs), preserving readable bright identity and the U3 lifecycle/input behavior; known trees share the category color and unknowns remain honestly labeled.
- [ ] T037 [US1] Add owned [SelectionBoundsOverlayAdapter.cs](../../src/viewer/WoWViewer/Workbench/Services/Scene/SelectionBoundsOverlayAdapter.cs) to apply that same category/style to existing enabled bounds through a narrow replacement/delegation at [WorldScene.cs](../../src/viewer/WoWViewer/Terrain/WorldScene.cs) bounds submission. Add no scene members, selection state or extra geometry; preserve bounds toggles, culling, visibility and selected/hovered emphasis.
- [ ] T038 [US1] Update category/header/bounds descriptions in [ui-controls.json](../../eng/ui/ui-controls.json) and affected [UI-REFERENCE.md](../../docs/WoWViewer/UI-REFERENCE.md) via [Update-UiReference.ps1](../../eng/ui/Update-UiReference.ps1), making the same category policy and explicit Unknown behavior clear without inventing new controls.
- [ ] T039 [US1] **Source/test gate:** run B, U4, U3 and R, recording category agreement, fallback/evidence and preserved overlay submission policy in [u4-source-tests.md](evidence/u4-source-tests.md) with exact exits and FR-017/018 mapping. Attach T034's evidence and keep SC-009 real category/bounds appearance pending for T040; do not claim renderer-fidelity completion.
- [ ] T040 [US1] **Operator-only runtime gate:** execute the U3/U4 [card](quickstart.md) with named/unnamed known-tree, WMO, NPC and unknown examples; record matching header/hover/enabled-bounds colors, readable labels, preserved emphasis/toggles/culling and no extra geometry/selection changes in [u4-operator-proof.md](evidence/u4-operator-proof.md). Link U3 identity/input results to complete SC-009 coverage; missing category examples remain gaps. Stop before U5 until proof passes.

## U5 — Illustrated guide and final receipts (US3, P2)

**Goal:** The existing guide accurately explains the implemented shell/workflows using genuine reviewed running-build images, then passes a real walkthrough.
**Independent check:** Image/link/control checks pass and the operator follows client → Maps/spawn → tool/inspection → View recovery → workbench workflows against the documented build (FR-010–013 and amendment examples; SC-006/007).

- [ ] T041 [P] [US3] Update affected prose/navigation in [USERGUIDE.md](../../docs/WoWViewer/USERGUIDE.md) for accepted icon bars, independent tool close, View/three resets, automatic Maps, spawn choice, FPS disclosure and contextual selection/categories, inspection and workbench pages; retain unrelated supported-feature documentation and remove affected retired-sidebar/Spellbook guidance. Do not publish placeholder screenshots or describe uncaptured visuals as verified.
- [ ] T042 [P] [US3] Prepare bounded screenshot/caption cards in [u5-capture-cards.md](evidence/u5-capture-cards.md), using [quickstart.md](quickstart.md): shell/menu/icon bars/map-minimap/chat/tool behavior; View/scale/all resets; client-to-Maps/spawn; FPS diagnostics; contextual identity/category bounds; inspection; workbench navigation. Specify build/version, display/DPI/user/effective scale, client fingerprint where relevant, target control/expected state, review/privacy checks and repository-relative output destinations; operators execute all captures.
- [ ] T043 [US3] **Operator-only capture task:** after T041/T042, capture the implemented running build and record provenance/review state per card in [u5-capture-review.md](evidence/u5-capture-review.md), with genuine images under [evidence/screenshots](evidence/screenshots/). Review readable labels/captions, build match and privacy before publication; missing, rejected, old-build, diagram or mock images remain open and cannot satisfy FR-011/012.
- [ ] T044 [US3] Integrate only T043's reviewed genuine images into [USERGUIDE.md](../../docs/WoWViewer/USERGUIDE.md), using [docs/WoWViewer/images/spec-286](../../docs/WoWViewer/images/spec-286/) for published copies, relative image links, useful alt text/captions and explicit illustrated build/version; tie each image back to its capture/review record and documented workflow, excluding private local paths and placeholders.
- [ ] T045 [US3] Reconcile guide click targets with [ui-controls.json](../../eng/ui/ui-controls.json) and [UI-REFERENCE.md](../../docs/WoWViewer/UI-REFERENCE.md), refresh the reference with [Update-UiReference.ps1](../../eng/ui/Update-UiReference.ps1) if required and verify each guide image/document link resolves inside the project; record the exact link-check command/output and FR-011 workflow → reviewed-image mapping in [u5-link-checks.md](evidence/u5-link-checks.md).
- [ ] T046 [US3] **Source/document gate:** run B, ALL and R, preserving actual command output/exits in [u5-source-docs.md](evidence/u5-source-docs.md); include T045's link results and reviewed capture provenance, and map FR-010–013 to the affected guide/reference sections. Tests/valid links do not close the walkthrough or prove screenshots genuine without T043's review.
- [ ] T047 [US3] **Operator-only walkthrough gate:** follow the illustrated guide against its named build, completing client opening, automatic Maps/map/spawn choice, tool open/close, inspection, scale recovery and workbench page selection, plus FPS/selection examples; record matching image labels/click targets, results and SC-006/007 mapping in [u5-operator-walkthrough.md](evidence/u5-operator-walkthrough.md). Keep this unchecked if any required reviewed image or workflow proof is missing.
- [ ] T048 [US3] Assemble the final FR-001–018 / SC-001–010 criterion → real-output index in [implementation-receipt.md](evidence/implementation-receipt.md), linking each U0–U5 source/operator receipt and exact verification commands/exits; have the primary synchronize [tasks.md](tasks.md), [STATUS.md](../STATUS.md) and [activeContext.md](../../memory-bank/activeContext.md). Close only criteria with adequate receipts; report any remaining operator-owned gap explicitly and do not claim runtime/visual/FPS/audio or renderer work from compilation/tests.

## Verification command catalog (prepared, not executed by task generation)

Run from the project directory in PowerShell 7. Phase receipts must contain the full commands actually executed, not only these aliases. The source-linked pure policy tests remain headless; client/window/input/display/screenshot runs are operator-only.

**B — viewer build**

```powershell
dotnet build ./WowViewer.slnx -c Debug
```

**U0 — footer/layout**

```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~ViewerFooterLayoutTests|FullyQualifiedName~ViewerUiLayoutTests'
```

**U1 — scale/reset/persistence policy and layout**

```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~ViewPreferencePolicyTests|FullyQualifiedName~ViewerUiLayoutTests'
```

**U2 — activation/Maps**

```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~MapsActivationPolicyTests'
```

**U3 — identity/lifecycle/viewport**

```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~SelectionPresentationTests'
```

**U4 — category/style agreement**

```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~SelectionCategoryPolicyTests'
```

**R — authoritative UI reference**

```powershell
pwsh -NoProfile -File ./eng/ui/Verify-UiReference.ps1
```

**ALL — final cumulative focused policy/document regression**

```powershell
dotnet test ./tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter 'FullyQualifiedName~ViewerFooterLayoutTests|FullyQualifiedName~ViewerUiLayoutTests|FullyQualifiedName~ViewPreferencePolicyTests|FullyQualifiedName~MapsActivationPolicyTests|FullyQualifiedName~SelectionPresentationTests|FullyQualifiedName~SelectionCategoryPolicyTests|FullyQualifiedName~UiControlDocumentationTests'
```

## Criterion-to-phase coverage

| Approved criteria | Source/policy tasks | Required real proof |
|---|---|---|
| FR-014/015/018; SC-008/010 footer/disclosure | T001–T007 | T008: fitted footer, reachable FPS, full detail and input ownership across display/scale cases |
| FR-001–006/013; SC-001–003 | T009–T015 | T016: menu reachability, exact resets, retained scene, restart and display readability |
| FR-007–009; SC-004/005 | T017–T023 | T024: current client Maps transitions and negative/empty cases |
| FR-016/018; identity/input SC-009/010 | T025–T031 | T032: actual selected identity, lifecycle, hover/viewport/input |
| FR-017/018; category SC-009 | T033–T039 | T040, joined to T032: evidenced category examples and enabled bounds agreement |
| FR-010–013; SC-006/007; amendment illustrations | T041/T042/T044–T046 | T043 reviewed genuine captures and T047 real guide walkthrough |
| All criteria; no overclaiming | T048 | Links to the above actual outputs; unresolved rows remain open |

## Dependencies and incremental delivery

```text
Design/setup already available
  → U0 T001–T007 source → T008 operator STOP GATE
  → U1 T009–T015 source → T016 operator STOP GATE
  → U2 T017–T023 source → T024 operator STOP GATE
  → U3 T025–T031 source → T032 operator STOP GATE
  → U4 T033–T039 source → T040 operator STOP GATE
  → U5 (T041 || T042) → T043 → T044 → T045 → T046 → T047 → T048
```

- **Immediate bounded delivery/MVP:** U0 only. The next implementation pass is T001–T007, followed by operator-owned T008; no U1 implementation while U0 proof is pending.
- **US1:** U1 is independently testable after U0, while U3 identity/input and U4 category/bounds extend the dated operator amendment in the prescribed order. U3 does not pretend to close U4's category criteria.
- **US2:** U2 has independent activation/state acceptance cases; it follows U1 as an execution gate, not because scale settings should own client state.
- **US3:** U5 documents only behavior validated in previous phases. Capture/review precedes screenshot integration and real walkthrough precedes final criterion closure.
- Each phase adds a bounded accepted-shell increment. No code/demo/client run or screenshot is implied by task generation.

## Safe parallel examples and write ownership

| Story/phase | Allowed parallel work | Serial boundary |
|---|---|---|
| US1/U0 | Read-only review of T001 cases and T003 diagnostic-field coverage can run alongside the primary's focused implementation. | No parallel writes to status/manager/test-project seams; T007 then operator T008 are serial. |
| US1/U1 | Read-only reset truth-table review and menu-group inventory are independent review questions. | Preference, Settings/menu integration, inventory and receipt tasks remain ordered. |
| US2/U2 | Read-only review of negative/stale-transition cases can accompany the T018 route audit. | All activation writes have one owner; no overlapping session/CASC/manager patches. |
| US1/U3 | Read-only long-name/viewport test review and inspection-primitive review are independent. | Selection presentation/presenter/input writes remain ordered with one owner. |
| US1/U4 | Read-only category-evidence review and comparison of enabled bounds routes are independent. | Policy precedes both consumers; no concurrent edits to the scene seam. |
| US3/U5 | T041 guide prose and T042 capture cards are `[P]`: different files, both depend only on the completed U4 gate. | T043 needs both; T044–T048 remain ordered. Operators, never sub-agents, capture/run clients. |

Primary owns scope interpretation, source integration and gate decisions. Any bounded discovery/review delegation is read-only or has an explicitly disjoint write set; never use Terra. Do not parallelize across U0–U5 stop gates or delegate client/GPU/visual proof.
