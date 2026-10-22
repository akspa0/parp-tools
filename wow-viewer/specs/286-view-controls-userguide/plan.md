# Implementation Plan: View Controls and Contextual Disclosure

**Branch**: 286-view-controls-userguide | **Date**: 2026-10-22 | **Spec**: [requirements](spec.md)
**State**: Planning only; no implementation, client, visual or screenshot completion claimed.

## Summary

Keep the accepted icon-bar shell. Expose View/scale/reset actions, open Maps once per successful
client activation, remove bottom-bar scrolling/debug overflow, reuse FPS-hover diagnostics, and
add readable type-colored contextual selection. Finish with a genuine screenshot guide.
Rendering work stays under [Epic 249](../249-epic-renderer-performance-and-correctness/plan.md)
and [Spec 259](../259-ground-effects-and-wmo-detail-doodads/plan.md), not hidden UI scope.

## Operator Intent Check

| Request | Planned response | Excluded implication |
|---|---|---|
| View, scale resets, auto Maps, screenshot guide | U1/U2/U5 | New help engine, release bump or camera rewrite |
| Bottom bar never scrolls; debug data on FPS hover | U0 | Permanent profiler/sidebar or lost off-screen controls |
| Bright/bold larger top-middle selected name | U3 | Fake HP or a second selection store |
| Category colors including trees, matching bounds | U4 | Guessing semantic truth from file format alone |
| Lay everything out using Spec Kit | Named-owner plans and gates | Implementation, commits, client/GPU jobs |

## Technical Context

- **Language/platform**: existing C#/.NET 10 desktop viewer, ImGui.NET, Silk.NET OpenGL.
- **Storage**: existing settings/layout only; transient presentation/selection snapshots are not
  persisted. No new database, network endpoint or dependency.
- **Testing**: xUnit, headless ImGui/layout and state-policy tests, UI-reference verifier;
  operator-owned actual input/DPI/client/screenshot checks.
- **Performance**: reuse existing frame and selection snapshots; no per-hover asset/DBC/disk I/O.
  No invented numerical FPS target or performance claim without matched captures.
- **Constraints**: no new app/scene god-class members; cohesive owned services, shared pure policy
  where appropriate; UI outside Core; no reader/writer or terrain-loading changes.
- **Scope**: FR-001–018 / SC-001–010. Renderer lanes are independent of UI implementation.

## Constitution Check — before research / after design

| Gate | Result |
|---|---|
| Operator-originated scope | PASS: original requirements plus dated operator amendment |
| One authoritative owner/library-first | PASS: shared preferences/selection/counters, presentation adapters only |
| Reader/reference/client protection | PASS: no client-container, format or legacy reference writes |
| God-class freeze/file budget | PASS: owned services; narrow delegation; split before growing oversized files |
| Real proof/write containment | PASS in design: per-phase receipts under owning spec, visual gates open |
| Accepted shell/shared UI/inventory | PASS: UI manager owns windows/input, register changed controls |

No exception is requested. These are design checks, not implementation signoff.

## Project Structure and Phase 0 outputs

[Research](research.md), [data model](data-model.md), [local contracts](contracts/contextual-view.md)
and [verification quickstart](quickstart.md) contain the planning outputs. No REST/GraphQL API
is warranted. Unknown semantic categories resolve explicitly to Unknown; client-specific proof
is an execution gate, not an unresolved design answer disguised as fact.

| Existing seam | Bounded implementation responsibility |
|---|---|
| [UI manager](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.cs:60), [persistence](../../src/viewer/WoWViewer/Workbench/Services/UiManager/ViewerUiManager.Persistence.cs:86) | Actual region geometry, effective display scale, no-scroll flags; preserve layout-only reset |
| [Main menu](../../src/viewer/WoWViewer/Workbench/Services/MainMenu/MainMenuBarService.cs:170), [Settings UI](../../src/viewer/WoWViewer/Workbench/Services/Settings/SettingsWindowService.cs:157), [store](../../src/viewer/WoWViewer/Workbench/Services/Settings/ViewerSettingsService.cs:229) | One new owned view-preference service; all routes delegate to it |
| [Status provider](../../src/viewer/WoWViewer/Workbench/Services/MinimapStatus/MinimapAndStatusService.MinimapAndStatus.cs:499) | Compact fit policy, FPS anchor, existing diagnostic snapshot disclosure |
| [Session discovery](../../src/viewer/WoWViewer/Workbench/Services/DataSourceSession/DataSourceSessionService.cs:232), [CASC path](../../src/viewer/WoWViewer/Workbench/Services/CascAhdr/CascAhdrSourceService.cs:325) | Committed activation/discovery notifications and once-per-activation Maps coordinator |
| [Scene hover/nameplate](../../src/viewer/WoWViewer/Workbench/Services/Scene/SceneHoverAndPickService.cs:520) | Owned header/tooltip presenter consuming existing selection |
| [Bounds submission](../../src/viewer/WoWViewer/Terrain/WorldScene.cs:3929) | Shared category result through an owned bounds-overlay adapter; no new scene state |
| [Layout tests](../../tests/WowViewer.Core.Tests/ViewerUiLayoutTests.cs:6), [inventory](../../eng/ui/ui-controls.json), [guide](../../docs/WoWViewer/USERGUIDE.md) | Focused tests, authoritative control routes, accurate illustrations |

## Phase 1 — bounded implementation roadmap (future work)

Each numbered pass has one concern. Stop at each gate before dependent work. Do not present
renderer controls/fixes as delivered merely because this UI work lands.

### U0 — non-scrolling footer and FPS disclosure (immediate defect)
1. Pure footer-fit policy uses actual region width/effective scale; preserve FPS/essential actions
   and move detailed fields into an immutable view of existing diagnostics.
2. Prohibit scrolling on both footer regions; fit content, not just hide scrollbar flags. FPS hover
   shows bounded contextual detail; optional interactive long detail uses the existing popup owner.
3. **Gate:** long labels/counters and narrow/1080p/1440p/4K layout tests; operator confirms reachable
   FPS, pending counts and no scrollbars/input leakage (SC-008/010).

### U1 — one scale/reset owner and View routes
1. Owned preference service normalizes finite user scale to the existing UI range 0.85–2.20,
   default 1.0; clamp old persisted outliers once. Manager retains display-aware composition.
2. Route Settings/View through it. Expose existing visual groups within one submenu; preserve
   layout-only reset, add scale-only and explicitly named combined reset.
3. **Gate:** normalization/reset/persistence tests and UI-reference checks; operator verifies
   consistent text, restart, all three meanings and retained loaded scene (SC-001–003).

### U2 — Maps once per successful activation
1. Session-owned monotonic activation identity replaces previous-map-count inference. Audit folder,
   saved-client, startup, archive and CASC success routes into the same committed transition.
2. Consume current discovery completion once to open existing Maps, including empty results.
   Refresh does not create an activation; failure/cancel/stale completion does not open old data.
3. **Gate:** replacement/refresh-after-close/empty/fail/cancel/stale tests; operator verifies
   representative actual client routes (SC-004/005).

### U3 — contextual selected identity and hover
1. Immutable presentation consumes existing selection/hover identity, honest asset-name fallback,
   clears on deselection/client replacement, and never owns picking or invents HP.
2. Owned presenter draws a readable larger bright/bold top-middle identity bar in the scene
   viewport; reuse inspection primitives and respect chrome/capture/input rules.
3. **Gate:** viewport/long-name/fallback/lifecycle tests; operator checks selected non-entity and
   entity examples, scales and hover/input ownership (SC-009/010).

### U4 — one category palette for header and bounds
1. Pure resolver/palette prefers existing semantic metadata; known-tree reference mappings are
   bounded and documented. Unknown remains Unknown; no new harvest, ML classifier or reader.
2. Feed the same category to header/hover and existing bounds through an owned adapter. Preserve
   selected/hovered emphasis, bounds toggles, visibility and culling.
3. **Gate:** tree/WMO/NPC/unknown agreement tests; operator confirms matching category colors,
   readable labels and no extra geometry/selection behavior changes (SC-009).

### U5 — illustrated guide and final receipts
1. Update guide prose and control inventory, then prepare capture cards for shell, View/reset,
   Maps/spawn, FPS disclosure, selection/categories, inspection and workbench.
2. Operator captures the running build; integrate only reviewed genuine screenshots with version,
   scale/workflow provenance and captions. Missing images stay open, not published placeholders.
3. **Gate:** image/link/control checks plus operator walkthrough (SC-006/007 and new examples).
   Compilation does not close visual, input, screenshot or runtime criteria.

## Phase 2 — task-generation handoff

Next Spec Kit step generates unchecked ordered tasks U0 → U1 → U2 → U3 → U4 → U5; maximum ten
per phase, exact paths/tests, per-phase receipts and criterion mapping. Renderer tasks remain
with their own plans, not duplicated here. Implement only one validated phase at a time.

## Verification / complexity / agent context

See [quickstart](quickstart.md). No new project/package, alternate picker, profiler, selection
store or backend is needed. Guarded PowerShell Spec Kit setup selected this branch/spec.
This checkout has no agent-context updater; continuity is updated explicitly instead. No new
technology warrants rewriting global agent instructions. No code or runtime jobs run in planning.
