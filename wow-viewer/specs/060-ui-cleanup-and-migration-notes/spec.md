# Feature Specification: Viewer UI Cleanup + ImGui Migration Notes

**Feature Branch**: `060-ui-cleanup-and-migration-notes`
**Created**: 2026-06-12
**Status**: Draft
**Owner**: wow-viewer (viewer shell)
**Related**: spec 049 (Viewer UI Consolidation), spec 044 (Viewer Shell Usability)

**Input**: User described the current viewer UI as "rough and ugly edges" with specific pain points: duplicated Runtime Stats in 5 places, status bar collision (stats + PM4 overlap), mislabeled "Copy Scene" / "Log Scene" buttons in the status bar, capture/recording capturing the UI in frame, and two near-identical inspector/SceneInspector tab bars. The user also flagged ImGui as a long-term pain point: it requires constant manual layout tweaking, scaling is fragile, docking is rough, and the cost of "one simple edit" is days of refactor hell.

## Context

The viewer has 12 shell panels, 15 floating windows, a bottom status bar, and a fullscreen hover overlay. Most of this grew organically from spec 044 (dockable shell), spec 049 (consolidation), and the PM4 scene work. The result is functionally rich but visually rough: the same content is rendered 2-5 times, status bar elements overlap with panel content, button names don't match what they do, and capture windows include the chrome that surrounds them.

The user wants two things:
1. **A concrete UI cleanup** that addresses the specific pain points named above. The ImGui refactor hell stops being a blocker for small fixes.
2. **A captured note** that the ImGui-based UI is a candidate for a future migration to a multiplatform-native GUI library (Avalonia, MAUI, or WinUI 3). The note is a deliberate deferral, not an active project.

## User Scenarios

### US1 — Runtime Stats appears once, in one place (Priority: P1)

**Given** the user has the viewer open with a world loaded,
**When** they want to see FPS, render stats, or world stats,
**Then** the Runtime Stats content appears in a single, dedicated "Runtime Stats" panel that they can place in any dock,
**And** Runtime Stats is NOT duplicated in the Navigator, the Inspector tab bar, the SceneInspector tab bar, the Terrain Controls trailing block, or anywhere else.

**Acceptance**: grep for `DrawRuntimeStatsPanelContent` returns exactly one call site (or several that are explicit, documented tab instances — no silent duplication).

### US2 — Status bar is clean, not a button dump (Priority: P1)

**Given** the bottom status bar is rendered,
**When** the user looks at it,
**Then** it shows only: status message, FPS (color-coded), and possibly coordinates when terrain is loaded,
**And** it does NOT show the "Copy Scene" / "Log Scene" buttons (move to the Capture Automation panel or remove entirely if redundant).

**Acceptance**: status bar has at most 3 columns. No buttons. Copy/Log Scene either get a clear new home or get deleted.

### US3 — Capture/recording hides the UI by default (Priority: P1)

**Given** the user starts a video recording or a single-shot capture sequence,
**When** each shot is captured,
**Then** `_hideUiChrome = true` is set automatically during the capture frame,
**And** the captured PNG/frame shows only the 3D scene, not the panels/buttons/docks around it.

**Acceptance**: A `with_ui` flag in the capture request still exists for users who want UI captured, but `no_ui` is the default and is also applied to all single-shot captures and to video recording.

### US4 — SceneInspector and Inspector tab bars deduped (Priority: P2)

**Given** the user opens the SceneInspector panel,
**When** they look at its tabs,
**Then** the tabs are not a duplicate of the Inspector tab bar with slightly different content,
**And** either the two are merged (SceneInspector becomes the canonical panel), or one is removed, or the content is genuinely different and the overlap is intentional.

**Acceptance**: One of: (a) SceneInspector removed, (b) Inspector removed, (c) the two have non-overlapping content. The current state of "two near-identical tab bars" does not remain.

### US5 — Migration note exists for Avalonia/MAUI (Priority: P3, deferred)

**Given** a future contributor opens the docs,
**When** they ask "why is this all ImGui when the user hates it",
**Then** a doc explains the historical context, the cleanup done in this spec, the options considered (Avalonia, MAUI, WinUI 3), and the rough cost/effort to migrate.

**Acceptance**: A doc exists in `wow-viewer/docs/architecture/`. It's an exploration note, not a roadmap. No commitment.

## Requirements

### Cleanup (concrete work, ships)

- **FR-001**: `DrawRuntimeStatsPanelContent()` has exactly one caller, or a documented small set of callers in the tab system.
- **FR-002**: Navigator sidebar does NOT include Runtime Stats (move to a dedicated panel).
- **FR-003**: Inspector tab bar ("Selection / World / Model / Stats / Settings / PM4") does NOT include a "Stats" tab.
- **FR-004**: SceneInspector tab bar does NOT include a "Stats" tab (or SceneInspector is removed entirely).
- **FR-005**: Terrain Controls panel does NOT trail with a Runtime Stats block.
- **FR-006**: Status bar has at most 3 columns (status, FPS, optional coords).
- **FR-007**: "Copy Scene" button is removed from the status bar. It already exists in the Capture Automation window — that remains the canonical home.
- **FR-008**: "Log Scene" button is removed from the status bar entirely. The current implementation is a misnomer (it sets `_statusMessage`, doesn't log anywhere). If a real log is wanted, it lives in the Capture Automation window.
- **FR-009**: Single-shot capture sets `_hideUiChrome = true` for the duration of the capture frame unless `includeUi: true` is explicitly requested.
- **FR-010**: Video recording sets `_hideUiChrome = true` for the duration of every captured frame.
- **FR-011**: `with_ui` flag in the capture request still works (user can opt in to UI-in-frame).
- **FR-012**: SceneInspector panel is removed if it duplicates Inspector's tabs; otherwise their content is deduplicated.
- **FR-013**: The `FixedBottomDrawerTab` enum, the `_activeBottomDrawerTab` field, and the `DrawRightSidebarSection()` helper stay as-is unless the cleanup makes them provably dead code.

### Migration note (deferred, no code)

- **FR-014**: A doc at `wow-viewer/docs/architecture/ui-migration-options.md` exists with:
  - Why ImGui was chosen originally (prototyping speed, game-engine fit)
  - What hurt (manual layout, scaling, docking fragility, duplication traps)
  - The three options: Avalonia, MAUI, WinUI 3 (rough comparison, no commitment)
  - Rough cost estimate: 2-3 months to a working Avalonia port with the same surface area
  - Explicit "no commitment" disclaimer

## Out of Scope

- **Avalonia/MAUI migration code** — this spec only captures the note. Migration is its own multi-month project.
- **New UI features** — only cleanup, no new panels.
- **Visual design pass** — the goal is "not actively hostile to users", not "pretty".
- **PM4 status text consolidation** in the right sidebar — out of scope for this pass; covered by future PM4 work.
- **Settings refactor** — the `ShellPanelLayouts` save/load path works; not changing it.

## Follow-Up Tasks (small, independently-shippable slices)

1. **Runtime Stats caller audit** (FR-001 to FR-005): grep + tab-bar cleanup. One PR.
2. **Status bar button removal** (FR-006 to FR-008): surgical edit to `DrawStatusBar()`. One PR.
3. **Capture UI-hide default** (FR-009 to FR-011): change `includeUi: true` to `false` in default single-shot and video capture paths. One PR.
4. **SceneInspector dedup** (FR-012): either delete the panel or restructure. One PR.
5. **Migration note doc** (FR-014): doc-only, no code. One commit.
6. **Smoke + final test pass**: confirm viewer still launches, all panels render, capture works.

## Notes on the ImGui Decision

The user explicitly raised ImGui as "one major mis-step" — the cost of every small UI change is days of refactor hell. This spec addresses the SPECIFIC issues (duplication, status bar, capture) because they hurt regardless of UI library. The ImGui vs. Avalonia/MAUI question is a separate, much larger decision and is captured in the migration note (FR-014) for a future contributor to pick up.

If/when a migration happens, the work in this spec makes it easier: the Runtime Stats panel becomes a real component that any UI library can host, the capture UI-hide logic becomes a single toggle any UI library can honor, and the status bar cleanup is a content concern, not a layout concern.
