# Feature Specification: UI Re-Audit — Sidebar Standardization & Deduplication (Spec 227)

**Feature Branch**: `227-ui-reaudit`
**Created**: 2026-09-06
**Status**: Draft — authored verbatim from operator directive; audit not started
**Input**: Operator directive 2026-09-06 (condensed verbatim): "Only the Viewer and Quick/Inspector
tabs seem sane, the Editor tabs are insane and nonsensical, we have multiple weak terrain signal
amplifiers for no reason, the Archeology option doesn't match the rest of the UI styling. We need to
consolidate on a single fucking sidebar design format, and continue consolidation efforts, we still
duplicate data in like 3-5 places in the inspector… multiple minimaps but only the tiny minimap
allows teleporting reliably… We need to get things into either a drop-down based sub-category system
of menus in the sidebars, or better deduplicate and organize the newest of the UI features to the
forefront, or just re-engineer the UI from the ground up… The app is not very approachable because
of the UI… We need a proper full audit of what is there, from top to bottom."

## Context

Spec 223 consolidated navigation into four tabs, but the operator's acceptance reviews keep failing
on the same class of problem: **the consolidation moved content without standardizing it**. The
result is workbenches with different visual languages, duplicate analysis surfaces (weak-signal
amplifiers, minimaps), and Inspector data repeated 3–5 times. This spec re-opens the audit with a
harder mandate: one sidebar design system, zero duplicate surfaces, and approachability as a goal.

## User Stories

### US1 — Full top-to-bottom surface re-audit (P0)
The Spec 223 surface inventory (`surface-inventory.md`) is re-run against the current build: every
panel, tab, dropdown, toolbar control, and floating window is re-listed with its owner, its
duplicates, and its disposition. The audit records screenshots. Anything not in the inventory is a
defect.

**Acceptance criteria**:
- A dated inventory v2 exists under this spec's directory, organized by workspace profile.
- Every Spec 223 "retire/merge" disposition is verified as actually done; undone ones are reopened.
- The inventory names every duplicate (weak-signal amplifiers, minimaps, Inspector repetition).

### US2 — One sidebar design system (P0)
All workspace tabs use one sidebar format: consistent section headers, help affordances, control
sizes, and a **dropdown/collapsible sub-category navigation** so tools are findable without
scrolling a wall of panels.

**Acceptance criteria**:
- Every sidebar section uses `SharedUiWidgets` primitives — no bespoke section styling (the
  Archaeology styling mismatch is fixed under this rule).
- Sub-category navigation is dropdown/collapsible; every tool is reachable in ≤ 3 interactions.
- Editor and Archaeology workbenches are reorganized until they are no longer "insane": each
  sub-page has one purpose, one header, one action area.

### US3 — Deduplication with teeth (P0)
For every data surface shown in multiple places, exactly one authoritative surface remains and the
others link to it (Spec 223 FR-3, now enforced by the audit).

**Named duplicates to resolve (operator-named)**:
- Weak terrain signal amplifiers: multiple amplifier panels exist "for no reason" — consolidate to
  the single owning surface (Spec 194 owner) and retire the rest.
- Minimaps: multiple minimap surfaces; only the tiny minimap teleports reliably — either fix
  teleport in the others or retire them in favor of one canonical minimap with teleport.
- Inspector repetition: the same selected-object data appears 3–5 times in Inspector pages — one
  authoritative page per object type, others deep-link.

**Acceptance criteria**:
- SC-1 of Spec 223 (walk the viewer against the inventory, find no unlisted panel) passes, and in
  addition no information surface shows the same data as another (SC-2 enforced by screenshot
  comparison per profile).

### US4 — Approachability (P2)
A new operator can find Load, fog, wireframe, capture, and inspect without guidance.

**Acceptance criteria**:
- The newest/most-used features are at the forefront of each profile's sidebar (operator option 2).
- The guide's UI chapter is rewritten from the post-audit inventory, not from memory.

## Constraints

- Spec 223's binding decisions still hold (four tabs, legacy shell kept until Spec 212 HUD works).
- No consolidation change without the inventory v2 row naming the replacement (FR-2 of Spec 223).
- This audit is the first consumer of the `speckit-cleanup` procedure's audit discipline.

## Dependencies

- Spec 223 (owns the four-tab structure and inventory v1).
- Spec 194 (weak-signal owner), 144/147 (camera capture/minimap owners) for dedupe targets.
- Spec 212 (HUD) is adjacent but not blocking.

## Stakeholders

- Operator: accepts the inventory v2 and each consolidation phase; the UI is currently called "a
  giant hinderance to productivity" — approachability is the acceptance bar.
