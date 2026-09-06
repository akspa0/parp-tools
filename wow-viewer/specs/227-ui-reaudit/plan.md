# Implementation Plan: UI Re-Audit — Sidebar Standardization & Deduplication

**Branch**: `223-ui-consolidation-audit` | **Date**: 2026-09-06 | **Spec**: [Spec 227](spec.md)

## Summary

Re-audit every present ImGui surface before changing navigation or data homes. The audit produces
an inventory v2 with source provenance, an explicit duplicate-to-authoritative-home mapping, and
operator screenshots. Only then, one independently validated surface at a time, replace bespoke
sidebar presentation with existing `SharedUiWidgets` while preserving each authoritative action
and route. This is a consolidation and approachability pass, not a renderer, format-reader, or
HUD rewrite.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: ImGui.NET; existing `WoWViewer.UI.SharedUiWidgets`; existing
`WorkbenchNavigator` and `OpenWorkbenchTab` routing

**Storage**: Markdown inventory and receipts under `specs/227-ui-reaudit/`; existing viewer
settings only when an already-persisted UI state must be preserved

**Testing**: focused C# tests where a route or pure presentation helper is changed; viewer Debug
build for each source slice; operator walkthrough and screenshots for visual, input, and
three-interaction claims

**Target Platform**: existing Windows desktop viewer

**Project Type**: desktop application UI consolidation

**Performance Goals**: no new render loop, allocation-sensitive, or terrain-loading work; preserve
the current UI frame behavior

**Constraints**: do not add members to `ViewerApp` or `WorldScene`; do not alter format readers;
do not change an action's authority while restyling; no consolidation source edit before inventory
v2 records the removed/replacement surface; preserve the legacy shell pending Spec 212.

**Scale/Scope**: Viewer, Editor, Archaeology, and legacy shell surfaces; the operator-named
weak-signal, minimap, and Inspector duplicate families.

## Constitution Check

| Gate | Result | Evidence / handling |
|---|---|---|
| Repo independence | PASS | All source and evidence remain below `wow-viewer/`. |
| Library-first / format ownership | PASS | No format or shared-data behavior is introduced. |
| Real-data validation | PASS with operator gate | Source/build evidence cannot prove UI arrangement, input, or teleport. Inventory screenshots and walkthrough remain operator-owned. |
| No client-root assumptions | PASS | This pass introduces none. |
| One validated phase at a time | PASS | Each phase has a source/build receipt and stops before the next phase. |
| Spec 224 receipts and scope freeze | PASS | Each completed task links an evidence receipt; replacement rows precede source changes. |
| Spec 228 god-class freeze | PASS | UI work uses existing helpers/routes; any required new state or service extraction is deferred to Spec 228 after the relevant audit. |

The re-check after the inventory phase must confirm that every planned source edit has an inventory
v2 row and no task claims unobserved runtime behavior.

## Project Structure

```text
specs/227-ui-reaudit/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── surface-inventory-v2.md
├── evidence/
│   ├── t001-source-audit.md
│   └── screenshots/                 # operator captures; absent until captured
└── tasks.md

src/viewer/WoWViewer/
├── UI/SharedUiWidgets.cs             # presentation primitives only
├── Workbench/WorkbenchNavigator.cs   # existing labels and destination routing
├── ViewerApp_Sidebars.cs              # existing sidebar hosts; no new members
├── ViewerApp_MinimapAndStatus.cs      # existing minimap host and interaction
├── MinimapHelpers.cs                  # shared minimap rendering
└── ViewerApp_InspectorPayloads.cs     # existing inspector payload source
```

**Structure Decision**: UI behavior remains in its current owned surfaces until the inventory
selects a canonical home. No new service, data store, or API contract is needed for the inventory
or presentation-only standardization.

## Complexity Tracking

No constitution violation is introduced. The deliberately narrow first phase prevents an
unapproved shell redesign or source-decomposition work from riding along with the audit.

## Phase Roadmap

### Phase 0 — Inventory v2 and evidence baseline

1. Re-run the source audit by workspace and record every visible host, direct route, duplicate
   family, current owner, disposition, and replacement destination.
2. Create empty, named screenshot slots for the same inventory IDs; an operator captures each
   current-build surface with build and client identity.
3. Re-open every unfulfilled Spec 223 retire/merge disposition in v2 rather than treating a
   comment or old task checkbox as completion.
4. Re-check this plan's gates and make the first source slice only from rows with a named
   replacement. Stop if visual inventory evidence materially contradicts the source audit.

### Phase 1 — Sidebar standardization, one surface at a time

1. Choose the first inventory-approved Archaeology or Editor surface.
2. Replace only its bespoke headers, help, status, and action presentation with existing
   `SharedUiWidgets` calls; retain the existing actions and routes.
3. Build the viewer and run focused route/helper tests when a testable route changes.
4. Obtain an operator screenshot showing that surface remains reachable in at most three
   interactions. Do not move to a second surface until that receipt exists.

### Phase 2 — Duplicate families, independently

1. Weak-signal amplifiers: retain the Spec 194 owner named by v2; convert any other surface to a
   route/link or retire it only after the owner remains usable.
2. Minimaps: use the v2 screenshot/input matrix to select the one teleport-authoritative surface;
   fix an alternate only if its route is retained, otherwise link/retire it.
3. Inspector: map each object type to one authoritative page; convert repeated readouts to a
   direct route or concise pointer, preserving the payload authority.
4. Build and operator-walk each family separately. Never use unit tests as a minimap-teleport or
   visual-layout receipt.

### Phase 3 — Approachability and guide

1. Promote Load, fog, wireframe, capture, and inspect according to the accepted v2 order.
2. Rewrite the guide UI chapter from the final accepted inventory, then record the full
   profile-by-profile screenshot comparison.
