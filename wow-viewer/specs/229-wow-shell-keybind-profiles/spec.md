# Feature Specification: WoW-Style Shell & Contextual Keybind Profiles (Spec 229)

**Feature Branch**: `229-wow-shell-keybind-profiles`
**Created**: 2026-09-06
**Status**: Draft — authored verbatim from operator directive; not planned
**Input**: Operator directive 2026-09-06 (verbatim intent): "I asked over and over again to implement
something that looked like the actual WoW interface, which would de-complicate the UI considerably,
but nothing has come of that request or plan. If we can bind things to keybinds and profiles that
enable/disable different sets of keybinds, then we have a means of simplifying all the functionality.
Have a look at how noggit and noggit-red handle contextual keybindings in their editing surfaces.
Those two projects are a huge inspiration for this project."

## Context

A recurring unimplemented operator request: the shell should **look and behave like the actual WoW
interface** — action-bar-like tool rows, WoW-style panel chrome — which inherently de-complicates the
UI. The complementary simplification mechanism is **contextual keybind profiles**: the active
workspace (or editing surface) determines which keybind set is live, exactly as Noggit and
Noggit-Red handle contextual bindings in their editing surfaces. Together these replace
"everything visible at once" with "the right tools for the active context, on keys".

## User Stories

### US1 — WoW-style interface shell (P1)
The viewer shell adopts the WoW interface language: action-bar-style tool rows for the active
context, WoW-flavored panel headers/buttons (using the existing `SharedUiWidgets` base), and a
minimal permanent chrome. The full sidebar machinery remains available but recedes behind the
contextual rows.

**Acceptance criteria**:
- Active-context tool rows (action bars) expose the current profile's top actions; slot tooltips
  show the bound key.
- The shell visually reads as WoW-like (frames, header ribbons, slot buttons) while staying
  `SharedUiWidgets`-based (Spec 227 §11 standard).
- The existing four-tab/profile structure (Spec 223) is preserved; the WoW shell is the skin and
  contextual layer over it, not a parallel navigation system.

### US2 — Contextual keybind profiles (P1)
Keybind sets are grouped into **profiles** (e.g., Navigation, Terrain Reconstruction, Placement,
Capture), and the active workspace/editing surface determines the live set. Noggit/Noggit-Red are
the reference for feel: entering a mode swaps the binding context and the visible action row
together.

**Acceptance criteria**:
- A keybind registry maps actions to keys per profile; conflicts are reported, not silently won.
- Switching workspace profile or entering an editing surface switches the live keybind set and
  shows the active set (HUD hint or action-bar highlight).
- Every sidebar/menu action is addressable by the registry (no dead actions, no second action
  implementation — FR-6 of Spec 223).
- Bindings persist with viewer settings and can be exported/imported like camera paths.

### US3 — On-screen key reference (P2)
The operator can see the live profile's bindings at a glance (help overlay, `?` affordances, or an
action-bar mode tooltip), so keys are discoverable without the guide.

**Acceptance criteria**:
- One overlay lists the active profile's bindings grouped by function.
- Pressing an unbound-but-bindable action suggests where to bind it.

## Constraints

- Sequences after the Spec 227 audit: the audit decides WHAT actions exist per context; this spec
  decides HOW they are presented and bound.
- All action invocation goes through canonical services (Spec 228's extraction provides them).
- Existing camera keybinds (WASD, Tab, I, M, P…) migrate into the Navigation profile unchanged.

## Dependencies

- Spec 227 (audit defines the action inventory per context).
- Spec 228 (services own the actions; the registry binds to services, not god-class methods).
- Noggit / Noggit-Red as design references (inspiration credited in the About box).

## Success Criteria

- **SC-1**: A session in Terrain Reconstruction exposes only that context's tools on-screen and on
  keys; switching to Capture swaps both atomically.
- **SC-2**: The operator confirms the shell "de-complicates the UI" in a real walkthrough — fewer
  visible panels at any moment than the pre-227 baseline screenshots.
