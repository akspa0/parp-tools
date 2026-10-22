# Feature Specification: Discoverable View Controls and Screenshot User Guide

**Feature Branch**: `286-view-controls-userguide`

**Created**: 2026-10-22

**Status**: Specified; ready for implementation planning. No implementation or new screenshots claimed.

**Input**: Operator: "this is very nice looking and pretty functional"; "the reset shell layout option
does not also reset the font scale, but that should also be an option in the menu somewhere, maybe a
'View' menu, where we put all options for view options in an easy to get-to place"; "The maps tab
should automatically open when you load a client. We need a proper screenshots-based userguide, too.
use speckit."

## Context and scope

The operator accepts the scene-first icon-bar direction delivered by
[Spec 283](../283-v070-ui-presets-and-explore/spec.md). This is a usability follow-up, not another
shell redesign. Preserve anchored pictogram bars, full-name tooltips, independently closeable tool
windows, and the removal of the rejected Spellbook and two-character control labels.

The positive feedback accepts the overall direction; it does not prove every setting, client source,
scale, or screen size. Scope is discoverable existing view controls, scale recovery, automatic Maps
opening, and an accurate illustrated user guide.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Adjust the view and recover readable scaling (Priority: P1)

A user opens View and finds controls for the current presentation, including text size, interface
visibility and existing visual settings. If scaling looks wrong, the user can reset it without
hunting through general Settings or losing the loaded client/map.

**Why this priority**: The operator likes the UI but cannot easily find its readability/recovery
controls. Reset must be discoverable and predictable.

**Independent Test**: Change text scale, use the named reset choices in View, and verify their effects
immediately and after restarting, without opening general Settings.

**Acceptance Scenarios**:

1. **Given** the viewer is open, **When** the user opens View, **Then** text scale, scale reset,
   layout reset and existing visual-setting groups are reachable with readable names.
2. **Given** a custom text scale, **When** the user resets scale, **Then** the user adjustment returns
   to its standard default while display-appropriate sizing remains active and the client/map stays loaded.
3. **Given** custom layout and scale, **When** the user chooses layout-only reset, **Then** layout
   returns to default without silently changing text scale; an explicitly named combined reset resets both.
4. **Given** a scale change, **When** a tool window opens or the viewer restarts, **Then** the chosen
   scale is consistent and labels/controls do not overlap or clip because of the change.

---

### User Story 2 - Start exploring from Maps after opening a client (Priority: P1)

A user opens a supported client and sees Maps automatically, ready to select a map, without needing
to know the toolbar first.

**Why this priority**: Client opening should naturally lead to choosing a world rather than leaving
the user in an empty scene with undiscovered navigation.

**Independent Test**: Open a client, close Maps, then open a different client. Confirm one Maps
window opens for each successful activation and shows the current client's discovery result.

**Acceptance Scenarios**:

1. **Given** no active client, **When** a client finishes loading and map discovery completes,
   **Then** Maps opens with that client's available maps.
2. **Given** another client was active, **When** a replacement client loads successfully, **Then**
   the same Maps window opens or comes forward with the replacement client's maps; no duplicate appears.
3. **Given** the user closes Maps after automatic opening, **When** ordinary updates or discovery
   refreshes continue, **Then** Maps stays closed until a new client activation or explicit request.
4. **Given** client opening fails or is cancelled, **When** the operation ends, **Then** no automatic
   opening falsely presents the previous client's maps as the new result.
5. **Given** a successfully loaded client has no maps, **When** Maps opens, **Then** it shows a
   concise empty state, not a stale list or repeated opening loop.

---

### User Story 3 - Learn the current viewer from real screenshots (Priority: P2)

A new user follows screenshots of the actual icon bars, View menu, Maps window, spawn chooser and
tool windows. Captions explain which control to use, what to expect, how to close it and how to
recover the view.

**Why this priority**: A low-noise icon UI needs visual onboarding, not instructions for retired
sidebars or unreleased features.

**Independent Test**: Follow the illustrated guide against its documented viewer build to open a
client, choose a map/spawn, open/close a tool window, adjust/reset scale and find a workbench page.

**Acceptance Scenarios**:

1. **Given** the guide, **When** the user reads its overview, **Then** a genuine current-viewer
   screenshot identifies the menu, icon bars, map/minimap, chat and tool-window behavior.
2. **Given** a documented workflow, **When** the user follows it, **Then** screenshot labels,
   captions and click targets match the documented running build.
3. **Given** View/scale reset, client-to-Maps, spawn choice, inspection or workbench navigation,
   **When** the user reaches its guide section, **Then** real screenshots illustrate that workflow.
4. **Given** an image is not captured or approved, **When** guide readiness is reported, **Then**
   that gap stays open; diagrams, old UI images and placeholders do not count as current screenshots.

### Edge Cases

- Saved custom scale, invalid scale or an older persisted layout is loaded at startup.
- A window is resized or moved between displays with different scaling; reset must not force
  unreadably small physical text on a high-resolution display.
- Scale changes while menus or tool windows are already open.
- Clients are opened through supported folder, saved-client, archive/installation or startup routes.
- The user switches clients with Maps already open, refreshes the current client's map list, or
  closes Maps before ordinary updates continue.
- Client loading is cancelled/fails, discovery errors, or a successful client contains no maps.
- A proposed screenshot includes private paths or an older build/UI and needs review before publication.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: View MUST be the directly discoverable menu home for existing camera, visual
  presentation, interface visibility and scale controls. Users MUST NOT need general Settings to
  access these actions. Logical submenus are allowed.
- **FR-002**: View MUST show the current user-adjustable text scale and allow immediate adjustment
  through clearly named controls.
- **FR-003**: View MUST offer an explicit scale reset restoring the standard user adjustment without
  disabling display-aware readability or unloading the active client/map.
- **FR-004**: Reset choices MUST distinguish layout-only, scale-only and an explicitly named
  combined layout-and-scale reset. Labels MUST describe what will be reset.
- **FR-005**: Menu actions MUST operate on the same existing settings as their tool-window routes;
  exposing a setting in View MUST NOT create conflicting copies of it.
- **FR-006**: Scale adjustments/reset results MUST be consistent across menus, tool windows and
  restart. Supported scales MUST preserve legible labels and usable, non-overlapping controls.
- **FR-007**: Maps MUST automatically open once after each successful client activation and map
  discovery result, including replacement of an already loaded client through supported load routes.
- **FR-008**: Automatic opening MUST reuse the existing Maps window with the active client's result.
  Closing it MUST be respected until a new client activation or explicit open request.
- **FR-009**: Failed/cancelled opening MUST NOT trigger success behavior. Successfully loaded clients
  without maps MUST have an honest empty state, not a stale map list.
- **FR-010**: The guide MUST describe the accepted icon-bar UI and implemented View/Maps behavior,
  replacing retired-sidebar instructions in affected workflows without removing documentation of
  unrelated still-supported features.
- **FR-011**: The guide MUST include genuine screenshots and readable captions for the shell,
  View/scale/reset, client-to-Maps flow, spawn chooser, inspection and workbench navigation. It MUST
  identify the viewer build/version used for the illustrated UI.
- **FR-012**: Screenshot-dependent completion MUST require actual captured, reviewed images.
  Missing captures, mockups, old screenshots and build/test output MUST NOT count as current
  screenshot or real-client behavior proof.
- **FR-013**: Preserve the accepted scene-first interaction model, full-name tooltips and existing
  view behavior. Do not reintroduce the Spellbook, abbreviated rail labels or permanent data sidebars.

### Key Entities *(include if feature involves data)*

- **View preference**: Existing presentation/visibility/text-scale choice, its value and default;
  shared across every route that exposes it.
- **Reset choice**: Clearly named layout, user-scale or combined recovery action preserving the
  client and scene.
- **Client activation**: Successful change of active client and its discovery result, distinct from
  ordinary updates/refreshes within that client.
- **Guide illustration**: Real screenshot associated with a documented build, workflow, caption and
  review state. Missing/unapproved images remain completion gaps.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Scale adjustment, all three reset choices and every existing visual-setting group are
  reachable by opening View and at most one submenu, without general Settings.
- **SC-002**: All three reset choices produce exactly their named effects in tested cases, preserve
  the loaded client/map and retain the result after restart.
- **SC-003**: At standard and representative supported custom scales on 1080p, 1440p and 4K displays,
  View and tested tool windows have no scale-induced overlapping text or clipped controls.
- **SC-004**: Each tested successful client activation opens exactly one current Maps window;
  closing it is respected through subsequent ordinary updates and discovery refreshes.
- **SC-005**: All tested failed/cancelled/no-map transitions show no false-success opening or stale
  client map list.
- **SC-006**: Every workflow in FR-011 has a real reviewed screenshot and matching instructions;
  every image link resolves and every capture is tied to the documented viewer build.
- **SC-007**: A guide walkthrough completes client opening, map/spawn selection, tool open/close,
  scale recovery and workbench-page selection without reference to retired sidebars, the rejected
  Spellbook or an unreleased feature presented as available.

## Assumptions

- Reuse View; this is not another menu system, theme or shell redesign.
- The operator requests a discoverable scale-reset option, not silent changes to the existing
  layout-only reset. Explicit separate choices preserve that distinction.
- Scale reset restores the normal user preference while retaining suitable automatic display sizing.
  The supported adjustment range is resolved from current behavior during implementation planning.
- "Maps tab" means the existing Maps tool window, not a new permanent tab.
- Exposing existing visual settings does not authorize new renderer features, camera behavior,
  converter changes, or completion of Explore.
- Update the existing user guide; do not add a help engine or release/version bump as implied scope.
  Broader integrated help remains owned by [Spec 271](../271-v060-release-and-integrated-help/spec.md).
- The operator owns real-client/GPU/visual runs and screenshot capture. Agents can prepare bounded
  capture instructions and integrate approved images, but tests/builds cannot close visual criteria.
- No game data is bundled. Client roots remain runtime configuration; portable documentation must
  not hardcode machine-local client paths.
