# Feature Specification: Real Client Cursor Model and Precise PM4 Picking

**Feature Branch**: `282-client-cursor-and-pm4-pick`

**Created**: 2026-10-21

**Status**: Draft (operator-originated)

**Input**: Operator, 2026-10-21: "clicking on pm4 objects does not always select the pm4 that was
clicked on, in fact, it seldom highlights any of the pm4 object data that is near the click. That's
not right. It's also hard to click on any objects at all. We need to build in support for the real
client's cursor.mdx/cursor.m2 and actually use it, render it as an object in the viewer that we
control and use as a mouse to interact with all the 3d objects in the viewer, instead of the
shit-ass 2d mouse that does not click things properly. I asked for this months ago and got a
shitty '3d cursor' that doesn't fucking work 99% of the time."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The client cursor model is the pointer (Priority: P1)

The operator points at the world and sees the client's own cursor model (`Interface\Cursor\cursor.mdx`
or `cursor.m2`) rendered in the scene at the point under the pointer, oriented to the camera, and
scaled so it reads as a pointer. The hardware 2D arrow is hidden while the model cursor is active.

**Why this priority**: The operator has asked for this repeatedly and the existing 3D cursor is not
the client cursor and does not track the pointer reliably.

**Independent Test**: Load a world, enable the cursor model, and move the pointer across terrain,
WMO and M2 surfaces. The model follows the surface under the pointer, stays upright and readable,
and the 2D arrow is hidden.

**Acceptance Scenarios**:

1. **Given** a loaded world and the cursor model enabled, **When** the pointer moves over terrain,
   **Then** the cursor model is drawn at the terrain hit point and faces the camera.
2. **Given** the pointer is over a WMO or M2 surface, **When** it moves, **Then** the cursor model
   is drawn at that surface's hit point, not at the terrain behind it.
3. **Given** the pointer is over chrome (sidebar, bar, popup), **When** it moves, **Then** the
   cursor model is hidden and the normal 2D pointer is shown.
4. **Given** the active data source has no `cursor.mdx`/`cursor.m2`, **When** the cursor model is
   enabled, **Then** the viewer reports the missing asset and falls back to the 2D pointer.

### User Story 2 - Clicks select what is under the cursor (Priority: P1)

The operator clicks a PM4 object and the object under the cursor is selected, not a different
object whose bounding box happens to be nearer along the ray.

**Why this priority**: The operator reports that PM4 selection "seldom highlights any of the pm4
object data that is near the click", which makes PM4 inspection unusable.

**Independent Test**: Click the centre of several PM4 surfaces at different depths and confirm the
selected object is the one drawn under the pointer.

**Acceptance Scenarios**:

1. **Given** the PM4 overlay is on, **When** the operator clicks a PM4 surface, **Then** the
   selected object is the surface under the pointer.
2. **Given** two PM4 objects overlap in screen space, **When** the operator clicks the nearer one,
   **Then** the nearer one is selected.
3. **Given** the operator clicks empty space, **When** no PM4 surface is under the pointer,
   **Then** no PM4 object is selected.

### Edge Cases

- The cursor model is missing, unreadable, or has no geometry: report it, keep the 2D pointer.
- The pointer is over a transparent or very thin surface: the pick must still resolve the nearest
  drawn surface, not the first bounding box.
- The cursor model must not be picked by its own ray (it is chrome, not scene content).
- A cursor model with animations: play the idle sequence, or hold the first frame if none exists.

## Requirements *(mandatory)*

- **FR-001**: The viewer MUST resolve `Interface\Cursor\cursor.mdx` and `Interface\Cursor\cursor.m2`
  from the active data source (MPQ, CASC, loose folder, WC3 Beta) using the existing asset path
  resolver, without hardcoding a client path.
- **FR-002**: The viewer MUST render the resolved cursor model at the world point under the pointer,
  oriented to the camera, with a configurable scale, using the existing model renderer.
- **FR-003**: The viewer MUST hide the hardware pointer while the model cursor is active and over
  the viewport, and restore it over chrome.
- **FR-004**: The cursor model MUST be excluded from scene picking and from its own hover.
- **FR-005**: PM4 picking MUST test the actual PM4 surface triangles, not only object bounding
  boxes, and select the nearest triangle hit under the pointer.
- **FR-006**: PM4 picking MUST prefer the surface drawn under the pointer when several objects
  overlap in screen space.
- **FR-007**: The cursor model and PM4 picking MUST be toggleable from Options > Objects and MUST
  be documented in the generated UI reference.
- **FR-008**: A missing or unreadable cursor model MUST be reported in the status line and MUST
  fall back to the 2D pointer.

## Success Criteria *(mandatory)*

- **SC-001**: The cursor model follows the pointer over terrain, WMO and M2 surfaces with no visible
  lag or jump.
- **SC-002**: Clicking a PM4 surface selects that surface's object in at least 95% of clicks on
  visible surfaces in an operator spot check.
- **SC-003**: The 2D pointer is hidden only while the model cursor is drawn.
- **SC-004**: A data source without the cursor asset produces an explicit message, not a silent
  no-op.

## Assumptions

- The existing model renderer can draw a small standalone model at an arbitrary world transform;
  this spec adds the placement and lifetime, not a new renderer.
- PM4 surface triangles are already available in the decoded overlay data; this spec adds the
  triangle test, not a new decoder.
- The cursor model is chrome: it is drawn after the scene and before ImGui, and never participates
  in scene selection.
