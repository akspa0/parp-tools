# Feature Specification: Precise Object Selection — Real Geometry Picking and a World-Space Cursor

**Feature Branch**: `156-precise-object-selection`

**Created**: 2026-08-16

**Status**: Draft

**Input**: User description: "build pm4 objects for every object and use that data for literally better
conforming object bounding boxes, instead of the horrible giant boxes we employ currently. The mouse ray
hit stuff is really oldschool and horrible in practice, we need to improve how our mouse works. The game
engine literally renders the mouse as an object in the renderer, that the user has full control over, at
all times. that's the solution to the object selection issue, the mouse becomes a floating thing in xyz
space, instead of a 2D mouse cursor that cannot possibly select the right things with ray hits."

## Scope Note — read before the user stories

The request as stated proposes PM4 geometry as the mechanism for giving every placed object a precise
selection volume. Two prior specs already measured that mechanism for *regular* placed objects and it
does not work: matching a PM4 object to the ADT placement (MDDF/MODF) it corresponds to is currently
accurate 1.3% of the time (`specs/046-pm4-asset-matching`, `specs/065-pm4-correlation-to-world-assets`,
both closed findings, not re-litigated here). Using PM4 data to build a regular object's selection volume
requires knowing *which* PM4 object is that placement, and that lookup does not currently exist.

There is a better source of the same thing for regular objects: the object's own mesh, already loaded
and already rendered every frame. Testing a click against that mesh directly delivers what "precise
selection for every object" actually means, without depending on the unsolved correlation problem at
all. This spec leads with that path (User Story 1) and scopes the PM4-specific angle to where it already
works today — the PM4 overlay's own objects, which are not placements and carry no correlation
dependency (User Story 2). Extending PM4 identity to regular placements is recorded as explicitly
out of scope (see Assumptions) rather than silently promised.

The second half of the request — the mouse as a world-space object — is independent of both of the
above and is scoped as its own user story (User Story 3).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Click the model you're looking at, not the box around it (Priority: P1)

A user viewing the scene clicks on a doodad or building that is oddly shaped, sparse, hollow, or
mid-animation — an L-shaped building, a scaffolding-like prop, a creature with its arm raised. The click
selects that object only when the cursor is actually over its visible surface. Clicking empty space next
to the object — space that today falls inside its bounding box but is visibly not part of the model —
does not select it, and does not select whatever lies behind that empty space either.

**Why this priority**: This is the direct, general fix for "the horrible giant boxes" and "cannot
possibly select the right things" — it applies to every regular placed object (every WMO and every M2
doodad in the scene), not a subset, and depends on nothing unresolved. The object's render mesh is
already loaded and drawn every frame; this reuses it rather than building or resolving anything new.

**Independent Test**: Load a scene containing a known non-convex object (e.g., an L-shaped building or a
sparse foliage model) whose bounding box visibly extends into empty space. Click a point inside the box
but outside the visible mesh; confirm nothing is selected (or whatever is actually there behind it is
selected instead). Click a point on the visible mesh; confirm the object is selected. This delivers
value standalone — no other user story in this spec needs to exist for this one to be complete and
demonstrable.

**Acceptance Scenarios**:

1. **Given** a placed object whose bounding volume includes empty space not covered by its mesh, **When**
   the user clicks a point in that empty space, **Then** the object is not selected by that click.
2. **Given** the same object, **When** the user clicks a point on its visibly rendered surface, **Then**
   the object is selected.
3. **Given** an animated creature doodad rendered mid-animation with a limb extended beyond its resting
   pose, **When** the user clicks the limb's current on-screen position, **Then** the object is selected
   (precise picking follows what is visibly drawn, not a fixed resting-pose assumption, to the extent the
   picked geometry can reflect current pose — see Assumptions for the static-pose fallback case).
4. **Given** an object whose asset failed to load or has not finished loading, **When** the user clicks
   near it, **Then** selection behavior is unchanged from today (existing bounding-volume fallback),
   never a crash and never a silently unpickable object.

---

### User Story 2 - Distinguish adjacent PM4 objects by their real shape (Priority: P2)

An investigator using the PM4 diagnostic overlay hovers and clicks among several PM4 objects whose
bounding boxes overlap. Each object is selected by its own decoded surface, so two adjacent objects with
overlapping boxes can be told apart by clicking on what is actually drawn for each.

**Why this priority**: The real triangle/line geometry PM4 objects need for this already exists in memory
(it is what draws the overlay); this is a picking-side change only, with no new geometry assembly and no
dependency on User Story 1. Scoped below Story 1 because it improves a diagnostic overlay used by
developers, not the general scene the ordinary user interacts with.

**Independent Test**: Load a build with the PM4 overlay enabled in an area where two PM4 objects' boxes
are known to overlap but their decoded surfaces do not. Click inside the overlapping-box region on each
object's actual surface in turn; confirm each click selects the correct object.

**Acceptance Scenarios**:

1. **Given** two PM4 objects whose bounding boxes overlap but whose decoded triangle/line geometry does
   not, **When** the user clicks a point inside the overlap that lies on only one object's geometry,
   **Then** that object is selected and not the other.
2. **Given** a PM4 object whose decoded geometry is empty (assembly produced no triangles or lines for
   it), **When** the user clicks within its bounding box, **Then** the existing bounding-volume test is
   used as a fallback rather than the object becoming unpickable.

---

### User Story 3 - See exactly where the cursor is in the world (Priority: P3)

While moving the mouse over the 3D scene, the user sees a marker rendered as a real object in the world —
with correct depth and correct occlusion by nearer geometry — at the point their cursor currently
intersects the world, so they can see where a click will land before they click, the same way the marker
would be hidden behind a hill or wall in the real game client.

**Why this priority**: Valuable on its own and independently testable, but it is a visualization/feedback
improvement layered on top of picking, not a fix to selection accuracy itself — Stories 1 and 2 are what
actually make selection precise. Ordered after them because it depends on nothing they add (it can track
terrain today) but delivers most of its value once precise object surfaces exist to track in Story 1.

**Independent Test**: Move the mouse across varied terrain (a slope, a cliff edge) with no other change
applied; confirm a visible marker appears at the terrain point under the cursor, moves with the cursor,
and disappears behind the terrain when the geometry would occlude that point from the current camera
angle. This does not require Stories 1 or 2 to be built first.

**Acceptance Scenarios**:

1. **Given** the cursor is over the terrain, **When** the user moves the mouse, **Then** a world-space
   marker tracks the point on the terrain surface under the cursor, updated every frame the cursor moves.
2. **Given** the marker's tracked point is behind a hill or wall relative to the current camera, **When**
   the scene renders, **Then** the marker is not visible (correctly occluded), not drawn on top of the
   occluding geometry.
3. **Given** the cursor points at empty sky with no terrain or object under it, **When** the scene
   renders, **Then** no marker is shown (or the marker is clearly distinguished as "no hit", never shown
   at a misleading fixed distance).
4. **Given** Story 1 has been delivered, **When** the cursor is over a precisely-pickable object's
   surface rather than terrain, **Then** the marker tracks that object's surface instead of passing
   through it to the terrain behind it.

---

### Edge Cases

- Two candidate objects' precise geometry both intersect the ray at different distances: the nearer
  intersection wins, exactly as box-based picking already resolves distance today.
- Two candidate objects' precise geometry both intersect the ray at effectively the same point (rare,
  e.g. coincident/overlapping placements): the existing multi-candidate disambiguation overlay is shown,
  the same UX already used when multiple bounding-volume hits are ambiguous — this feature must not
  bypass or weaken that existing mechanism.
- A model's mesh is a known-degenerate case for the picking test (zero triangles, all-degenerate
  triangles, a decal-thin sliver at a grazing ray angle): falls back to bounding-volume picking for that
  object rather than becoming unpickable or erroring.
- A dense scene has many candidate objects near the cursor at once (e.g. a city center): precise
  per-triangle testing must not be run against every candidate's full mesh unconditionally — a cheap
  bounding-volume test (today's existing mechanism) is expected to remain the first-pass filter, with
  precise testing applied only to the small set of candidates that pass it. This is a direct consequence
  of this project's existing measured frame-pacing sensitivity and must not reintroduce it.
- PM4 surfaces are fan-triangulated from possibly non-convex loops (an existing, known property of the
  current MSUR assembly, not introduced by this feature); a precise pick may occasionally reflect a fan
  artifact already present in the rendered geometry. This is a pre-existing geometry-assembly property
  becoming newly visible through picking, not a new defect this feature introduces.
- The world-space cursor marker while the camera itself is moving/rotating: the marker must update to
  the new ray's hit point every frame, never lag a frame behind or show a stale position.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST test selection hits for regular placed objects (WMO and M2/MDX) against the
  object's actual render mesh geometry, when that geometry is available, rather than only its bounding
  volume.
- **FR-002**: System MUST fall back to today's existing bounding-volume picking for any object whose
  precise mesh geometry is unavailable (not yet loaded, failed to load, or degenerate), and MUST NOT
  make such an object unpickable as a result of this feature.
- **FR-003**: System MUST use a cheap bounding-volume test as a first-pass filter before any precise
  geometry test, so precise testing runs only against the small set of candidates near the cursor rather
  than every loaded object in the scene.
- **FR-004**: PM4-overlay object picking MUST test against the object's already-assembled triangle/line
  geometry rather than its bounding box, when that geometry is non-empty, and MUST fall back to the
  bounding-volume test when it is empty.
- **FR-005**: When a pick ray's precise-geometry test resolves more than one candidate object at
  effectively the same hit distance, System MUST continue to present the existing multi-candidate
  disambiguation UX rather than arbitrarily resolving to one.
- **FR-006**: System MUST render a marker at the world-space point the cursor currently intersects,
  submitted through the same view/projection pipeline as other scene geometry, so it is correctly scaled,
  depth-tested, and occluded by nearer geometry rather than drawn as a flat screen overlay.
- **FR-007**: The world-space cursor marker MUST update every frame the cursor's position or the camera
  changes, and MUST show no marker (or an explicit no-hit state) when nothing is under the cursor.
- **FR-008**: The world-space cursor marker MUST track real terrain-surface hit points at minimum; once
  User Story 1 is delivered, it MUST also track precise object-surface hit points using that same
  hit-test path rather than a separate one.
- **FR-009**: System MUST NOT attempt to resolve PM4 object identity to ADT MDDF/MODF placement identity
  as part of this feature. Precise PM4-derived geometry applies only within the PM4 overlay's own object
  identity (User Story 2); it MUST NOT be presented as, or silently become, the selection volume for a
  regular scene object.
- **FR-010**: This feature MUST NOT change selection behavior for any object or state it does not cover
  (e.g. objects using the bounding-volume fallback keep exactly today's behavior).

### Key Entities

- **Pick Ray**: the world-space origin and direction unprojected from the cursor's current screen
  position for the current camera, recomputed each frame the cursor or camera moves.
- **Pick Candidate**: an object eligible for hit-testing against a pick ray; carries whichever of
  (precise mesh geometry, bounding volume) is currently available for it, and which of those was
  actually used to resolve the current hit.
- **Hit Result**: the outcome of testing a pick ray against the scene — nearest intersection distance,
  world-space point, and a reference to which object and which test kind (precise or bounding-volume)
  produced it, or no-hit.
- **World Cursor Marker**: the rendered representation of the most recent Hit Result, positioned and
  depth-tested as a real scene object rather than a screen-space overlay.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can select a non-convex placed object (e.g. an L-shaped building) by clicking its
  visible surface, and clicking within its bounding volume but outside its visible surface does not
  select it — demonstrated on a known real object where today's box-based picking currently over-selects.
- **SC-002**: A user can independently select each of two PM4 overlay objects whose bounding boxes
  overlap but whose decoded geometry does not, by clicking each one's own surface.
- **SC-003**: The world-space cursor marker is visibly occluded by nearer terrain in a scene with
  elevation changes, rather than showing through it.
- **SC-004**: Hover and click responsiveness in a densely populated scene (at least as many nearby
  candidate objects as this project's existing dense-city test scenes) shows no perceptible added lag
  after this feature ships, consistent with this project's existing frame-pacing standards.
- **SC-005**: No selection behavior changes for objects in the bounding-volume-fallback state (unloaded
  or failed assets) — confirmed unchanged before/after.
- **SC-006**: Regular (non-overlay) placed objects never acquire a PM4-sourced selection volume through
  this feature — confirmed by inspection that no code path resolves a regular placement to a PM4 CK24
  identity.

## Assumptions

- The render mesh data (vertices and triangle indices) already loaded to draw a WMO or M2/MDX object each
  frame is assumed to be reachable from the code path that performs picking, in some form suitable for a
  CPU-side ray intersection test. Confirming exactly what form that data is in today (and whether any of
  it is GPU-only and would need a readback or a duplicate CPU-side copy) is left to the planning phase's
  research, not asserted as settled here.
- For an animated M2 doodad, testing against its *currently posed* mesh (matching what is visibly
  rendered, per Acceptance Scenario 3 of User Story 1) is preferred, but testing against its static
  resting-pose mesh is an acceptable fallback if current-pose vertex data is not practically available on
  the picking path — this would be a smaller improvement over today (still fixes the animation-range
  header-box overshoot) but not a complete fix for a limb mid-swing. Which of these is achievable is a
  planning-phase question; this spec accepts either as satisfying User Story 1 provided the chosen
  behavior is stated plainly, not silently assumed to be current-pose.
- Extending PM4-derived precision to regular (non-overlay) placed objects generally — the literal "for
  every object" phrasing of the original request — is out of scope for this feature. It requires solving
  CK24-to-MDDF/MODF correlation, currently measured at 1.3% top-1 precision (specs 046, 065), and is not
  attempted here. This spec should not be read as declining that goal — only as recording that it is not
  achievable by this feature as scoped, and pointing at the specific unsolved problem that blocks it.
- Sound, taxi nodes, area POIs, and liquid-body selection (all already handled by separate paths in the
  existing click-selection flow) are unaffected by this feature.
- "World-space cursor marker" means a rendered indicator of the current hit point, not a change to how
  the OS/UI mouse cursor itself is drawn; the existing UI cursor is unaffected.
