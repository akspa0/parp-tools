# Feature Specification: Precise Object Selection, PM4 Match Confirmation, and a World-Space Cursor

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
dependency (User Story 2). Extending PM4 identity to regular placements *automatically* is recorded as
explicitly out of scope (see Assumptions) rather than silently promised.

That is not the only way to attack the correlation problem, though. Automatic matching being unsolved
does not mean a *human* can't tell two objects are the same the moment picking is reliable enough to
trust which two things were actually clicked — and every one of those human judgments is a labeled data
point the automatic matchers in specs 046/065 never had. User Story 3 turns that into a durable,
queryable library: confirm a PM4 object and a real placement are the same thing, and it's recorded for
good — both as one fewer object left to identify, and as a growing reference set future matching work
can be measured against or built from. This is explicitly *not* the same thing as FR-009's prohibition
on automatic/silent correlation — it is a separate, always-explicit, always-evidenced human action, kept
distinct from the live picking mechanism throughout this spec.

The mouse-as-a-world-space-object half of the request is independent of all of the above and is scoped
as its own user story (User Story 4).

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

### User Story 3 - Confirm a PM4 object's real identity, and keep a growing library of what's known (Priority: P3)

While inspecting the scene, a user who can see that a specific PM4 object and a specific real placement
(a WMO or an M2/MDX doodad) are clearly the same object — because their position, footprint, and shape
line up once actually looked at side by side — explicitly confirms that match. The confirmation is
recorded permanently: which PM4 object (tile, CK24, part), which real placement (build, tile,
UniqueId/asset path), when, and why the user is confident. Confirmed matches accumulate into a library
that answers, for any PM4 object, whether it's already been identified and what to — and surfaces other
PM4 objects that share a confirmed object's geometry fingerprint as candidates worth reviewing next,
without ever confirming them automatically.

**Why this priority**: This is the user's second explicit goal for this spec, not a stretch add-on:
build real ground truth for the correlation problem specs 046/065 measured as unsolved, by capturing
what a human can already tell just by looking. Every confirmed match is simultaneously one fewer object
to figure out later and a labeled reference point future matching work can be measured against or built
from. Ordered after Stories 1 and 2 because a confirmation is only as trustworthy as the click that
produced it, and precise picking is what removes the ambiguity a box-based click leaves about which
object was actually meant.

**Independent Test**: Works with today's existing (box-based) selection, without needing Stories 1 or 2
built first. Select any PM4 object and any placement a user is confident represents the same real
object, confirm the match through whatever UI this story adds, restart the viewer, and verify the
confirmation is still there and answers a lookup for that PM4 object. Confirming and retracting a match
work standalone; the library is only more trustworthy to build once Stories 1 and 2 exist, not dependent
on them existing.

**Acceptance Scenarios**:

1. **Given** a PM4 object selected and a real placement selected that the user believes are the same
   object, **When** the user confirms the match, **Then** the pairing (PM4 identity, placement identity,
   confirmation evidence, timestamp) is recorded durably and survives a restart.
2. **Given** a PM4 object that has already been confirmed-matched, **When** the user selects it or looks
   it up, **Then** its confirmed match is shown rather than treated as unknown.
3. **Given** a confirmed match that was made in error, **When** the user retracts it, **Then** the
   retraction is recorded (not silently deleted without trace) and the object returns to
   unconfirmed/candidate status.
4. **Given** two objects that are merely near each other or share a tile, **When** nothing has been
   explicitly confirmed, **Then** they are never auto-confirmed — confirmation is always an explicit user
   action, never inferred from proximity or any matcher's score alone.
5. **Given** a confirmed match exists for one instance of an asset, **When** another PM4 object elsewhere
   in the corpus shares that confirmed object's geometry fingerprint, **Then** the library surfaces it as
   a candidate for review, and a human still confirms or rejects it individually.

---

### User Story 4 - See exactly where the cursor is in the world (Priority: P4)

While moving the mouse over the 3D scene, the user sees a marker rendered as a real object in the world —
with correct depth and correct occlusion by nearer geometry — at the point their cursor currently
intersects the world, so they can see where a click will land before they click, the same way the marker
would be hidden behind a hill or wall in the real game client.

**Why this priority**: Valuable on its own and independently testable, but it is a visualization/feedback
improvement layered on top of picking and identification, not a fix to selection accuracy or a
contribution to the match library — Stories 1 through 3 are where this spec's substance lives. Ordered
last because it depends on nothing the others add (it can track terrain today) but delivers most of its
value once precise object surfaces exist to track in Story 1.

**Independent Test**: Move the mouse across varied terrain (a slope, a cliff edge) with no other change
applied; confirm a visible marker appears at the terrain point under the cursor, moves with the cursor,
and disappears behind the terrain when the geometry would occlude that point from the current camera
angle. This does not require Stories 1 through 3 to be built first.

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
- A user attempts to confirm a match for a PM4 object (or placement) that already has a *different*
  confirmed match: the existing confirmation is not silently overwritten — the conflict is surfaced so
  the user chooses which is correct, and the losing confirmation's history is retained, not erased.
- Two different users (or the same user on different occasions) confirm the same pairing independently:
  this reinforces confidence in the match rather than being treated as a conflict.
- A candidate surfaced from a shared geometry fingerprint (Acceptance Scenario 5 of User Story 3) turns
  out to be a coincidental resemblance, not the same asset: rejecting it must be as durable and visible
  a record as confirming one, so the same false candidate is not resurfaced as new every session.

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
- **FR-009**: System MUST NOT automatically or silently resolve PM4 object identity to ADT MDDF/MODF
  placement identity, and MUST NOT substitute PM4-derived geometry as the live selection volume for a
  regular scene object. Explicit, human-confirmed correlation between a PM4 object and a placement is a
  separate, in-scope capability (User Story 3, FR-011 through FR-016) that produces a recorded match, and
  never changes how the regular object is picked.
- **FR-010**: This feature MUST NOT change selection behavior for any object or state it does not cover
  (e.g. objects using the bounding-volume fallback keep exactly today's behavior).
- **FR-011**: System MUST let a user explicitly record that a specific PM4 object and a specific real
  placement are confirmed to be the same real-world object, capturing both identities, when the
  confirmation was made, and the user's stated reason for confidence.
- **FR-012**: A confirmed match MUST NOT be inferred automatically from proximity, shared tile membership,
  or any matcher's score alone — confirmation is always an explicit user action.
- **FR-013**: System MUST let a user retract a previously confirmed match; the retraction MUST be
  recorded as its own event rather than erasing the original confirmation without trace.
- **FR-014**: The confirmed-match library MUST be queryable by PM4 object identity (tile, CK24, part) to
  answer whether it is already known and to what, without re-deriving the answer.
- **FR-015**: When a confirmed match exists for a PM4 object, System MUST reuse that object's
  already-computed geometry fingerprint (the existing correlation extractor, not a new one) to surface
  other unconfirmed PM4 objects sharing that fingerprint as review candidates — candidates only, never
  auto-confirmed.
- **FR-016**: The confirmed-match library MUST persist across sessions and MUST NOT store client asset
  bytes — only identifiers, paths, and provenance, consistent with this project's existing
  no-client-data-in-repository constraint.

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
- **Confirmed Match**: a durable, evidenced record pairing one PM4 object identity (tile, CK24, part)
  with one real placement identity (build, ADT/tile, MDDF or MODF, UniqueId or equivalent, asset path);
  carries who/when/why it was confirmed and its current status (confirmed or retracted).
- **Match Candidate**: a suggested, unconfirmed pairing surfaced for human review — from simple signals
  (shared tile, position proximity) or from a shared geometry fingerprint with an already-confirmed
  match; never itself a Confirmed Match until a user acts on it.

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
- **SC-006**: Regular (non-overlay) placed objects never acquire a PM4-sourced selection *volume* through
  this feature's picking mechanism — confirmed by inspection that no code path substitutes PM4 geometry
  for a regular object's hit-test bounds. (Explicit, human-confirmed identity correlation is a separate,
  opt-in capability — User Story 3 — and does not change how regular objects are picked.)
- **SC-007**: A user can confirm a PM4-object-to-placement match and retrieve that confirmation after
  restarting the viewer.
- **SC-008**: Retracting a confirmed match, or rejecting a surfaced candidate, leaves an auditable record
  of that action rather than a silent deletion.
- **SC-009**: No match is ever recorded as confirmed without an explicit user action — confirmed by
  inspection that no code path writes a Confirmed Match from a score or proximity threshold alone.

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
- Extending PM4-derived precision to regular (non-overlay) placed objects' *picking mechanism*
  generally — the literal "for every object" phrasing of the original request, read as "use PM4 geometry
  as the live selection volume" — is out of scope for this feature. It would require solving
  CK24-to-MDDF/MODF correlation automatically, currently measured at 1.3% top-1 precision (specs 046,
  065), and this feature does not attempt that. User Story 3 pursues the same underlying goal — knowing
  which PM4 object is which real object — through explicit human confirmation instead, which needs no
  automatic solution first and produces evidence future automatic-matching work can use.
- The confirmed-match library (User Story 3) is a growing research/data artifact, not a claim that
  automatic PM4-to-placement matching is solved by this feature. Whether accumulated confirmed matches
  meaningfully improve the existing matcher's precision is a downstream empirical question for future
  work, not a promised outcome measured here — this spec is responsible for the library existing, being
  queryable, and being trustworthy (always explicit, always evidenced, always retractable), not for
  automatic matching accuracy.
- The confirmed-match library's storage format and location are planning-phase decisions; this spec only
  requires that it persists across sessions, is queryable by PM4 object identity, and never stores client
  asset bytes — identifiers, paths, and provenance only, matching this project's existing convention for
  every other generated record (e.g. Spec 155's sweep reports).
- Geometry fingerprints for surfacing match candidates (FR-015) reuse the existing correlation extractor
  as-is; this feature does not modify or improve the fingerprint algorithm itself.
- Sound, taxi nodes, area POIs, and liquid-body selection (all already handled by separate paths in the
  existing click-selection flow) are unaffected by this feature.
- "World-space cursor marker" means a rendered indicator of the current hit point, not a change to how
  the OS/UI mouse cursor itself is drawn; the existing UI cursor is unaffected.
