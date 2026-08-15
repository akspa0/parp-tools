# Feature Specification: Renderer frame-time stability and per-era terrain lighting

**Feature Branch**: `v0.5.3-dev`

**Created**: 2026-08-15

**Status**: Draft

**Input**: The renderer "gallops" — jagged frame pacing with periodic hitches and FPS dips during
camera movement. Terrain also renders too dark on client builds 1.0.0 and later. v0.5.3 is the
release line that makes the renderer render properly.

## Context: why this was never caught

Two facts were confirmed by reading the current source on this branch before writing this spec.
They are the reason the defects survived every previous "renderer performance" pass.

1. **The automated profiler cannot see the defect.**
   `ProductionWorldSceneProfiler` resolves camera position, forward vector, view matrix, and
   projection matrix **once**, then passes those identical values to every rendered frame. The
   camera never moves, never crosses an ADT tile boundary, never changes streaming admission, and
   never triggers a mid-flight GPU tile upload. Its report emits no per-frame timing distribution
   of any kind — no per-frame wall-clock array, no maximum, no median, no p95/p99, no variance —
   only aggregate workload counters. Default sample is 12 measured frames after 8 warmup frames.
   A stationary 12-frame sample with no distribution **cannot** detect a movement-induced hitch.
   Any past "no regression found" result from this harness is a false null, not evidence.

2. **Terrain lighting has no era model.**
   `TerrainLighting` is documented as derived from Ghidra analysis of the Alpha 0.5.3 lighting
   system and exposes no build or era parameter. One Alpha-derived model is applied to every client
   build from 0.5.3 through 4.0.x. `LightService` *does* resolve exact-build `Light*` DBC chains, so
   the gap is in the terrain lighting model that consumes those values, not in DBC resolution.

The ordering consequence is the central constraint of this feature: **the harness must be made
capable of detecting the defect, and proven capable, before any renderer change is attempted.**

### Measured structural evidence for the scene-graph hypothesis

Read from source on this branch. These are the concrete costs behind "scene graphs were a wrong
turn". They are a strong hypothesis for the hitch mechanism, not yet a confirmed cause — confirming
them is US1/US2 work.

- **Culling costs more than it saves.** `WorldSceneTraversalDiagnostics.RecordRejectedSubtree`
  recursively walks the *entire rejected subtree* to attribute skipped counts by node kind. Rejecting
  a subtree is supposed to avoid touching it; here rejection triggers a full recursive walk of it,
  every frame. The larger the culled region, the more work culling costs.
- **Per-frame heap churn proportional to tile count.** `WorldSceneTraversal.Traverse` allocates two
  fresh `List<WorldSceneNode>` (visible *and* rejected), a diagnostics object holding four
  `Dictionary<WorldSceneNodeKind,int>`, and a result record — **per graph, per frame**. ADT tiles are
  deliberately isolated into independent graphs, so this multiplies by the resident tile count.
- **Caller-side per-frame allocation.** The scene-graph visibility preparation additionally performs
  a LINQ `.ToList()` and builds a `HashSet` of active graphs every frame.
- **Diagnostics run in the production path.** Rejected-node collection and per-kind attribution are
  not behind a diagnostic switch; they execute on every production frame.
- **Scale.** `WorldScene.cs` is ~15,456 lines with ~219 allocation-site matches. The renderer has no
  retained, flat, ordered draw list — visibility produces node objects that are then consumed
  per-object.

Periodic allocation of this shape produces exactly the symptom reported: mostly acceptable frames
punctuated by regular hitches, worst when the resident tile set changes.

### Measured evidence for the UI duplication complaint

- 214 distinct `Draw*` methods across the `ViewerApp*` files (~35,685 lines total).
- **71 of them are invoked from two or more call sites.** Some are legitimate shared widgets
  (`DrawColorRow`, `DrawTopTabButton`), but whole panels are multiply-routed:
  `DrawTerrainControlsAdjustmentContent` (8 sites), `DrawTerrainTileScopeSelector` (6),
  `DrawChunkClipboardContent` (6), `DrawSelectedObjectSummaryContent` (5),
  `DrawModelInfoPanelContent` (4).

This is the "massive data explorer with duplication" in numbers: the same surface is reachable and
rendered from many routes, with no single owner.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Prove the harness can see a hitch (Priority: P1)

As the maintainer, I need the automated profiler to reproduce the gallop I can see with my own eyes,
so that every later claim about the renderer rests on a measurement instead of an opinion.

**Why this priority**: Every other story in this feature depends on a trustworthy detector. Shipping
an optimization measured by a blind harness is worse than shipping nothing, because it manufactures
false confidence. This story is the only one that can be done first.

**Independent Test**: Run the profiler along a moving-camera trajectory that crosses ADT tile
boundaries, then inject a synthetic delay of known size into a known frame and confirm the report
flags it at the right frame index and the right magnitude. Delivers a detector whose power is
demonstrated rather than assumed.

**Acceptance Scenarios**:

1. **Given** a staged client and a camera trajectory that crosses at least one ADT tile boundary,
   **When** the profiler runs, **Then** the report contains a per-frame wall-clock series plus
   median, maximum, p95, p99, and a count of frames exceeding a stated hitch threshold.
2. **Given** a synthetic delay of known duration injected at a known frame index, **When** the
   profiler runs, **Then** the report identifies that frame as a hitch and reports its magnitude
   within a stated tolerance.
3. **Given** a run with no injected delay, **When** the profiler runs twice on identical inputs,
   **Then** the reported hitch statistics are stable enough that run-to-run noise cannot be mistaken
   for a real regression, and the noise floor is stated in the report.
4. **Given** a stationary camera, **When** the profiler runs, **Then** the report explicitly labels
   the trajectory as stationary so a stationary result can never be read as movement evidence.

---

### User Story 2 - Attribute each hitch to a cause (Priority: P2)

As the maintainer, I need each detected hitch classified as CPU work, GPU/driver time, or
I/O/streaming stall, so that I fix the actual cause instead of the most convenient suspect.

**Why this priority**: A hitch count alone does not tell anyone what to change. Attribution is what
converts a measurement into a work item. It depends on Story 1 but blocks Story 4.

**Independent Test**: Run the profiler over a boundary-crossing trajectory and confirm each flagged
hitch carries a dominant-cause attribution with the supporting per-stage numbers, then confirm the
attribution changes correctly when a known cause is artificially aggravated.

**Acceptance Scenarios**:

1. **Given** a flagged hitch frame, **When** the report is read, **Then** it names a dominant cause
   category and shows the per-stage evidence supporting that attribution.
2. **Given** terrain streaming is artificially slowed, **When** the profiler runs, **Then** the
   affected hitches are attributed to I/O/streaming rather than CPU or GPU.
3. **Given** the measurement cannot separate GPU time from driver time on the current setup,
   **When** the report is produced, **Then** it states that limitation explicitly rather than
   presenting an unsupported attribution.

---

### User Story 3 - Terrain renders at correct brightness on every supported era (Priority: P2)

As a user loading a 1.0.0-or-later client, I need terrain lit correctly for that client's era rather
than through an Alpha 0.5.3 model, so the world is not uniformly too dark.

**Why this priority**: This is a visible, reproducible correctness defect affecting most supported
client eras. It is independent of the frame-pacing work and can ship separately.

**Independent Test**: Load the same map on an Alpha client and on a 1.0.0-or-later client and
confirm each renders at its era-appropriate brightness, with the selected era profile named in the
viewer's diagnostics.

**Acceptance Scenarios**:

1. **Given** a client build with a known era profile, **When** a world loads, **Then** terrain
   lighting uses that era's profile and the viewer reports which profile was selected.
2. **Given** a client build with no matching era profile, **When** a world loads, **Then** the
   viewer flags the build as unprofiled and names the fallback it used, rather than silently
   applying an Alpha model.
3. **Given** a 1.0.0-or-later client, **When** terrain is compared against the same scene in the
   native client, **Then** the brightness difference is within a stated tolerance and the comparison
   evidence records client root, build identity, map, and camera position.
4. **Given** an era profile is selected, **When** the exact-build `Light*` DBC chain is available,
   **Then** DBC values remain authoritative and the era profile does not override them.

---

### User Story 4 - Land renderer fixes that are provably better (Priority: P3)

As the maintainer, I need each renderer change measured before and after on the proven harness and
reverted if it does not help, so the renderer improves monotonically instead of drifting.

**Why this priority**: This is the actual repair work, but it is worthless without Stories 1 and 2.
Sequencing it last is deliberate.

**Independent Test**: Take one candidate change, record before/after hitch statistics from the same
trajectory and client, and confirm the change is accepted or reverted on that evidence alone.

**Acceptance Scenarios**:

1. **Given** a candidate renderer change, **When** it is proposed, **Then** a before measurement
   exists on the same trajectory, client, build, and map.
2. **Given** a landed renderer change, **When** the after measurement does not improve the target
   statistic beyond the stated noise floor, **Then** the change is reverted rather than kept.
3. **Given** multiple candidate changes, **When** they are evaluated, **Then** each is measured
   individually so no improvement or regression is attributed to the wrong change.

---

### User Story 5 - Flatten the scene into ordered draw lists (Priority: P2)

As the maintainer, I need the per-frame render path to consume retained, flat, ordered lists of draw
work grouped into explicit passes, rather than walking a tree of discrete node objects and
allocating per frame, so frame cost becomes predictable and hitching stops.

**Why this priority**: This is the leading hypothesis for the gallop and the core architectural
complaint. It is sequenced after US1/US2 only because it must be measured, not guessed. It does not
require US3.

**Independent Test**: Run the same boundary-crossing trajectory before and after the flattening pass
and compare per-frame allocation volume and hitch statistics. Delivers a measurable reduction in
frame-time variance, or is reverted.

**Acceptance Scenarios**:

1. **Given** a steady-state frame with an unchanged resident tile set, **When** the frame renders,
   **Then** the render path performs no per-frame heap allocation proportional to node or tile count.
2. **Given** the resident tile set changes, **When** draw lists are rebuilt, **Then** the rebuild is
   incremental for the tiles that changed rather than a full rebuild of all resident tiles.
3. **Given** a culled region, **When** it is rejected, **Then** the cost of rejecting it does not
   scale with the size of the rejected subtree.
4. **Given** diagnostic attribution is not requested, **When** a production frame renders, **Then**
   no diagnostic collection or per-kind attribution work is performed.
5. **Given** the flattened path is active, **When** the scene renders, **Then** output is visually
   equivalent to the previous path for the same camera, client, build, and map.
6. **Given** the flattened path regresses frame-time variance against baseline, **When** measured,
   **Then** it is reverted rather than kept.

---

### User Story 6 - Focused view modes instead of one universal scene (Priority: P2)

As a user, I want to choose an exploration mode — terrain, model, WMO, PM4 — and have the viewer
build and render only that mode's scene, so I am not paying the cost of every overlay at once for
information I rarely need simultaneously.

**Why this priority**: It directly reduces per-frame work and is the structural fix for treating the
whole map as discrete pickable objects at all times. It also gives the UI a single organizing
principle.

**Independent Test**: Switch between modes on the same map and confirm each mode constructs only its
own scene content, with per-frame work and draw counts measurably lower than the combined view.

**Acceptance Scenarios**:

1. **Given** a selected view mode, **When** the scene is constructed, **Then** only that mode's
   content is built, traversed, and submitted.
2. **Given** a selected view mode, **When** the user switches modes, **Then** the previous mode's
   scene content is released and the new mode's is constructed without requiring a world reload.
3. **Given** per-object pickability is not required by the active mode, **When** the scene renders,
   **Then** the renderer does not maintain per-object discrete state solely to support picking.
4. **Given** a mode is active, **When** the user needs cross-mode facts, **Then** those remain
   available through the sidebars without requiring the other mode's scene to be rendered.
5. **Given** any mode, **When** measured on the proven harness, **Then** its per-frame work is
   reported so modes can be compared honestly.

---

### User Story 7 - One owner per surface in the UI (Priority: P3)

As a user, I want each piece of information to live in exactly one place, so the sidebars stop
duplicating the same panels across routes.

**Why this priority**: It is the visible symptom of the same structural problem, and it is what the
user still sees after the previous consolidation pass. It is sequenced last because the view-mode
decision in US6 determines the correct organizing structure.

**Independent Test**: Inventory every panel and its routes, then confirm each has exactly one owning
route, with any remaining shared element being a genuine reusable widget rather than a duplicated
panel.

**Acceptance Scenarios**:

1. **Given** the panel inventory, **When** it is reviewed, **Then** every content panel has exactly
   one owning route, and multi-route items are only primitive widgets.
2. **Given** a user looking for a piece of information, **When** they navigate, **Then** there is
   exactly one place it appears.
3. **Given** the UI structure, **When** compared to the active view mode, **Then** the sidebar
   organization follows the mode rather than presenting every surface at all times.
4. **Given** a panel is removed from a duplicate route, **When** the former route is used, **Then**
   the user is routed to the owning location rather than losing access.

---

### Edge Cases

- A trajectory that never crosses a tile boundary — must be reported as such, since it cannot
  exercise the suspected cause.
- A view mode with no content on the current map — must state that plainly rather than rendering an
  empty scene that looks like a load failure.
- Switching view modes mid-flight while tiles are still streaming — must not leak the previous mode's
  residency or double-load.
- A panel whose duplicate routes have subtly different behavior — the differences must be reconciled
  deliberately, not silently resolved by picking one.
- The flattened draw list going stale against a changed resident set — staleness must be impossible
  by construction or detected, never rendered.
- A machine whose background load perturbs frame timing — the stated noise floor must make this
  visible rather than letting it masquerade as a result.
- A client whose `Light*` DBC chain is missing or malformed — era profile selection and DBC fallback
  must remain distinguishable in diagnostics.
- A build that sits between two known eras — must be flagged as unprofiled rather than snapped to
  the nearest era silently.
- A hitch that occurs during warmup only — must not be counted as a steady-state hitch, and warmup
  must be reported separately.
- A trajectory long enough that terrain unloads behind the camera — unload cost must be attributable
  in the same way as load cost.

## Requirements *(mandatory)*

### Functional Requirements

#### Detector capability (must be satisfied before any renderer change)

- **FR-001**: The profiler MUST support camera trajectories that move over time, including at least
  one trajectory that crosses ADT tile boundaries.
- **FR-002**: The profiler MUST record a per-frame wall-clock time series for every measured frame.
- **FR-003**: The profiler MUST report median, maximum, p95, p99, and a count of frames exceeding a
  stated hitch threshold, in addition to any aggregate counters it already reports.
- **FR-004**: The profiler MUST support injecting a synthetic delay of known magnitude at a known
  frame, and MUST correctly flag that frame, so detector power is demonstrated rather than assumed.
- **FR-005**: The profiler MUST report its run-to-run noise floor so that a difference smaller than
  the noise floor cannot be presented as an improvement.
- **FR-006**: The profiler MUST label each run with its trajectory type, and MUST mark stationary
  runs as incapable of demonstrating movement-induced behavior.
- **FR-007**: The profiler MUST record the client root, build identity, map, trajectory, and frame
  counts in every report, so two reports can be compared only when they are comparable.
- **FR-008**: The default measured-frame count MUST be large enough to observe periodic hitches
  across at least one full tile-crossing cycle.

#### Attribution

- **FR-009**: The system MUST classify each flagged hitch by dominant cause among CPU work,
  GPU/driver time, and I/O/streaming stall.
- **FR-010**: The system MUST present the per-stage evidence supporting each attribution.
- **FR-011**: The system MUST state explicitly when a cause cannot be separated with the available
  measurements, rather than reporting an unsupported attribution.

#### Terrain lighting per era

- **FR-012**: Terrain lighting MUST select a profile based on the active client build's era, across
  Alpha 0.5.3 through Cataclysm 4.0.x.
- **FR-013**: The system MUST report which era profile was selected for the active build.
- **FR-014**: The system MUST flag builds with no matching era profile as unprofiled, and MUST name
  the fallback applied, rather than silently applying an Alpha model.
- **FR-015**: Exact-build `Light*` DBC values MUST remain authoritative where available; the era
  profile MUST NOT override them.
- **FR-016**: Era profile selection MUST carry provenance, not just values, so a rendered result can
  be traced to the profile and build that produced it.

#### Flattened render pipeline

- **FR-021**: The per-frame render path MUST consume retained, ordered draw lists grouped into
  explicit passes, rather than walking a node tree per frame.
- **FR-022**: A steady-state frame MUST NOT perform heap allocation proportional to node or tile
  count.
- **FR-023**: Draw-list rebuilds MUST be incremental with respect to residency changes, not full
  rebuilds of the whole resident set.
- **FR-024**: The cost of rejecting a region MUST NOT scale with the size of the rejected subtree.
- **FR-025**: Diagnostic collection and per-kind attribution MUST be opt-in and MUST NOT execute on
  production frames when not requested.
- **FR-026**: The flattened path MUST produce output visually equivalent to the previous path for the
  same camera, client, build, and map.
- **FR-027**: Draw lists MUST NOT be able to go stale against the resident set without detection.

#### Focused view modes

- **FR-028**: The system MUST support selectable exploration modes covering at least terrain, model
  (M2/MDX), WMO, and PM4.
- **FR-029**: Scene construction MUST be scoped to the active mode's content.
- **FR-030**: Switching modes MUST release the previous mode's scene content and construct the new
  mode's without a world reload.
- **FR-031**: Per-object discrete state MUST be maintained only when the active mode requires it.
- **FR-032**: Cross-mode facts MUST remain reachable through sidebars without rendering the other
  mode's scene.
- **FR-033**: Per-frame work MUST be reported per mode so modes can be compared.

#### UI ownership

- **FR-034**: Every content panel MUST have exactly one owning route.
- **FR-035**: Elements shared across routes MUST be primitive widgets, not duplicated content panels.
- **FR-036**: Sidebar organization MUST follow the active view mode.
- **FR-037**: Retiring a duplicate route MUST redirect to the owning location rather than removing
  access.

#### Change discipline

- **FR-017**: Every renderer change MUST have a before and after measurement from the proven harness
  on the same trajectory, client, build, and map.
- **FR-018**: Renderer changes MUST be evaluated one at a time.
- **FR-019**: A change whose measured effect does not exceed the stated noise floor MUST be reverted.
- **FR-020**: No FPS, frame-time, or performance claim may be made from a successful build alone.
- **FR-038**: Every phase MUST be independently revertible, and the viewer MUST remain runnable at
  the end of each phase.

### Key Entities

- **Camera trajectory**: A named, reproducible path through a map over time, including whether and
  where it crosses ADT tile boundaries.
- **Frame sample**: One measured frame's wall-clock time plus its per-stage breakdown and workload
  counters.
- **Hitch**: A frame whose time exceeds the stated threshold, carrying its index, magnitude, and
  dominant-cause attribution.
- **Run report**: A complete measurement with its identity (client root, build, map, trajectory,
  frame counts), its statistics, its noise floor, and its stated limitations.
- **Era lighting profile**: A named terrain lighting model bound to a client-era range, with
  explicit unprofiled handling and provenance.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The harness detects a synthetic injected hitch of known size at the correct frame in
  100% of verification runs, establishing detector power before any renderer change lands.
- **SC-002**: The gallop the user observes interactively is reproduced as a measured hitch pattern in
  an automated run, so the defect is characterized by number rather than description.
- **SC-003**: Every flagged hitch in a published report carries a dominant-cause attribution or an
  explicit statement that the cause could not be separated.
- **SC-004**: Frame-time variation during a boundary-crossing trajectory is reduced against the
  recorded baseline by a margin larger than the stated noise floor, on the same client, build, map,
  and trajectory.
- **SC-005**: Terrain brightness on 1.0.0-or-later clients matches the native client within a stated
  tolerance, with the comparison evidence recorded.
- **SC-006**: 100% of supported client builds either resolve to a named era lighting profile or are
  explicitly reported as unprofiled; none silently receive the Alpha model.
- **SC-007**: Every renderer change landed in this feature has a paired before/after measurement, and
  any change not clearing the noise floor is reverted.
- **SC-008**: A steady-state frame performs zero heap allocation proportional to node or tile count,
  measured, down from per-frame allocation that scales with the resident tile set.
- **SC-009**: The cost of rejecting a region no longer scales with the size of the rejected subtree.
- **SC-010**: Production frames perform no diagnostic attribution work unless diagnostics are
  explicitly requested.
- **SC-011**: Each focused view mode measurably reduces per-frame work against the combined view on
  the same map and camera.
- **SC-012**: Every content panel has exactly one owning route; the count of multiply-routed content
  panels reaches zero, down from a measured 71 multi-route `Draw*` methods of which the panel-level
  cases are the target.
- **SC-013**: The viewer builds and runs at the end of every phase, and each phase can be reverted
  independently.

## Assumptions

- The staged client library provides the eras needed for real measurement. Confirmed present:
  prealpha, Vanilla, TBC, WoW335, 4.0.0.12635, and a Cataclysm beta 11927 build.
- "Gallop" means frame-time variance and periodic hitching, not a low but steady frame rate. Both are
  worth fixing, but hitching is the defect this feature targets first.
- Headless measurement on the maintainer's machine is representative enough to rank causes and detect
  regressions, even though absolute numbers are machine-specific.
- The terrain darkness defect is a lighting-model era gap rather than a texture or asset decode
  defect. This is inferred from the absence of any era parameter in the terrain lighting model and
  must be confirmed by measurement before a fix is designed.
- Interactive visual and FPS confirmation remains maintainer-owned. Automated measurement ranks and
  gates the work; it does not replace the maintainer's sign-off.
- Ghidra analysis of the native client's renderer is a supporting evidence lane, not a porting
  exercise. No original client code is copied.

## Non-Goals

- No FPS or performance claim from build output alone.
- No optimization accepted without before/after measurement from a harness proven able to detect the
  defect.
- Not a port of the original client renderer.
- Not a visual-fidelity overhaul beyond the era lighting correctness described here.
