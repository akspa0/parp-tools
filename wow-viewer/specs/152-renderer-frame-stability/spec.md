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

### Edge Cases

- A trajectory that never crosses a tile boundary — must be reported as such, since it cannot
  exercise the suspected cause.
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

**Detector capability (must be satisfied before any renderer change)**

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

**Attribution**

- **FR-009**: The system MUST classify each flagged hitch by dominant cause among CPU work,
  GPU/driver time, and I/O/streaming stall.
- **FR-010**: The system MUST present the per-stage evidence supporting each attribution.
- **FR-011**: The system MUST state explicitly when a cause cannot be separated with the available
  measurements, rather than reporting an unsupported attribution.

**Terrain lighting per era**

- **FR-012**: Terrain lighting MUST select a profile based on the active client build's era, across
  Alpha 0.5.3 through Cataclysm 4.0.x.
- **FR-013**: The system MUST report which era profile was selected for the active build.
- **FR-014**: The system MUST flag builds with no matching era profile as unprofiled, and MUST name
  the fallback applied, rather than silently applying an Alpha model.
- **FR-015**: Exact-build `Light*` DBC values MUST remain authoritative where available; the era
  profile MUST NOT override them.
- **FR-016**: Era profile selection MUST carry provenance, not just values, so a rendered result can
  be traced to the profile and build that produced it.

**Change discipline**

- **FR-017**: Every renderer change MUST have a before and after measurement from the proven harness
  on the same trajectory, client, build, and map.
- **FR-018**: Renderer changes MUST be evaluated one at a time.
- **FR-019**: A change whose measured effect does not exceed the stated noise floor MUST be reverted.
- **FR-020**: No FPS, frame-time, or performance claim may be made from a successful build alone.

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
