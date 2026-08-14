# Feature Specification: Artifact World Simulator Runtime

**Feature Branch**: `148-world-simulator`

**Created**: 2026-08-14

**Status**: Draft

**Input**: Establish a provenance-first local runtime for exploring first-decade World of Warcraft
client artifacts, beginning with spatial audio diagnostics, an explicit camera actor, fog-bounded
residency, and measurable renderer performance.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Explain Every Spatial Audio Decision (Priority: P1)

When a user is over a loaded terrain tile or inside a WMO, they can inspect every relevant MCSE
emitter and area-music trigger. The inspector shows the source record, tile/chunk ownership, raw
coordinates, converted world coordinates, distance/range admission, SoundEntries resolution,
virtual resource path, data-source provenance, decoder result, and playback result. Silence is
therefore an actionable state rather than an empty screen.

**Why this priority**: Audio currently has several independent failure gates and no way to tell
which gate failed. This is the smallest useful step toward audible playback and prevents guesses
about MCSE coordinate layouts or archive contents.

**Independent Test**: Open a client-backed map with MCSE data, select the audio inspection surface,
and verify that each current-tile emitter receives a deterministic status even when playback is
unavailable. Resolve one known WAV/OGG/MP3 entry and verify its path, source, decode state, and
OpenAL state independently.

**Acceptance Scenarios**:

1. **Given** a loaded tile with MCSE records, **When** the user opens the audio inspector, **Then**
   every record on the active tile is listed with tile, chunk, IDs, raw XYZ, world XYZ, range, and
   a non-empty resolution/playback status.
2. **Given** a record whose SoundEntries row resolves to a resource, **When** the resource is
   requested, **Then** the UI distinguishes archive/loose-file discovery, byte read, decoder, and
   OpenAL failures instead of reporting only “not playing”.
3. **Given** an area with DBC-backed zone music or ambience metadata, **When** the camera actor
   enters or leaves the area, **Then** the selected trigger and transition decision are visible
   and the runtime does not silently substitute a guessed file or hard-coded mapping.
4. **Given** MIDI/DLS data without a supported playback backend, **When** it is selected, **Then**
   the UI reports the exact unsupported pair and keeps the trigger inspectable for a later backend.

### User Story 2 - Treat the Camera as a World Actor (Priority: P1)

The exploration camera is an explicit world actor with a position, orientation, roll, area/WMO
context, collision state, residency lease, and audio-listener state. Camera input, path playback,
audio spatialization, collision, tile selection, and diagnostics consume the same authoritative
actor transform rather than maintaining separate observer-only coordinates.

**Why this priority**: The viewer has accumulated parallel camera, listener, path, and streaming
state. A shared actor contract makes the tool behave like a local client session and removes a
class of coordinate and lifecycle mismatches. A visible camera model is optional; spawning extra
geometry is not assumed to improve performance.

**Independent Test**: Move the camera manually and play a camera path while observing a diagnostic
snapshot. The actor transform, audio listener, active tile, collision context, and path sample must
agree at the same timestamp.

**Acceptance Scenarios**:

1. **Given** a loaded world, **When** WASD, mouse, roll, or path playback changes the camera,
   **Then** one actor transform is published to rendering, audio, collision, and residency.
2. **Given** the actor is inside a WMO, **When** the actor moves between WMO groups or terrain,
   **Then** the diagnostic context identifies the current area/WMO state without requiring a
   separate observer-only query.
3. **Given** a path playback lease, **When** playback is stopped or the path changes, **Then** the
   actor and its residency/audio leases are released or retained according to an explicit policy.

### User Story 3 - Stream and Batch the Artifact World (Priority: P1)

The runtime loads terrain, WDL horizon data, WMO placements, and doodads according to the actor's
fog-visible coverage, active path warmup, and explicit inspection needs. It does not load the whole
map merely because a map was selected. Static resources are prepared once, and per-frame work is
reported as unique assets, instance submissions, WMO-internal doodad submissions, terrain work,
audio work, and total draw calls.

**Why this priority**: Near-field popping and Stormwind-scale doodad cost are the main remaining
renderer failures. Selection, residency, preparation, batching, and drawing must be measurable
separately before further optimization can be trusted.

**Independent Test**: Run a fixed camera capture over a dense WMO and a terrain-heavy area with a
known fog end. Record resident tiles, object residency, unique model counts, batch submissions,
draw calls, and stage timings while moving and while idle.

**Acceptance Scenarios**:

1. **Given** a map with hundreds of tiles, **When** the actor is placed at one tile, **Then** only
   tiles justified by fog coverage, path warmup, or an explicit inspection lease become resident.
2. **Given** a dense WMO with repeated doodads, **When** the actor enters or looks across it,
   **Then** the runtime reports shared asset preparation separately from instance submission and
   does not traverse a nested scene graph as the per-frame work unit.
3. **Given** a tile or object is within the actor's effective visible range, **When** the camera
   rotates or moves without crossing the range boundary, **Then** it does not flash out because of
   a reversed FOV, stale tile center, or premature lease release.
4. **Given** a camera path is warmed, **When** playback begins, **Then** the path's swept coverage
   remains resident for the configured hold interval and the capture log identifies any missed
   resource before the frame is recorded.

### User Story 4 - Explore Local Client Artifacts as a Museum Session (Priority: P2)

The user can open a configured client data root and explore its maps, models, lighting, sound, and
camera artifacts as a local session without requiring a Blizzard executable, server, or checked-in
client assets. The session can expose future seams for NPC/game-object data and CPU-local creative
tools, but those future systems are not silently introduced as part of the renderer/audio fix.

**Why this priority**: This defines the product direction without conflating a useful artifact
explorer with a complete game-server implementation.

**Independent Test**: Start a session from a user-selected client root, load a map, inspect its
artifacts, and close the session. The session records the client build/root identity and remains
usable when optional tables or media formats are absent.

**Acceptance Scenarios**:

1. **Given** a supported client root, **When** a map is opened, **Then** all discovered data sources
   identify their build and provenance and no proprietary client data is added to the repository.
2. **Given** optional DBC, LIT, soundbank, or camera resources are absent or malformed, **When**
   the session loads, **Then** the affected capability is disabled with an actionable diagnostic
   while unrelated terrain/model inspection continues.
3. **Given** a future simulator integration, **When** it is proposed, **Then** it has an explicit
   runtime contract and does not replace the working viewer path without an operator-approved
   migration.

### Edge Cases

- An audio file may appear in an MPQ catalog but fail to read from the requested virtual path; the
  diagnostic must distinguish catalog discovery from a successful byte read.
- An era's MCSE coordinate layout may be uncertain. The system must display raw and transformed
  values and identify the transform profile; it must not silently apply a guessed tile/chunk offset.
- A WMO may contain the actor while its placement is not resident. The actor context must report
  the missing residency reason and avoid dereferencing stale GPU data.
- Fog coverage, path warmup, and explicit inspection leases may disagree. The runtime must retain
  the union of justified leases and show which owner keeps each tile/object resident.
- OpenAL may be unavailable. Diagnostics and decode tests must remain usable, while live playback
  is marked unavailable rather than crashing during finalization.
- An audio trigger may have no SoundEntries row, multiple candidate paths, or an unsupported format.
  The status must distinguish unresolved, ambiguous, unsupported, and decoded.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The runtime MUST expose an inspectable record for every MCSE emitter and active area
  music/ambience trigger relevant to the actor's current residency set.
- **FR-002**: Each audio diagnostic MUST preserve source build, map/tile/chunk ownership, source
  record IDs, raw coordinates, transformed coordinates, coordinate-space/profile name, distance
  inputs, and range-admission result.
- **FR-003**: The audio path MUST report SoundEntries row resolution, candidate virtual paths,
  data-source provenance, byte-read result, format detection, decoder result, OpenAL/backend
  result, and the final playback state as separate stages.
- **FR-004**: The runtime MUST support the existing decodable WAV, OGG, and MP3 paths without
  silently treating MIDI/DLS as another compressed audio format.
- **FR-005**: The runtime MUST represent MIDI/DLS requirements as explicit backend capabilities and
  diagnostics, including the soundbank dependency, so a later backend can be added without
  rewriting MCSE or area-trigger ownership.
- **FR-006**: The camera MUST have one authoritative world transform and actor context consumed by
  rendering, audio, collision, path playback, and tile/object residency.
- **FR-007**: Camera path playback MUST publish its swept residency requirements and hold/release
  policy, including enough information to diagnose a missed preload before capture.
- **FR-008**: Tile and object residency MUST be selected from actor position, effective fog range,
  path warmup, and explicit inspection leases; selecting a map MUST NOT imply loading every tile.
- **FR-009**: The renderer MUST separate resource preparation, selection/culling, instance/batch
  preparation, and draw submission in its counters and timing records.
- **FR-010**: The renderer MUST report unique model preparation, instance counts, WMO-internal
  doodad counts, terrain submissions, audio work, and total draw calls per frame or capture sample.
- **FR-011**: Optional or malformed client resources MUST fail closed with a visible diagnostic and
  MUST NOT prevent unrelated map inspection from continuing.
- **FR-012**: Client roots, build identities, schema profiles, coordinate transforms, and external
  research references MUST remain runtime/configuration or documentation data; proprietary client
  assets MUST NOT be committed.
- **FR-013**: New simulator behavior MUST be additive and reversible. Existing working viewer paths
  MUST remain the default until an operator-approved validation promotes a replacement.
- **FR-014**: The implementation MUST include focused automated coverage for coordinate reporting,
  audio-stage classification, lease ownership, and batch/metric attribution before user-run
  visual, audible, or performance proof is claimed.

### Key Entities

- **WorldSession**: A local exploration session bound to a user-selected client root, build, map,
  active actor, optional path, and diagnostic/provenance context.
- **CameraActor**: The authoritative position, orientation, roll, area/WMO context, collision
  context, audio-listener state, and residency leases for the exploring user.
- **AudioTriggerDiagnostic**: An MCSE, area music, or ambience trigger plus raw/transformed
  coordinates, resolution stages, range admission, and playback state.
- **ResidencyLease**: A named request to retain a tile, object, or resource, with owner, reason,
  coverage, start, and release information.
- **RenderPerformanceSample**: A timestamped attribution of selection, preparation, batching,
  submission, draw, audio, resident counts, and frame timing.
- **ArtifactProvenance**: The client build/root, virtual path, source archive or loose-file class,
  schema/profile, coordinate transform, and evidence status for a decoded artifact.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a fixture containing MCSE and area-trigger records, 100% of relevant records
  appear in diagnostics with a non-empty terminal state, including intentional unsupported/error
  states.
- **SC-002**: For every supported decoded audio fixture, the diagnostic identifies the exact
  virtual path, source class, byte-read result, decoder result, and backend result without requiring
  visual inspection of logs.
- **SC-003**: During a controlled camera move or path sample, actor, listener, active tile, area,
  and residency diagnostics report the same world position and timestamp.
- **SC-004**: A controlled map load does not make all known map tiles resident; every resident tile
  has at least one visible lease reason (fog coverage, path warmup, or explicit inspection).
- **SC-005**: A dense-WMO benchmark records separate internal-doodad preparation/submission costs
  and total draw-call attribution, making a before/after optimization comparison reproducible.
- **SC-006**: A user can inspect a client artifact session when audio backend, optional tables, or
  one malformed resource is unavailable, without the viewer crashing or losing unrelated map data.
- **SC-007**: The first implementation phase passes focused automated tests and a Debug build; live
  audible playback, visual correctness, and FPS improvement remain explicitly user-run proof gates.

## Assumptions

- The target is the first decade of client data, with era-specific readers and coordinate/schema
  profiles preserved rather than flattened into one guessed format.
- Client data is supplied by the user at runtime. No Blizzard executable, client archive, extracted
  asset, or proprietary soundbank is added to the repository.
- Existing MPQ, DBC, ADT, WDT, WDL, WMO, M2/MDX, LIT, and audio readers remain the owners; this
  feature adds contracts and diagnostics instead of duplicating readers.
- OpenAL is optional at test time. Decode/provenance diagnostics must be testable without an audio
  device or native OpenAL library.
- MIDI/DLS playback may require a platform-specific or separately selected backend; until one is
  proven, the runtime must expose the missing capability rather than synthesize or silently skip it.
- A visible camera model is not required for the actor contract. The actor's shared transform and
  lifecycle are the goal; rendering an additional camera asset is an optional later experiment.
- A complete MMO server, NPC simulation, network protocol, or local LLM game runtime is future
  scope. This epic defines seams for those directions but does not claim to implement them.
- External projects may inform architecture comparisons, but their code and assets are not copied
  into the viewer without a separate license and compatibility review.
