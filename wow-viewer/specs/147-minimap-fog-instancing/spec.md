# Feature Specification: Minimap Interaction, Fog-Bounded Residency, and Doodad Instancing

**Feature Branch**: `147-minimap-fog-instancing`

**Created**: 2026-08-14

**Status**: Draft

**Input**: User description: "Repair full-screen minimap dragging and triple-click camera teleport, use the active fogEnd as the radius that determines which ADT tiles are loaded and drawn, and improve asset instancing/batching because doodads remain the primary rendering-performance problem."

## Why This Slice Exists

The viewer has three user-visible failures at the boundary between interaction, streaming, and
submission:

- The full-screen minimap is no longer a dependable navigation surface. Dragging does not
  reliably pan it, and a triple click does not reliably teleport the camera.
- The terrain streaming policy receives the active fog range but currently treats fog as a render
  effect instead of the authoritative near-world coverage signal. This permits the residency and
  draw sets to disagree with what the user can actually see.
- Doodads remain the dominant dense-scene cost. Some model renderers already have batch paths, but
  the runtime still needs a stable contract for shared asset geometry, compatible material/pass
  buckets, placement transforms, and diagnostics.

This spec composes the existing minimap/teleport, world-scene-graph, and M2-doodad work rather than
replacing them. It does not reopen format readers, WDL horizon work, lighting reconstruction, or
the broader UI overhaul.

## User Scenarios & Testing

### User Story 1 - Full-Screen Minimap Navigation (Priority: P1)

As a viewer user, I want the full-screen minimap to behave like a map navigation surface: I can
drag it to pan, and I can triple-click a map location to move the camera there.

**Why this priority**: The full-screen map is currently useful as an image but not as an
interactive world-navigation tool. This blocks camera planning and makes the existing teleport
workflow appear broken.

**Independent Test**: Open the full-screen minimap on a loaded map, drag across the map, release,
then triple-click the same visible map location without moving the pointer between clicks. Verify
that panning changes the map view and that the third click changes the camera tile and world
position.

**Acceptance Scenarios**:

1. **Given** the full-screen minimap is open and the pointer is over the map surface, **when** the
   user presses and drags the left mouse button, **then** the map pans continuously, the gesture is
   consumed by the minimap, and the camera is not teleported.
2. **Given** the user has completed a drag, **when** the left button is released, **then** the
   gesture is classified as a pan rather than a click and any pending teleport sequence is reset.
3. **Given** the pointer is over a valid map location, **when** the user performs three left clicks
   on the same map tile within the confirmation window without dragging, **then** the camera moves
   to that map location on the third click and the active tile/residency request is refreshed.
4. **Given** the user clicks different map tiles or waits beyond the confirmation window, **when**
   another click occurs, **then** the confirmation count restarts for the new tile and no teleport
   occurs prematurely.
5. **Given** the map surface has no loaded tile texture or is partially outside the map bounds,
   **when** the user clicks it, **then** the interaction remains stable and does not produce an
   invalid camera coordinate.

### User Story 2 - Fog-Bounded World Coverage (Priority: P1)

As a viewer user, I want detailed ADT terrain and normal world objects to be resident and drawable
only within the effective fog range around the camera, so nearby content remains stable while
far-away content does not consume CPU, I/O, or GPU work.

**Why this priority**: The renderer can only be responsive if its data set matches the visible
world. The active `fogEnd` already comes from the lighting system and is the correct client-facing
signal for the radius at which detailed content stops contributing to the frame.

**Independent Test**: Load a real or deterministic world scene, record the active fog start/end,
move the camera across a tile boundary, and change the active lighting/fog profile. Compare the
selected, retained, resident, and submitted tile sets against the fog-bounded coverage window.

**Acceptance Scenarios**:

1. **Given** a valid active `fogEnd`, **when** streaming targets are computed, **then** tile
   admission is derived from that distance in renderer world units and includes every tile whose
   near bounds can contribute inside the effective fog window, including the camera tile and the
   required near-field safety neighbors.
2. **Given** a tile is outside the effective fog window and is not protected by an explicit camera
   path preload or diagnostic full-load mode, **when** a normal frame is prepared, **then** its
   detailed ADT terrain, normal tile-owned WMO placements, and normal tile-owned doodads are not
   decoded, uploaded, traversed, or submitted for that frame.
3. **Given** a tile remains inside the fog window while the camera turns or crosses a nearby tile
   boundary, **when** subsequent frames are prepared, **then** that tile is not evicted merely
   because it moved outside a directional cone or because another tile became the forward tile.
4. **Given** the active fog profile changes, **when** the next streaming decision is made, **then**
   the effective coverage window and its diagnostics update without requiring a full-map scan or
   causing rapid load/unload oscillation.
5. **Given** a camera-path preload lease exists, **when** a path sample lies outside the current
   camera-centered fog window, **then** the lease may retain its explicitly named tiles, but the
   diagnostics identify them as preloaded exceptions rather than ordinary visibility.
6. **Given** `fogEnd` is missing, invalid, or non-positive, **when** normal streaming is evaluated,
   **then** the existing safe active-lighting fallback is used and the viewer fails closed to a
   bounded window; it must not interpret invalid fog as permission to load the whole map.

### User Story 3 - Shared Doodad Assets and Compatible Instance Batches (Priority: P1)

As a viewer user, I want repeated doodads to share immutable geometry and material state while
their placements contribute only transforms and placement-local state, so dense forests, clutter,
and WMO doodad sets do not multiply the same CPU and GPU work by instance count.

**Why this priority**: Doodads remain the largest dense-scene performance cost. Improving tile
selection without improving repeated-object submission only moves the bottleneck.

**Independent Test**: Render a deterministic scene containing many placements of the same static
opaque doodad plus animated, alpha-tested, transparent, particle, and ribbon variants. Inspect
asset-load, animation-update, batch, instance, and draw-call counts and compare them with the
current path while checking visible identities.

**Acceptance Scenarios**:

1. **Given** multiple placements reference the same compatible static doodad asset, **when** the
   active tile set is prepared, **then** the asset geometry is loaded/uploaded once and the
   placements are represented by a shared batch with per-instance transforms.
2. **Given** visible placements share an asset but differ in material/pass requirements, fade
   state, animation state, or effect features, **when** batches are prepared, **then** they are
   split into deterministic compatible buckets rather than forcing unsafe state sharing or
   producing incorrect visuals.
3. **Given** a doodad uses transparency, particles, ribbons, animated bones, or another
   unsupported instance feature, **when** it is submitted, **then** it remains on a named
   correctness fallback and is counted separately from static instanced batches.
4. **Given** the same animated asset appears at many placements, **when** the frame advances model
   state, **then** shared animation state is updated according to the asset's declared semantics,
   not once per placement unless placement-local state requires it.
5. **Given** a tile leaves the fog window, **when** its residency is released, **then** its instance
   transforms and placement references are removed without destroying an immutable asset buffer
   still used by another resident tile.
6. **Given** a WMO is placed more than once, **when** its internal doodad sets are submitted,
   **then** the shared WMO/doodad asset state is reused while placement-local transforms, group
   visibility, animation, and transparent ordering remain correct.

### User Story 4 - Truthful Runtime Diagnostics (Priority: P2)

As a viewer developer, I want the renderer to expose why content is resident, visible, batched, or
on a fallback path, so that a capture can distinguish a streaming failure from a culling failure
or a submission bottleneck.

**Why this priority**: The recent popping and performance regressions cannot be diagnosed from one
combined tile or FPS number. Diagnostics are required before user-run capture evidence can close
the slice.

**Independent Test**: Run the same fixed camera sequence before and after each phase and inspect a
structured per-frame report containing fog values, tile sets, object counts, batch keys, and draw
submissions.

**Acceptance Scenarios**:

1. **Given** a rendered frame, **when** diagnostics are collected, **then** they report active
   `fogEnd`, effective fog coverage radius, selected/detail tiles, retained tiles, capture-preload
   tiles, resident tiles, submitted tiles, and excluded tiles with reasons.
2. **Given** a frame with doodads, **when** diagnostics are collected, **then** they report unique
   asset count, compatible batch count, instance count, fallback instance count, animation-update
   count, and draw submissions separately for normal MDX/M2 doodads and WMO-internal doodads.
3. **Given** an invariant is violated, such as a visible near tile being evicted or a batch mixing
   incompatible material state, **when** the frame completes, **then** a named diagnostic is
   emitted with the tile, asset, and admission/submission reason.
4. **Given** a real-client capture is handed off, **when** the user reviews its report, **then** it
   is possible to attribute frame time to tile streaming, visibility, model preparation, batching,
   fallback submission, and GPU/driver wait without inferring one stage from another.

## Edge Cases

- A minimap drag crosses the map edge, the fullscreen overlay is resized, or ImGui reports the
  pointer outside the invisible interaction button while the button is still active.
- A click lands on a minimap tile that has no BLP, no ADT, or no currently resident world tile.
- Fog changes every frame because a local lighting profile is changing or the user is editing a
  fog override; hysteresis must prevent thrashing without retaining content beyond the declared
  effective window indefinitely.
- The camera is inside a WMO whose root bounds are broad or whose group bounds are local; the WMO
  and its visible groups must not be evicted solely because the terrain tile selector changed.
- A tile lies at the 0/63 map edge, a phase overlay supplies only part of the tile, or a tile has
  no detailed ADT while WDL data exists.
- The same doodad asset is referenced by several tiles, or an asset changes from loading to ready
  while a batch is being prepared.
- A model has transparent, particle, ribbon, animated, or effect materials that cannot safely use
  the static instance buffer.
- Full-load diagnostics and camera-path preload are explicitly enabled; they must remain visible
  as named exceptions and must not redefine normal fog-bounded behavior.

## Requirements

### Functional Requirements

- **FR-001**: The full-screen minimap MUST provide a stable, exclusive interaction surface for
  left-button drag, release, and click classification; the parent fullscreen window MUST NOT
  swallow or reinterpret those events.
- **FR-002**: A completed drag MUST pan the minimap and MUST NOT count toward teleport confirmation.
- **FR-003**: Full-screen minimap teleport MUST use the armed three-click contract: three clicks on
  the same valid map target within the confirmation window execute one camera teleport, while
  target changes, timeout, or dragging reset the sequence.
- **FR-004**: A successful minimap teleport MUST convert the selected map location through the
  existing WoW map-coordinate contract, update the camera tile/residency request, and expose a
  status message or diagnostic event.
- **FR-005**: Normal detailed tile admission MUST use the effective active `fogEnd` in renderer
  world units as its coverage boundary, with tile bounds intersection and near-field protection
  preventing nearby terrain from popping out at tile seams.
- **FR-006**: Normal detailed tile/object submission MUST NOT expand radially to the whole map or
  retain tiles outside the effective fog window unless an explicit named preload or diagnostic
  mode protects them.
- **FR-007**: Fog-driven streaming MUST preserve separately reported selected, retained, resident,
  drawable, and preloaded tile states, and MUST apply stable hysteresis or equivalent protection
  against rapid oscillation when the camera or fog changes.
- **FR-008**: The effective fog range MUST follow the existing lighting source hierarchy, including
  active LIT/DBC/global/user override behavior, without introducing a second hardcoded fog truth.
- **FR-009**: Repeated compatible doodad placements MUST share immutable asset geometry/material
  resources and submit placement transforms through a compatible instance batch.
- **FR-010**: Doodad batch compatibility MUST account for render pass, material/texture state,
  alpha/transparency, fade, animation, and effect requirements; incompatible placements MUST use a
  deterministic fallback rather than being merged unsafely.
- **FR-011**: Asset preparation and per-frame animation work MUST be deduplicated by the declared
  asset/batch contract where placement-local state does not require duplication.
- **FR-012**: WMO-internal doodads MUST reuse shared asset state while preserving placement-local
  transforms, group visibility, portal/interior behavior, animation, and transparent ordering.
- **FR-013**: The viewer MUST report fog, residency, visibility, batching, fallback, and draw-call
  counters at a granularity sufficient to explain a near-field pop or a doodad frame-time spike.
- **FR-014**: Existing ADT, WMO, MDX/M2, capture-preload, full-load diagnostic, lighting, and
  minimap texture routes MUST remain available; this slice MUST NOT duplicate or rewrite format
  readers.
- **FR-015**: The feature MUST have focused deterministic tests for minimap interaction state,
  fog-window tile selection, batch compatibility/grouping, and exception/lease accounting before
  real-client proof is requested.

## Key Entities

- **Minimap Interaction State**: The active surface, pointer gesture state, pan offset, click target,
  confirmation count, and timeout used to classify drag versus triple-click teleport.
- **Effective Fog Coverage Window**: The active fog source, fog start/end, world-space radius,
  tile-boundary policy, hysteresis state, and named exceptions used for normal streaming.
- **Tile Residency Record**: A tile's indexed, decoded, GPU-ready, retained, drawable, preloaded,
  and eviction states plus the reason for each transition.
- **Doodad Asset Batch**: An immutable asset resource plus its deterministic compatibility key,
  placement transforms, instance count, fallback reason, and submission counters.
- **Capture Residency Lease**: An explicit camera-path or diagnostic protection record that may keep
  content resident outside the normal fog window without changing normal admission policy.
- **Frame Residency Diagnostics**: The per-frame report joining fog values, tile states, object
  visibility, unique assets, compatible batches, fallbacks, draw submissions, and stage timings.

## Success Criteria

### Measurable Outcomes

- **SC-001**: In a focused interaction test, a fullscreen drag changes the persisted pan offset and
  never changes camera position; a valid same-target triple click teleports exactly once on click
  three, with no premature teleport on clicks one or two.
- **SC-002**: In a fixed world fixture, every normal detailed tile whose bounds can contribute
  inside the effective fog window is admitted, and no tile outside that window is admitted without
  a named preload/full-load exception.
- **SC-003**: Changing the active fog end produces a corresponding bounded residency-window change
  in the next streaming decision, with no unbounded map-wide decode/upload work and no repeated
  unload/reload loop during a stable camera/fog sample.
- **SC-004**: In a dense repeated-doodad fixture, identical compatible assets have one immutable
  geometry resource per asset and compatible placements are submitted through grouped batches;
  the report exposes fewer submission groups than placements without losing visible identities.
- **SC-005**: A fixed real-client camera movement/capture report can separately attribute tile
  streaming, object visibility, doodad batch preparation, fallback submission, and GPU/driver wait;
  no FPS or visual-parity claim is considered closed from source tests alone.
- **SC-006**: Focused tests and the viewer solution build pass, and the real-client proof handoff
  identifies the exact configured client root/build, map, camera path, resolution, warm-up policy,
  and enabled rendering controls.

## Assumptions

- The effective `fogEnd` is the value already resolved by `WorldScene`'s lighting pipeline. A user
  override changes that effective value through the existing owner; it does not create a second
  hardcoded source.
- Fog-bounded admission is a normal camera-driven policy. Explicit capture-path preload and
  diagnostic full-load remain opt-in, named exceptions.
- Tile intersection is evaluated conservatively against tile bounds, not only tile centers, so a
  nearby tile cannot disappear when the camera is close to its edge.
- Static opaque/alpha-tested doodads are the first batching target. Transparent, animated,
  particle, ribbon, effect, and unsupported material paths retain correctness fallbacks until their
  contracts are proven.
- Existing renderer coordinate transforms, minimap coordinate conversion, shared data-source
  readers, and format/profile adapters remain authoritative.
- Real-client visual, FPS, capture, and GPU proof is user-owned after focused source/build checks.

## Out of Scope

- Reconstructing WDL/WL* horizon rendering, skybox, sun/moon, stars, or synthesized minimap data.
- Rebuilding BLS shaders, local WMO lighting, audio playback, or camera-path import.
- Rewriting ADT/WMO/M2/MDX/DBC/LIT readers or changing Alpha versus Standard terrain ownership.
- Replacing the existing scene graph with a new whole-map architecture in this slice.
- Claiming a target FPS, visual parity, or real-client correctness before the user-run capture gate.

## Branch/Workspace Exception

Spec Kit's branch helper attempted to create `147-minimap-fog-instancing` but the shared workspace
returned `Permission denied` while creating `.git/index.lock`. The specification is therefore being
authored on the existing `142-world-scene-graph` branch with the unrelated user-owned
`wow-viewer/imgui.ini` change preserved. No production code is changed by this planning slice.
