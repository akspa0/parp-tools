# Feature Specification: PM4 Region Navigation and Audio Trigger Controls

**Feature Branch**: `149-pm4-region-audio-controls`

**Created**: 2026-08-14

**Status**: Draft

**Input**: User description: "Replace the PM4 correlation workbench with a list of decoded PM4 regions that can be double-clicked to move the camera to the region, remove matching details from PM4 tooltips, and expose all audio triggers as opt-in UI controls that default off. Include legacy MCNK environmental and liquid-driven audio when MCSE is absent, and normalize MCSE coordinates to their owning tile/chunk before range checks. Defer player-height/game-mode movement to a later feature."

## User Scenarios & Testing

### User Story 1 - Browse and focus decoded PM4 regions (Priority: P1)

When a PM4 overlay is available for the current world session, the user can open a region list and
see the decoded regions that are actually represented by the loaded PM4 data. Each row identifies the
region and gives enough geometry totals and spatial information to distinguish it from neighboring
regions. Double-clicking a row selects that region and moves the exploration camera to a useful view
of its geometry, including regions that span more than one tile.

**Why this priority**: Region identity and decoded surface ownership are now more trustworthy than the
old asset-correlation guesses. This gives the user a direct way to inspect the data that the client
actually supplied and removes the need to hunt through geometry manually.

**Independent Test**: Load a PM4-backed fixture with at least two non-empty regions, open the PM4
region list, and double-click each row. The selected row, PM4 highlight, camera position, and loaded
tile context must identify the requested region without invoking WMO/M2 matching.

**Acceptance Scenarios**:

1. **Given** a loaded PM4 overlay with multiple decoded regions, **when** the user opens the PM4
   region list, **then** every non-empty available region appears once with its region ID, tile count,
   object count, surface count, and a deterministic center or bounds summary.
2. **Given** a region row, **when** the user double-clicks it, **then** the viewer selects that region,
   moves the camera to a bounded view of its decoded geometry, and requests the normal residency update
   for the destination rather than loading the entire map.
3. **Given** a region spans multiple PM4/ADT tiles, **when** it is focused, **then** the camera target is
   derived from the region bounds/center and the region remains identifiable after the residency window
   changes.
4. **Given** a PM4 file is missing, malformed, or has no non-empty regions, **when** the list is opened,
   **then** the UI reports that state and does not move the camera using guessed coordinates.

---

### User Story 2 - Inspect PM4 facts without correlation UI (Priority: P1)

When the user selects PM4 geometry, the workbench and tooltip show decoded PM4 facts such as tile,
region, CK24 identity, surface totals, bounds, and proven grouping fields. The viewer no longer presents
WMO/M2 correlation candidates, saved matches, shape searches, or match scores as part of the PM4
inspection path.

**Why this priority**: Correlation results are a distracting and misleading interaction now that the
viewer can isolate authored regions and object surfaces directly. Removing them prevents a guessed
external asset association from being mistaken for decoded client truth.

**Independent Test**: Select and hover PM4 geometry, inspect the PM4 workbench, and open every PM4
inspection surface. No correlation tab, match button, saved-match row, candidate list, match score, or
correlation wording is presented. The remaining fields are sourced from the selected PM4 object/region.

**Acceptance Scenarios**:

1. **Given** a selected PM4 object, **when** the user opens the PM4 workbench, **then** region
   navigation and decoded object/surface facts are available without a WMO/M2 matching control.
2. **Given** the pointer is over PM4 geometry, **when** the tooltip appears, **then** it contains only
   decoded PM4 identity/geometry/context and contains no matching or correlation details.
3. **Given** an older saved correlation record or an unavailable client asset, **when** the PM4 UI is
   opened, **then** the record does not affect selection, tooltip content, region navigation, or camera
   focus.

---

### User Story 3 - Review and explicitly enable audio triggers (Priority: P1)

When a world is loaded, the audio panel lists every audio trigger in the current bounded trigger set,
including legacy MCNK environmental/water triggers derived from chunk flags and liquid data, resident
MCSE emitters when present, and the active area/ZoneMusic trigger when its metadata is available. Each
row shows its source and diagnostic state and has an enable toggle. Trigger playback is disabled by
default, and no looping emitter, liquid/environment trigger, or area track starts merely because the
camera entered a loaded tile or area. The user can enable one trigger, several triggers, or the master
trigger playback control and can turn them off again without losing diagnostics.

**Why this priority**: Automatic looping playback currently makes map inspection unpleasant and hides
whether a sound is actually proven, resolved, and supported. A visible, opt-in trigger surface makes
silence safe while preserving a path to test each decoded source.

**Independent Test**: Load a client-backed scene containing resident MCSE records and an area audio
assignment, open the audio panel, and verify that all listed trigger controls are off and no source is
active. Enable one row, verify its normal resolution/playback attempt, then disable it and verify that it
stops and remains disabled while the camera moves.

**Acceptance Scenarios**:

1. **Given** a new map/client session, **when** the audio panel opens, **then** every resident MCNK
   environmental/liquid trigger, resident MCSE trigger, and applicable current-area trigger is listed
   with a stable key, source record, coordinates or area context, resolution/provenance state, and an
   off control. A 0.5.3 map with no MCSE records still exposes its decoded MCNK-derived trigger rows.
2. **Given** all trigger controls are off, **when** the camera moves through resident tiles or changes
   area, **then** no MCNK, MCSE, or area-music source starts automatically; diagnostics continue to
   update and identify the intentional disabled state.
3. **Given** one trigger is enabled, **when** its source is resolvable and its backend is available,
   **then** only that trigger may start, with no duplicate source on repeated updates; unsupported or
   unresolved sources remain visible with a specific terminal reason.
4. **Given** an enabled trigger, **when** the user turns it off or disables the master trigger control,
   **then** the source stops promptly, future automatic starts are blocked, and the diagnostic row stays
   inspectable.
5. **Given** a map/client replacement or viewer restart, **when** audio state is initialized, **then**
   trigger enablement returns to off unless a later, explicitly specified persistence feature changes
   that policy.
6. **Given** resident MCSE and/or MCNK audio records, **when** the user enables the 3D sound-emitter
   marker toggle, **then** each finite normalized emitter position is shown as a source-colored pin in
   the world without probing files or starting playback; disabling the toggle submits no marker
   geometry.

### User Story 4 - Inspect resident zone and subzone boundaries (Priority: P2)

When a world is loaded, the user can opt into a 3D area overlay built from resident MCNK chunk
footprints. Resolved zones and subzones are grouped through the same AreaNumber/AreaTable lookup used
by the status bar, rendered with distinct deterministic colors, and labeled in world space. The overlay
is bounded by currently resident terrain and does not invent geometry for missing or unresolved rows.

**Why this priority**: The client exposes area identity per terrain chunk in this build, not a ready-made
boundary polygon. A resident-footprint overlay makes the decoded Zone/SubZone mapping inspectable while
keeping the geometry honest and useful during streaming investigation.

**Independent Test**: Load a client-backed map containing at least two resolved area values, enable the
area overlay, and verify that the 3D footprints, zone/subzone colors, and projected name tags follow
resident tile changes without changing terrain, camera, lighting, or audio behavior.

**Acceptance Scenarios**:

1. **Given** resident chunks with resolvable AreaNumber values, **when** the area overlay is enabled,
   **then** each resolved zone and subzone group appears once with a finite resident footprint, a
   distinct deterministic color, and a world-space label containing its resolved name.
2. **Given** a packed Alpha AreaNumber, **when** the overlay resolves it, **then** it uses the existing
   high-word Zone/low-word SubZone contract and the same map-aware AreaTable path as the status bar.
3. **Given** an unresolved or missing AreaTable row, **when** resident chunks use that value, **then**
   no guessed name or boundary is rendered; the UI reports the unresolved count in the area overlay
   section.
4. **Given** tiles stream in or out, **when** the resident set changes, **then** the overlay refreshes
   from the current resident chunks only and removes stale footprints and labels.
5. **Given** the area overlay is disabled, **when** the scene renders, **then** no area boundary lines,
   pins, or labels are submitted and there is no measurable whole-map scan introduced by the feature.

### Edge Cases

- Region 1 or another decoded empty-stub region may be present; it must be visibly marked or omitted by
  an explicit empty-region policy and must never produce a fake camera target.
- A region may have valid geometry but no current resident tile after streaming changes; focusing it
  must use the normal bounded residency path and report a pending/unavailable target instead of using a
  stale object position.
- A region may have a very large or flat bounding box; camera focus must choose a finite offset and
  avoid placing the camera inside invalid coordinates or beyond the current world limits.
- A PM4 selection may disappear while the panel is open; the region list must refresh without throwing
  and must clear stale selection/highlight state.
- Multiple MCSE records may share a SoundEntries ID or have the same display name; controls must use a
  trigger-instance key, not only the sound ID.
- MCSE coordinates may be local to the owning MCNK rather than global map coordinates; raw values must
  remain visible for diagnosis, but range checks, OpenAL positions, and displayed world positions must
  use a normalized tile/chunk-aware renderer position.
- A 0.5.3 tile may have MCNK flags and liquid data but no MCSE rows; that is a valid legacy audio
  source state, not an empty audio result. Later-build MCSE rows are additive and must not suppress the
  MCNK-derived environmental or water triggers.
- A liquid chunk may expose only a basic family in the current adapter; the trigger model must preserve
  raw liquid type identity when the source provides it and must report an unresolved mapping rather than
  silently selecting an unrelated SoundEntries ID.
- An audio trigger may resolve to no file, an unsupported decoder, a missing DLS/MIDI dependency, or an
  unavailable OpenAL backend; it remains listed and reports that stage without starting a substitute.
- Area music may change between day and night or between a zone and subzone; the row identity and
  diagnostic must follow the resolved packed Zone/SubZone context without restarting while disabled.
- Explicit SoundEntries preview remains a deliberate user action and is not considered automatic trigger
  playback; its loop checkbox must not enable world triggers.
- A resolved area group may contain only a subset of its real-world extent while neighboring tiles are
  not resident; the overlay must describe the resident footprint rather than imply a complete boundary.
- A zone and subzone may share a display name or have duplicate names on different maps; grouping keys
  must include the canonical AreaTable identity and map context, not text alone.

## Requirements

### Functional Requirements

- **FR-001**: The PM4 region browser MUST use the canonical decoded PM4 overlay/region data, label the
  grouping as the proven/current `MSHD` region, and MUST not imply that a region is an external asset
  identity, coordinate-frame key, or WMO/M2 match.
- **FR-002**: The PM4 region browser MUST expose each non-empty available region exactly once with its
  region ID, tile count, object count, surface count, and finite world-space bounds/center when geometry
  is available.
- **FR-003**: The PM4 region browser MUST provide a double-click action that selects the region and
  focuses the camera on its decoded bounds through the viewer's authoritative camera/residency path.
- **FR-004**: Region focus MUST work for regions spanning multiple tiles and MUST remain bounded by the
  existing tile/object streaming policy; it MUST not require whole-map residency.
- **FR-005**: Region selection MUST have a stable empty, loading, unavailable, and selected state and
  MUST clear or refresh stale state when PM4 data or the world session changes.
- **FR-006**: The PM4 workbench MUST replace the correlation interaction with region navigation and
  decoded object/surface inspection; the user-facing Correlation tab/window, WMO/M2 match actions,
  saved-match controls, candidate lists, and match-score summaries MUST be removed or retired from this
  workflow.
- **FR-007**: PM4 tooltips MUST contain only proven decoded PM4 identity, region, surface, bounds, and
  relevant source context; they MUST not contain matching, correlation, candidate, score, or saved-match
  information.
- **FR-008**: Existing PM4 matching/research code MAY remain only when a non-UI caller still requires it,
  but the implementation MUST audit its callers and remove or clearly retire orphaned viewer state,
  persistence, and UI code rather than carrying dead matching controls forward.
- **FR-009**: The audio trigger inspector MUST list every decoded legacy MCNK environmental or
  liquid/water trigger in the current bounded resident set, every MCSE trigger in that set, and every
  applicable active area/ZoneMusic trigger represented by the proven catalog, with a stable
  trigger-instance key and source type. MCNK-derived rows MUST remain available when a 0.5.3 map has no
  MCSE data; later-build MCSE data is additive.
- **FR-010**: Each audio trigger row MUST expose its source record, tile/chunk or packed Zone/SubZone
  context, MCNK flags and liquid identity when applicable, raw/local and normalized world coordinates
  when applicable, resolved ID/path, provenance, decoder/backend state, terminal reason, and current
  enablement state.
- **FR-011**: World-trigger enablement MUST default to off for every trigger on world/session
  initialization, map/client replacement, and runtime reconfiguration; it MUST be separate from master
  gain and mute state.
- **FR-012**: The runtime MUST NOT start MCNK, MCSE, area-music, or other world-trigger playback solely
  because a trigger is resident, in range, or selected by area while its trigger control is off.
- **FR-013**: The UI MUST allow the user to enable or disable individual trigger instances and provide a
  master world-trigger control that can block all automatic trigger starts; disabling a trigger MUST stop
  its active source and prevent it from restarting until explicitly enabled.
- **FR-014**: Disabled and unsupported audio triggers MUST remain visible in diagnostics with explicit
  states such as user-disabled, unresolved, unsupported, backend-unavailable, or active; diagnostics MUST
  not be hidden by the playback opt-in policy.
- **FR-015**: Audio trigger enumeration MUST remain bounded to resident map data plus the applicable
  current area trigger and MUST not load the whole map or client audio catalog merely to populate the
  interactive list.
- **FR-016**: Explicit SoundEntries preview MUST remain a separate deliberate action and MUST NOT change
  world-trigger enablement or cause other trigger instances to start.
- **FR-017**: Existing proven audio decoding, AreaNumber Zone/SubZone resolution, MCSE provenance,
  MCNK flag/liquid decoding, and optional MIDI/DLS capability reporting MUST be reused. Area identity
  resolution MUST branch on the active client layout: Alpha 0.5.x MCNK values use packed
  `AreaNumber` (`high16=Zone`, `low16=SubZone`) with map/continent qualification and
  `ParentAreaNum`, while 3.3.5+ MCNK values use the direct `AreaTable.ID` and `ParentAreaID` path.
  Standard direct IDs MUST NOT be reinterpreted through Alpha `AreaNumber` aliases. MCSE range
  evaluation MUST normalize its local position through the owning tile/chunk coordinate contract before
  comparing against the listener. Liquid/environment sound selection MUST use client-proven mappings or
  remain visibly unresolved; this feature MUST NOT invent a DLS pairing or claim unsupported audible
  playback.
- **FR-018**: Focused automated coverage MUST verify region list identity, finite region focus requests,
  removal of matching fields from PM4 presentation data, default-off trigger state, per-trigger enable/
  disable behavior, master blocking, duplicate-start prevention, and diagnostic visibility.
- **FR-019**: The area overlay MUST derive its records from resident terrain chunk metadata and the
  canonical AreaTable resolver; it MUST NOT scan unloaded tiles or infer polygon boundaries unavailable
  in the client data.
- **FR-020**: The area overlay MUST represent resolved Zone and SubZone groups with stable map-aware keys,
  finite resident bounds, deterministic distinct colors, and one world-space label position per group.
- **FR-021**: The area overlay MUST remain disabled by default, expose a user toggle, refresh when the
  resident chunk set or map changes, and submit no geometry or labels while disabled.
- **FR-022**: Missing or unresolved AreaTable values MUST remain visible as an explicit unresolved count
  or diagnostic state and MUST NOT produce guessed text, colors, or camera targets.
- **FR-023**: The viewer MUST provide an opt-in 3D speaker-marker overlay for every finite resident
  MCSE and MCNK/liquid emitter position, using the normalized renderer position and source-distinct
  colors. Marker rendering MUST use the cached resident snapshot, MUST submit no geometry while
  disabled, and MUST not probe files, enable world triggers, or start playback.

### Key Entities

- **Pm4RegionNavigationItem**: A stable presentation record for one decoded non-empty PM4 region,
  including region ID, tile/object/surface totals, bounds/center, availability, and selection state.
- **Pm4RegionFocusRequest**: A camera navigation request derived from a region's finite decoded bounds,
  carrying the target position, framing offset, and residency context without an external asset match.
- **AudioTriggerInstance**: One resident MCNK environmental/liquid trigger, MCSE emitter, or applicable
  area/ZoneMusic trigger with its source identity, spatial/area context, resolution stages, playback
  state, and stable instance key.
- **AudioTriggerEnablement**: User-owned off/on state for one trigger instance plus the master world-
  trigger gate; it is independent from volume, mute, and explicit preview state.
- **AreaOverlayRegion**: A resolved resident Zone or SubZone group with a map-aware key, display name,
  deterministic color, finite aggregate bounds, label position, and the resident chunk-footprint cells
  that support the visualization.
- **AudioEmitterMarker**: A resident MCSE or MCNK/liquid trigger projected from its normalized renderer
  position into the opt-in 3D speaker overlay, retaining source kind and sound/liquid identity for
  color/inspection without owning playback.

## Success Criteria

### Measurable Outcomes

- **SC-001**: On a PM4 fixture containing multiple non-empty regions, the browser reports exactly one row
  per decoded region and no row depends on a WMO/M2 candidate or correlation score.
- **SC-002**: Double-clicking every listed fixture region produces a finite camera-focus request for that
  region, selects/highlights it, and updates the bounded residency context without whole-map loading.
- **SC-003**: PM4 presentation surfaces contain zero user-facing correlation/matching controls or tooltip
  fields after the cleanup; the remaining PM4 details are traceable to decoded PM4 records.
- **SC-004**: For a fixture/session with N resident MCNK/environmental or liquid triggers, M resident
  MCSE triggers, and an applicable area trigger, the audio panel displays N+M plus the applicable area
  row, and 100% of world-trigger controls are off before explicit user action. A 0.5.3 no-MCSE fixture
  still displays its MCNK-derived rows.
- **SC-005**: With all world-trigger controls off, a controlled camera move through resident MCNK
  liquid/environment triggers, MCSE emitters, and area boundaries produces zero automatic world-trigger
  starts while retaining a terminal diagnostic for every listed trigger. A normalized MCSE coordinate
  is in-range only when its tile/chunk-aware position is in-range, not when its raw local coordinate is
  compared with the global listener position.
- **SC-006**: Enabling one supported trigger starts at most one source for that instance, disabling it
  stops it, and repeated updates do not create duplicate sources; unsupported/unresolved rows remain
  inspectable with a specific reason.
- **SC-007**: A map/client replacement restores the default-off world-trigger state and leaves explicit
  preview behavior independent.
- **SC-008**: Focused tests and a Debug build pass before any claim of viewer correctness; live visual,
  audible, and real-client streaming proof remains a separate user-run gate with the configured client
  root and build recorded.
- **SC-009**: On a resident fixture with multiple AreaTable values, enabling the overlay produces one
  finite labeled region per resolved Zone/SubZone group, with different Zone/SubZone styling, and a
  tile unload removes its cells without retaining stale labels. Disabled overlay submission is zero.
- **SC-010**: On a resident fixture with MCSE and/or MCNK/liquid records, enabling speaker markers
  submits one finite source-colored 3D pin per resident emitter from the normalized world position;
  with the toggle disabled, marker submission is zero and OpenAL/file-probe state is unchanged.

## Assumptions

- The existing decoded PM4 overlay and `MSHD.Field04` grouping are the source of truth for the region
  browser. The UI labels this as an MSHD region and does not promote it to an external asset identity or
  coordinate-frame claim; this feature does not reopen the PM4 binary decode or Ghidra investigation.
- The interactive region list covers the current PM4 session/residency data. A global all-client PM4
  catalog or whole-map scan is not required for the first implementation.
- `WorldScene` remains the owner of decoded PM4 selection/residency seams, while `ViewerApp` owns the
  ImGui list and authoritative camera mutation.
- The existing `WorldAudioRuntime` remains the playback/provenance owner; the UI adds enablement state
  and inspection but does not move decoder or client-file ownership into ImGui. Terrain loading remains
  the producer for MCNK flag/liquid trigger candidates, while the runtime owns normalization, mapping
  diagnostics, range checks, and playback policy.
- Alpha 0.5.3 MCNK flags and inline liquid data are a first-class legacy audio input. MCSE is not a
  required prerequisite for environmental or water trigger enumeration, and later MCSE data is additive
  rather than an override.
- Existing WAV/OGG/MP3 decoding and capability diagnostics remain unchanged. MIDI/DLS pairing and native
  MCSE callback installation remain explicit unsupported/unproven gates.
- The feature is additive to working rendering and audio diagnostics until the replacement UI is proven;
  no renderer rewrite or client-archive mutation is implied.
- The area overlay uses chunk footprints as the available 0.5.3 spatial evidence. It is an inspection
  aid for resident coverage, not a claim that MCNK records contain authoritative polygon boundaries.
- Player-height camera movement, walking/running speed, jumping, collision-aware game mode, and player
  actor controls are explicitly deferred to a later feature and are not acceptance criteria here.

## Out of Scope

- Implementing a player/game-mode camera, player-height calibration, walking/running movement, jumping,
  collision, or a simulated player actor.
- Re-decoding PM4 chunks, rewriting the global renderer coordinate system, or using the UI cleanup to
  assert that all PM4 research questions are solved. The audio path may add a tile/chunk-local MCSE
  normalization step, but it must reuse and test the established renderer axis convention.
- Deleting shared PM4 matching/research libraries solely because their old workbench was removed; they
  are retired only after the caller audit in FR-008 proves they are orphaned.
- Implementing MIDI/DLS playback, choosing a new audio backend, guessing sound-entry-to-file mappings,
  or adding proprietary client assets to the repository.
- Loading every PM4 tile or every client sound entry just to populate an interactive panel.
- Reconstructing complete zone polygons from unloaded terrain, inventing a new AreaTable hierarchy, or
  coupling area labels to audio playback.
