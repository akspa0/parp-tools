# Feature Specification: World Audio and Camera Playback

**Feature Branch**: `146-audio-camera-playback`
**Created**: 2026-08-12
**Status**: Draft
**Input**: Add client-linked audio playback to the viewer so built-in and authored camera paths can be previewed and captured with their intended sound, while building toward a complete world-audio system for the eventual single-player client experience.

## User Scenarios & Testing

### User Story 1 - Hear camera tracks with their world audio (Priority: P1)

As a viewer user, I want to play an imported client camera or an authored camera path with the audio associated with that scene, so FlyBys and demo videos feel like the original game presentation.

**Why this priority**: Camera playback is already a working viewer workflow. Audio is the next major missing part of making that workflow a faithful client-like experience.

**Independent Test**: Load a client, import a built-in camera path with a known audio association, press Play, and verify that the camera and audio begin from the same timeline origin and stop together.

**Acceptance Scenarios**:

1. **Given** a camera path has a resolvable audio binding, **when** the user presses Play, **then** the camera and audio start from time zero or the selected playhead time together.
2. **Given** the camera path is paused, scrubbed, looped, or stopped, **when** the transport changes state, **then** audio follows the same state without accumulating a second playback instance.
3. **Given** no client audio binding is available, **when** the user plays the path, **then** the camera still plays and the UI reports why audio is unavailable.

### User Story 2 - Reproduce area ambience and positional emitters (Priority: P1)

As a viewer user, I want terrain area ambience and client sound emitters to contribute to the current world audio, so standing in a zone or moving through a scene feels like being inside the game world.

**Why this priority**: The repository already reads area MIDI ambience metadata and MCSE sound-emitter records. Playback needs to connect those existing data contracts to spatial audio rather than leaving them as inspection-only data.

**Independent Test**: Load a world with an area audio binding and at least one decoded sound emitter, move the camera across the emitter radius and across an area boundary, and verify that the active ambience/emitter set changes without duplicate or stale sources.

**Acceptance Scenarios**:

1. **Given** the camera is over a resolved area, **when** time-of-day or underwater state changes, **then** the selected day/night or underwater ambience follows the client data for that area.
2. **Given** an MCSE emitter is resident and its sound identity resolves, **when** the camera enters or leaves its spatial range, **then** the emitter is audible only within the configured attenuation range.
3. **Given** an emitter record or sound identity cannot be decoded for the active build, **when** the scene updates, **then** the emitter is skipped with a diagnostic reason and other audio continues.

### User Story 3 - Capture audio with camera videos (Priority: P1)

As a viewer user, I want Play + Video to include the same audio mix I hear during preview, so the camera tooling can produce self-contained demo videos.

**Why this priority**: The camera-path system exists partly to create demonstrations of the viewer. A silent capture route would leave the primary presentation workflow incomplete.

**Independent Test**: Start Play + Video on a path with audio, wait for the existing preload gate, stop after several seconds, and verify that the resulting media contains synchronized video and audio or that a clear unsupported-audio report is produced.

**Acceptance Scenarios**:

1. **Given** audio is available, **when** Play + Video begins, **then** audio capture starts at the same logical timestamp as the first captured video frame.
2. **Given** recording stops, **when** the output is finalized, **then** the audio stream ends cleanly with the video and is not left playing in the viewer.
3. **Given** a selected backend cannot provide capture audio, **when** recording begins, **then** the UI states whether the output is silent, separately saved, or unavailable before the user relies on it.

### User Story 4 - Inspect and control audio capabilities (Priority: P2)

As a viewer user, I want to see which audio paths the loaded client supports and control their volume independently, so missing codecs, DLS banks, or build definitions do not appear as unexplained silence.

**Why this priority**: World audio spans several historical formats and client-era schemas. Capability reporting is necessary for honest support claims and for debugging real client data.

**Independent Test**: Open the audio diagnostics surface for clients containing different combinations of WAV, MP3, OGG, MIDI, DLS, and emitter records, and verify that each capability is labeled as available, unsupported, unresolved, or not present.

**Acceptance Scenarios**:

1. **Given** the loaded client has multiple audio asset families, **when** the user opens audio diagnostics, **then** each family reports its source, build/schema status, and playback/capture capability.
2. **Given** music, ambience, emitters, and UI/test audio are active, **when** the user changes a category volume or mutes the master bus, **then** only the intended category changes.
3. **Given** a format is not supported by the current runtime backend, **when** the user attempts playback, **then** the viewer fails gracefully and offers an inspect/export or offline-tooling path when one exists.

### User Story 5 - Preserve the long-term client/server direction (Priority: P3)

As a project maintainer, I want the audio system to become a reusable world-runtime service, so the viewer can eventually grow into a single-player client experience backed by the existing Alpha-Core SQL world data and a compatible local server/session layer.

**Why this priority**: The long-term goal is larger than camera playback. Audio, terrain, objects, NPCs, game objects, and session state must eventually share a coherent runtime boundary.

**Independent Test**: Review the architecture roadmap and verify that audio playback is an independently testable client capability, while the future server/session work has explicit prerequisites and is not represented as already implemented.

**Acceptance Scenarios**:

1. **Given** the viewer audio service is used by camera preview, **when** future world/session code is introduced, **then** it can consume the same world-audio events without depending on ImGui or capture UI state.
2. **Given** Alpha-Core SQL data is available, **when** the roadmap is evaluated, **then** its use as single-player world/session input is recorded as a future integration boundary rather than silently coupled into the audio MVP.

## Edge Cases

- A camera path has no direct audio metadata but is inside a zone with area ambience; the viewer may use explicitly selected area ambience, but must not invent a camera-specific soundtrack.
- A camera path crosses area boundaries; ambience transitions must be deterministic and must not restart on every frame.
- A camera path begins inside a WMO or emitter volume; positional audio must use the same map-bound placement as the camera and must not double-apply the FlyBy origin transform.
- Audio files are stored in an MPQ/archive source rather than loose files; resolution must use the active client data source and preserve provenance.
- A DBC/DB2 layout is unavailable or mismatched for the selected client build; the viewer must report the schema gap and retain raw evidence rather than applying a hardcoded field layout.
- A MIDI sequence references a DLS/DirectSound bank that is missing, unsupported, or platform-incompatible; the sequence must fail independently while other audio buses remain usable.
- A sound emitter has an undecoded or era-specific record stride; raw entry bytes may be retained for inspection, but no guessed positional playback is allowed.
- Multiple camera or capture actions start concurrently; one authoritative audio transport must prevent duplicate playback and stale capture handles.
- The user changes client, map, time of day, or audio settings while playback is active; resources must be released or re-bound without leaving orphaned streams.
- A recording is requested with no supported audio backend; video capture must remain available with an explicit silent-output status.

## Requirements

### Functional Requirements

- **FR-001**: The viewer MUST expose an audio runtime boundary separate from ImGui panels, camera authoring state, and the video-capture UI.
- **FR-002**: The audio runtime MUST provide independent master, music/ambience, emitter/effects, and optional UI/test volume controls.
- **FR-003**: The viewer MUST resolve audio assets through the configured client data source, including loose files and archives, and MUST preserve the requested virtual path and resolved source.
- **FR-004**: The viewer MUST maintain a capability matrix for WAV, MP3, OGG, MIDI sequence, DLS/DirectSound bank, and audio-capture support; unsupported capabilities MUST be reported explicitly.
- **FR-005**: Camera-path playback MUST support an optional audio binding with a shared logical transport for play, pause, stop, loop, scrub, and time changes.
- **FR-006**: Camera-path audio bindings MUST be sourced from proven client metadata, explicit project sidecars, or user selection; the viewer MUST NOT guess a soundtrack from a filename alone.
- **FR-007**: Area ambience MUST reuse the existing `AreaTable`/`AreaMIDIAmbiences` catalog where the active build schema proves those fields, including day/night and underwater variants.
- **FR-008**: MCSE sound-emitter data MUST be represented as map-bound emitter candidates with source tile/chunk, raw provenance, decoded identity when proven, position, range/volume inputs when available, and an explicit unresolved state when not.
- **FR-009**: The runtime MUST update positional emitters from the camera/player-head world position using bounded resident content; it MUST not require loading the whole map merely to evaluate sound.
- **FR-010**: WMO-contained or other non-ADT sound emitters MAY be added only through a build-specific reader contract with evidence for their placement and sound identity; no WMO field may be invented from an unproven layout.
- **FR-011**: Play + Video MUST use the same logical audio transport as preview and MUST report whether audio is muxed, separately written, or unavailable before finalization.
- **FR-012**: Audio playback MUST be stoppable and disposable on map/client replacement, viewer shutdown, capture cancellation, and playback completion.
- **FR-013**: A missing asset, decoder, DLS bank, or schema MUST fail closed for that source while leaving unrelated audio sources and video capture usable.
- **FR-014**: The runtime MUST support a backend-neutral playback contract so platform-specific or format-specific libraries can be evaluated without coupling client data readers to one library.
- **FR-015**: Python or external tooling MAY provide offline inspection, conversion, or capability probes, but the interactive viewer's authoritative playback/session contract MUST remain available to the C# runtime.
- **FR-016**: Audio diagnostics MUST expose enough provenance to distinguish absent client data, unresolved DBC/DB2 schema, unsupported decoder, missing bank, archive-read failure, and successful playback.
- **FR-017**: The design MUST keep a future integration seam for world/session events supplied by the Alpha-Core SQL-backed single-player server direction without implementing that server as part of this feature.

### Key Entities

- **Audio capability**: A format/backend feature with availability, source, build/schema status, playback status, and capture status.
- **Audio asset**: A client-referenced sound, sequence, bank, or stream with virtual path, resolved source, format, and provenance.
- **Audio binding**: A proven or explicitly authored association between a camera path/sequence/area and one or more audio assets.
- **Audio transport**: Shared playhead state and lifecycle used by camera preview, ambience, emitters, and capture.
- **Audio bus**: A controllable category such as master, music/ambience, emitters/effects, or UI/test.
- **World audio emitter**: A map-bound positional source derived from MCSE or a separately proven WMO/client record.
- **Area ambience binding**: Existing area and MIDI-ambience metadata joined to resolved day/night/underwater assets and optional DLS bank.
- **Audio diagnostic**: A structured result explaining resolution, schema, decoder, bank, playback, and capture state.
- **Single-player world session**: A future runtime boundary that consumes terrain, object, NPC, game-object, audio, and SQL-backed world events; it is roadmap context, not an implementation output of Spec 146.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A user can start, pause, scrub, loop, and stop an audio-enabled camera path without duplicate streams or transport drift in a focused playback test.
- **SC-002**: At least one representative client asset from every supported capability family is either played successfully or reported with a specific unsupported/unresolved reason; no capability is described as supported solely because a file exists.
- **SC-003**: Moving through a test scene with at least 100 resident emitter candidates evaluates only the bounded resident/camera-relevant set and does not trigger whole-map audio loading.
- **SC-004**: A successful Play + Video run produces synchronized audio/video output with a measured start offset within one capture frame, or records an explicit silent/unavailable result before finalization.
- **SC-005**: Area ambience transitions across a test area boundary without per-frame restart, duplicate playback, or stale source retention.
- **SC-006**: Missing DLS banks, unsupported formats, archive-only assets, and unproven build schemas each produce distinct diagnostic states while unrelated viewer rendering continues.
- **SC-007**: The audio service can be exercised without constructing the ImGui shell, and its contract can be consumed by future world/session code without importing capture-panel types.
- **SC-008**: The project roadmap clearly distinguishes implemented viewer audio, planned emitter coverage, and the future Alpha-Core-backed single-player client/server objective.

## Assumptions

- The existing `AlphaAreaAudioCatalog`, `AlphaAreaAudioAssetResolver`, and `AdtMcseReader` are foundational read/inspection contracts and will be reused rather than rewritten.
- Client/build-specific DBC/DB2 definitions remain authoritative. Hardcoded sound-ID, field-layout, or file-path lists are not acceptable as a compatibility strategy.
- C# is the canonical interactive runtime owner. Python may be used for offline inspection or conversion only when it provides a capability the runtime cannot reasonably own.
- The first implementation targets one active world/session and one authoritative audio transport; multi-world mixing is out of scope.
- The existing ffmpeg video route remains the initial capture integration point, subject to proving whether it can accept the runtime audio mix or requires a separate audio/mux bridge.
- Real-client playback and capture proof remains user-run. Build/test results do not establish audible or synchronized output.

## Out of Scope

- Implementing the Alpha-Core-backed single-player server, login/session protocol, NPC AI, quest simulation, or authoritative world mutation.
- Rewriting DBC, DB2, ADT, WMO, M2, MDX, MPQ, or MCSE readers that already exist.
- Guessing WMO audio layouts or treating raw MCSE bytes as decoded sound identities without build evidence.
- Shipping proprietary client audio, DLS banks, MIDI, or other game assets in the repository.
- Replacing the renderer or requiring whole-map residency for audio playback.
- Choosing a final third-party audio library before a capability/ABI/platform research phase.
