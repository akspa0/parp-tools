# Tasks: Artifact World Simulator Runtime

**Input**: Design documents from `specs/148-world-simulator/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, and `contracts/`

**Execution rule**: Complete and validate one phase before starting the next. Preserve the dirty
`wow-viewer/imgui.ini`; it is not part of this feature.

## Phase 1: Audio truth surface (US1, P1) — MVP

**Goal**: Explain every current-tile/WMO audio decision without requiring successful playback.

**Independent test**: A client-backed map with MCSE or area-trigger data shows one diagnostic row per
relevant trigger, including raw/transformed coordinates and a terminal stage status.

- [x] T001 [P] [US1] Add audio diagnostic enums/value records for trigger kind, coordinate profile,
  provenance, decoder/backend stages, and terminal state in the existing shared runtime/audio owner
  under `wow-viewer/src/core/WowViewer.Core.Runtime/`.
- [x] T002 [P] [US1] Extend the shared `TerrainSoundEmitter` contract in
  `wow-viewer/src/viewer/WoWViewer/Terrain/` to preserve raw MCSE position, transformed position,
  era/profile label, and existing range/start/end/mode data without changing the current transform.
- [x] T003 [P] [US1] Add pure tests for raw/transformed coordinate retention and Alpha/Standard
  emitter metadata in `wow-viewer/tests/WowViewer.Core.Tests/`.
- [x] T004 [US1] Add a read/provenance result at the `IDataSource`/`MpqDataSource` boundary in
  `wow-viewer/src/viewer/WoWViewer/DataSources/`, distinguishing catalog visibility, actual read,
  byte length, and failure without assuming a specific archive filename.
- [x] T005 [US1] Add `WorldAudioRuntime` diagnostic projection that evaluates resident MCSE triggers
  through resolution, provenance/read, decode, range, mute, and backend stages, while retaining the
  active area music/ambience decision in `AreaMusicStatus`, without starting playback in
  `wow-viewer/src/viewer/WoWViewer/Audio/`.
- [ ] T006 [P] [US1] Add focused tests for unresolved SoundEntries, missing resources, decode failure,
  OpenAL unavailable, out-of-range, muted, active, unsupported MIDI/DLS states, and
  `AreaTable.ZoneMusic -> ZoneMusic -> day/night SoundEntries` indirection.
- [ ] T006a [US1] Add a build-aware ZoneMusic reader/model and diagnostic resolver so a ZoneMusic ID
  is never treated as a direct SoundEntries ID in `wow-viewer/src/core/WowViewer.Core.IO/Dbc/` and
  `wow-viewer/src/viewer/WoWViewer/Audio/`.
- [x] T006b [US1] Align Alpha AreaNumber resolution across the status-bar and terrain audio paths:
  decode high/low `ushort` words, prefer `AreaNumber`/`ParentAreaNum` for Alpha rows, remove
  standalone half-word aliases, and forward the shared Zone/SubZone lookup context.
- [x] T007 [US1] Expose current-tile/WMO audio diagnostics from `WorldScene` using the active
  listener/actor position while retaining the existing audio playback path.
- [x] T008 [US1] Render a scrollable audio-trigger table with IDs, tile/chunk, raw XYZ, world XYZ,
  range/distance, SoundEntries/path, provenance, and terminal state in `ViewerApp_Audio.cs`.
- [ ] T009 [US1] Add an audio proof status line that separately reports archive/read, decoder, and
  OpenAL/backend availability; prevent finalizer or missing-native-library errors from hiding rows.
- [ ] T010 [US1] Run focused audio/area/minimap tests and the cross-platform Debug viewer build;
  update `specs/148-world-simulator/`, `memory-bank/activeContext.md`, and `memory-bank/progress.md`
  with the proof level and user-owned client/audio gate. **STOP for user client testing.**

## Phase 2: Explicit camera actor (US2, P1)

**Goal**: Make one authoritative transform/context feed camera, audio, collision, path playback, and
residency.

**Depends on**: Phase 1 diagnostics and user review of raw/transformed audio coordinates.

- [ ] T011 [P] [US2] Add `CameraActorState` and timestamped world-session snapshot contracts in
  `wow-viewer/src/core/WowViewer.Core.Runtime/World/`.
- [ ] T012 [P] [US2] Add pure actor snapshot tests for position, forward/up, roll, area/WMO context,
  audio listener, and path sample identity in `wow-viewer/tests/WowViewer.Core.Tests/`.
- [ ] T013 [US2] Route existing manual camera input and roll through the actor snapshot owner in the
  viewer shell without changing user-facing controls.
- [ ] T014 [US2] Route camera-path playback samples through the same actor snapshot and publish
  swept-path identity/hold interval for diagnostics.
- [ ] T015 [US2] Make `WorldScene`, `WorldAudioRuntime`, collision, and tile selection consume the
  actor snapshot or report their current snapshot ID when migration is incomplete.
- [ ] T016 [US2] Add WMO/terrain context transition diagnostics, including an explicit missing-WMO
  residency state, without rendering a new visible camera model.
- [ ] T017 [US2] Validate manual movement, roll, path playback, audio listener, and collision focused
  tests/build; update the phase handoff and stop for user visual/path proof.

## Phase 3: Fog/path residency attribution (US3, P1)

**Goal**: Retain the justified union of fog, path, inspection, and actor-context leases and stop
premature near-field unloads.

**Depends on**: Phase 2 actor snapshot and user-confirmed audio/coordinate context.

- [ ] T018 [P] [US3] Add `ResidencyLease` and lease-owner/value contracts in
  `wow-viewer/src/core/WowViewer.Core.Runtime/World/`.
- [ ] T019 [P] [US3] Add pure lease-union, hold/release, stale-target, and ownership tests in
  `wow-viewer/tests/WowViewer.Core.Tests/`.
- [ ] T020 [US3] Instrument current ADT/WDL/WMO residency selection with actor position, effective
  fog range, path warmup, inspection, and containing-WMO reasons in the existing terrain/runtime
  owners.
- [ ] T021 [US3] Replace premature release decisions with the justified lease union while preserving
  the existing user-configured detail/fog controls.
- [ ] T022 [US3] Add residency overlay/status showing active tile/object owners and release reasons,
  including the path recorder's swept warmup lease.
- [ ] T023 [US3] Add fixed selection tests for camera rotation, tile boundary crossing, WMO interior,
  and path playback without generalizing one era's fog formula to all clients.
- [ ] T024 [US3] Run focused tests/build and update the phase handoff; **STOP for user-run visual
  near-field and path-capture proof before batching changes.**

## Phase 4: Batch and WMO doodad performance (US3, P1)

**Goal**: Attribute and then reduce dense WMO doodad cost without per-frame nested scene-graph work.

**Depends on**: Phase 3 lease attribution and a user capture identifying the expensive owner.

- [ ] T025 [P] [US3] Add `RenderPerformanceSample` and batch attribution contracts in
  `wow-viewer/src/core/WowViewer.Core.Runtime/`.
- [ ] T026 [P] [US3] Add tests for unique-model, instance, WMO-internal doodad, terrain, audio, and
  total draw-call attribution in `wow-viewer/tests/WowViewer.Core.Tests/`.
- [ ] T027 [US3] Instrument selection, preparation, batch preparation, and draw submission timing
  in `wow-viewer/src/viewer/WoWViewer/Rendering/`.
- [ ] T028 [US3] Separate shared WMO doodad resource preparation from instance submissions and keep
  the per-frame path flat in the existing `WmoRenderer`/doodad owners.
- [ ] T029 [US3] Add a fixed camera-path benchmark export with client build, map, resolution, fog,
  resident leases, unique assets, submissions, and timings.
- [ ] T030 [US3] Compare a dense-WMO capture before/after the bounded optimization; retain the old
  path as default until user visual/FPS proof passes and update the owning spec.

## Phase 5: Optional audio backends and museum-session seams (US4, P2)

**Goal**: Support future local-simulator growth without mixing it into the first audio fix.

**Depends on**: Phase 1 diagnostics, Phase 2 actor/session, and explicit backend evidence.

- [ ] T031 [P] [US4] Add a backend capability contract for MIDI/DLS, including soundbank identity,
  native dependency, platform support, and failure state; do not add a dependency before evidence.
- [ ] T032 [US4] Add backend-independent tests proving WAV/OGG/MP3 behavior remains unchanged and
  MIDI/DLS stays explicit when unavailable.
- [ ] T033 [US4] Add local session/build/provenance summary suitable for artifact-museum workflows
  without persisting proprietary client assets.
- [ ] T034 [US4] Document extension seams for future NPC/game-object data and CPU-local creative tools
  without implementing an MMO server, network protocol, or LLM runtime in this feature.
- [ ] T035 [US4] Validate the quickstart and update `STATUS.md`, `activeContext.md`, and `progress.md`
  only when this phase has actual proof; otherwise leave it as a planned extension.

## Dependencies & Execution Order

### Phase dependencies

- Phase 1 is the MVP and blocks all later implementation phases.
- Phase 2 depends on the Phase 1 diagnostic contract and user review of coordinate evidence.
- Phase 3 depends on the Phase 2 actor snapshot.
- Phase 4 depends on Phase 3 lease attribution and a user capture.
- Phase 5 depends on the proven session/actor seams and a separately approved audio backend.

### Parallel opportunities

- T001, T002, and T003 can be developed in parallel because they have disjoint owners.
- T006 can be developed in parallel with T004 after the diagnostic state names are agreed.
- T011 and T012 can be developed in parallel; T013–T016 depend on the actor contract.
- T018, T019, T025, and T026 are independent contract/test slices within their phases.

### Implementation strategy

Deliver Phase 1 as the first usable increment, validate it, and stop. Do not begin camera actor or
renderer residency edits while the audio table still hides coordinate/path/backend decisions. Keep
working viewer behavior as the default until user-owned real-client proof promotes a later phase.
