# Tasks: World Audio and Camera Playback

**Input**: Design documents from `wow-viewer/specs/146-audio-camera-playback/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/audio-runtime.md`, `quickstart.md`

**Tests**: Focused C# tests are required by the specification; audible client playback and synchronized capture remain user-run proof.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the audio contract surface without selecting a final decoder library.

- [ ] T001 Add the backend-neutral audio contract namespace and file layout under `wow-viewer/src/core/WowViewer.Core/Audio/`.
- [ ] T002 [P] Add the Spec 146 focused test file under `wow-viewer/tests/WowViewer.Core.Tests/Audio/` and reference the core audio contract.
- [ ] T003 [P] Add the initial capability identifiers and diagnostic reason vocabulary under `wow-viewer/src/core/WowViewer.Core/Audio/`.

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the reusable transport, provenance, and capability foundation required by every audio source.

- [ ] T004 Implement `AudioAsset`, `AudioCapability`, `AudioDiagnostic`, and `AudioBinding` contracts in `wow-viewer/src/core/WowViewer.Core/Audio/`.
- [ ] T005 Implement `AudioBus` and `AudioTransportState` lifecycle contracts with generation invalidation in `wow-viewer/src/core/WowViewer.Core/Audio/`.
- [ ] T006 Implement transport lifecycle tests for start, pause, stop, loop, scrub, completion, failure, and stale-generation disposal in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.
- [ ] T007 Add backend-neutral interfaces for asset resolution, capability probing, playback, runtime ownership, and capture bridging in `wow-viewer/specs/146-audio-camera-playback/contracts/audio-runtime.md` and corresponding C# contract files.
- [ ] T008 Verify that the core audio contracts do not reference ImGui, camera-panel types, SQL repositories, or third-party decoder APIs in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.

**Checkpoint**: The transport and capability foundation is independently testable and no playback backend has been made authoritative.

## Phase 3: User Story 1 - Hear camera tracks with their world audio (Priority: P1) 🎯 MVP

**Goal**: Bind an imported or authored camera path to explicit client/project audio and play both from one transport.

**Independent Test**: Use a fixture camera path and explicit audio binding to play, pause, scrub, loop, and stop without duplicate streams.

### Tests for User Story 1

- [ ] T009 [P] [US1] Add camera-binding provenance tests covering client metadata, project sidecar, explicit user selection, and filename-only rejection in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.
- [ ] T010 [P] [US1] Add camera/audio transport synchronization tests for playhead offsets, pause/resume, looping, scrub, and stop in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.

### Implementation for User Story 1

- [ ] T011 [US1] Add explicit camera-path audio-binding fields and serialization support in `wow-viewer/src/core/WowViewer.Core.Runtime/M2/M2CameraPath.cs` and its JSON options.
- [ ] T012 [US1] Implement camera audio-binding resolution using proven client metadata, sidecars, or explicit selection in `wow-viewer/src/viewer/WoWViewer/` without inferring tracks from filenames.
- [ ] T013 [US1] Connect Camera Path preview play, pause, scrub, loop, and stop to the shared audio transport in `wow-viewer/src/viewer/WoWViewer/ViewerApp_CameraPaths.cs`.
- [ ] T014 [US1] Add audio binding status, unavailable reasons, and category volume controls to the Tools > Utilities > Capture/Camera Path surface in `wow-viewer/src/viewer/WoWViewer/`.

**Checkpoint**: An explicit camera path can play with an audio binding, and missing audio does not block camera playback.

## Phase 4: User Story 2 - Reproduce area ambience and positional emitters (Priority: P1)

**Goal**: Turn existing area ambience and MCSE reader contracts into bounded world-audio source candidates.

**Independent Test**: A fixture world with area metadata and resident emitter candidates changes its active source set as the camera crosses area and emitter boundaries.

### Tests for User Story 2

- [x] T015 [P] [US2] Add area ambience binding tests for day/night and underwater selection using `AlphaAreaAudioCatalog` in `wow-viewer/tests/WowViewer.Core.Tests/Audio/` (catalog inheritance coverage is in `AlphaAreaAudioCatalogTests.cs`).
- [x] T016 [P] [US2] Add focused MCSE emitter decoding tests for source identity, decoded position,
  ranges, the proven Alpha 0.5.3 scheduler fields, and raw-entry preservation in
  `wow-viewer/tests/WowViewer.Core.Tests/`.

### Implementation for User Story 2

- [x] T017 [US2] Adapt `wow-viewer/src/core/WowViewer.Core/Audio/AlphaAreaAudioCatalog.cs` and `wow-viewer/src/core/WowViewer.Core.IO/Audio/AlphaAreaAudioAssetResolver.cs` into runtime-ready area ambience bindings without changing reader ownership. The viewer now loads active-build AreaTable/AreaMIDIAmbiences metadata and preserves the AreaTable ZoneMusic reference with parent inheritance; direct ZoneMusic row resolution is tracked separately in T017a.
- [x] T017b [US2] Make Alpha AreaNumber matching use shared high/low `ushort` components and feed the
  status-bar's resolved Zone/SubZone result into terrain area-music diagnostics without registering
  either component as an independent area ID.
- [ ] T017a [US2] Add a build-aware ZoneMusic reader/model so `AreaTable.ZoneMusic` resolves to the
  client row's day/night SoundEntries IDs before playback or diagnostic claims.
- [ ] T018 [US2] Add `WorldAudioEmitter` and unresolved-provenance contracts for existing `AdtMcseReader` output in `wow-viewer/src/core/WowViewer.Core/Audio/` and `wow-viewer/src/core/WowViewer.Core.IO/Audio/`.
- [x] T019 [US2] Add bounded resident tile/chunk and camera/player-head emitter admission in `wow-viewer/src/viewer/WoWViewer/Terrain/` without whole-map audio loading.
- [x] T020 [US2] Add resident-emitter attenuation, edge-triggered source start/stop, and failure-isolated diagnostics in the viewer audio runtime under `wow-viewer/src/viewer/WoWViewer/`.
- [x] T021 [US2] Add an audio diagnostics surface that exposes backend status and resident/active emitter counts in the viewer status bar; detailed MIDI/DLS and compressed-format reasons remain future work.

**Checkpoint**: Area ambience and resident MCSE candidates are represented with provenance and bounded evaluation; unproven WMO layouts remain excluded.

## Phase 5: User Story 3 - Capture audio with camera videos (Priority: P1)

**Goal**: Make Play + Video use the same audio transport and report the actual capture result.

**Independent Test**: A path with a supported audio binding records synchronized audio/video, or reports silent/separate/unavailable status before finalization.

### Tests for User Story 3

- [ ] T022 [P] [US3] Add capture-bridge tests for muxed, separate-audio, silent-only, unavailable, and cancellation outcomes in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.
- [ ] T023 [P] [US3] Add transport-generation tests proving map replacement, capture cancellation, and viewer shutdown dispose active audio handles in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.

### Implementation for User Story 3

- [x] T024 [US3] Add the first evidence-backed OpenAL implementation for resident MCSE PCM-WAV playback in `wow-viewer/src/viewer/WoWViewer/Audio/`.
- [ ] T025 [US3] Connect Play + Video startup, preload completion, stop, and finalization to the shared audio transport in `wow-viewer/src/viewer/WoWViewer/ViewerApp_CaptureAutomation.cs` and `ViewerApp_CameraPaths.cs`.
- [ ] T026 [US3] Add explicit muxed/separate/silent/unavailable audio capture reporting to the Capture panel and capture diagnostics in `wow-viewer/src/viewer/WoWViewer/`.
- [ ] T027 [US3] Ensure map/client replacement and capture cancellation release audio resources without disabling unrelated rendering or video capture in `wow-viewer/src/viewer/WoWViewer/`.

**Checkpoint**: Preview and capture share one transport and the user can tell whether the output contains audio.

## Phase 6: User Story 4 - Inspect and control audio capabilities (Priority: P2)

**Goal**: Make historical-format support honest and inspectable across client builds and platforms.

**Independent Test**: The diagnostics surface reports representative WAV, MP3, OGG, MIDI, DLS, emitter, and capture capability states with provenance.

### Tests for User Story 4

- [x] T028 [P] [US4] Add capability matrix fixture tests for WAV, MP3, OGG, MIDI, DLS, MCSE, and audio capture in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.
- [ ] T029 [P] [US4] Add archive-backed and missing-asset resolution tests using `wow-viewer/src/core/WowViewer.Core.IO/Audio/AlphaAreaAudioAssetResolver.cs` contracts.

### Implementation for User Story 4

- [ ] T030 [US4] Record representative client/build samples and candidate backend findings in `wow-viewer/specs/146-audio-camera-playback/research.md` before adding historical-format support.
- [ ] T031 [US4] Add audio capability and source diagnostics export to the viewer inspection surface in `wow-viewer/src/viewer/WoWViewer/`.
- [ ] T032 [US4] Add master, music/ambience, emitter/effects, UI, and optional test bus controls with persisted viewer settings in `wow-viewer/src/viewer/WoWViewer/` (the visible master mute toggle is implemented; category buses and persistence remain open).
- [x] T033 [US4] Add only evidence-backed MIDI/DLS/DirectSound support, or retain explicit offline/unsupported states, in `wow-viewer/src/viewer/WoWViewer/Audio/`.
- [x] T041 [US4] Add a bounded SoundEntries preview/stop surface, resident-ID discovery, gain controls, and last-diagnostic reporting in `wow-viewer/src/viewer/WoWViewer/`.

**Checkpoint**: Format and capture claims are evidence-backed and visible; unsupported historical paths fail closed.

## Phase 7: User Story 5 - Preserve the long-term client/server direction (Priority: P3)

**Goal**: Make the audio runtime consumable by a future world/session authority without implementing that authority here.

**Independent Test**: The runtime contract can receive world/session audio events without importing SQL, AI, quest, networking, or UI ownership.

### Implementation for User Story 5

- [ ] T034 [US5] Add world/session audio event contracts for area, emitter, time-of-day, weather, and scripted sequence changes under `wow-viewer/src/core/WowViewer.Core/Audio/`.
- [ ] T035 [US5] Add contract tests proving future event consumers do not depend on ImGui, Alpha-Core SQL repositories, or capture-panel types in `wow-viewer/tests/WowViewer.Core.Tests/Audio/`.
- [ ] T036 [US5] Update `wow-viewer/docs/architecture/single-player-client-server-roadmap-2026-08-12.md` with the proven audio/session boundary and prerequisites for a future server spec.

**Checkpoint**: The project direction is preserved without claiming a single-player server exists.

## Phase 8: Polish & Cross-Cutting Concerns

- [x] T037 [P] Run focused audio tests and the viewer/cross-platform Debug builds; record warnings without suppressing them.
- [ ] T038 [P] Run the user-owned audible and synchronized capture matrix from `wow-viewer/specs/146-audio-camera-playback/quickstart.md`.
- [ ] T039 Update `wow-viewer/README.md`, release notes, and capability claims only for formats/builds proven by the matrix.
- [x] T040 Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` with implementation status, proof boundaries, and remaining backend gaps.

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup and blocks all user stories.
- **User Story 1 (Phase 3)**: depends on Foundational; MVP camera/audio transport.
- **User Story 2 (Phase 4)**: depends on Foundational; can proceed alongside US1 after shared contracts exist.
- **User Story 3 (Phase 5)**: depends on US1 and the first backend slice.
- **User Story 4 (Phase 6)**: depends on Foundational and backend research; may proceed alongside US2/US3.
- **User Story 5 (Phase 7)**: depends on stable runtime contracts; it does not depend on server implementation.
- **Polish (Phase 8)**: follows the desired implementation slices and user proof.

### Parallel Opportunities

- T002 and T003 can run in parallel after T001.
- T009/T010, T015/T016, T022/T023, and T028/T029 are parallel focused-test tasks within their stories.
- US1 and US2 can proceed in parallel after Phase 2 if they do not edit the same runtime files simultaneously.
- T037 and T038 are separate build/proof activities and can be performed by different owners.

## Implementation Strategy

### MVP First (User Story 1)

1. Complete Setup and Foundational phases.
2. Implement the explicit camera binding and shared transport.
3. Stop and validate camera play/pause/scrub/loop/stop with one proven audio backend.
4. Do not claim area emitters, MIDI, DLS, or capture muxing until their own gates pass.

### Incremental Delivery

1. Add area ambience and bounded MCSE candidates without whole-map loading.
2. Add synchronized Play + Video reporting.
3. Expand capability diagnostics and historical formats only from evidence.
4. Add the future world/session event seam and keep server implementation separate.
