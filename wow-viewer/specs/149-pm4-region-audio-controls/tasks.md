# Tasks: PM4 Region Navigation and Audio Trigger Controls

**Input**: Design documents from `/specs/149-pm4-region-audio-controls/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/)

**Tests**: Focused automated tests are required by FR-018. Real-client visual, streaming, and audible
proof remain user-owned gates.

## Phase 1: Foundation and contracts

**Purpose**: Establish shared identity/state contracts and complete the matching-retirement caller audit
before story implementation.

- [ ] T001 [P] Add `Pm4RegionNavigationItem`, `Pm4RegionFocusRequest`, availability state, and finite
  bounds invariants in `wow-viewer/src/core/WowViewer.Core.PM4/Models/`.
- [ ] T002 [P] Add typed audio trigger source kinds (including legacy `Mcnk`), instance keys, and
  default-off enablement state in `wow-viewer/src/core/WowViewer.Core/Audio/` without moving playback
  ownership into the UI.
- [ ] T003 [P] Write failing focused region aggregation/focus contract tests in
  `wow-viewer/tests/WowViewer.Core.Tests/Pm4RegionNavigationTests.cs` for one-row-per-region,
  multi-tile totals, empty stubs, stale rows, and finite focus requests.
- [ ] T004 [P] Write failing focused audio enablement/coordinate contract tests in
  `wow-viewer/tests/WowViewer.Core.Tests/AudioTriggerControlTests.cs` for default-off state,
  per-instance identity, MCSE raw-to-normalized position, MCNK rows without MCSE, master blocking, and
  preview independence.
- [ ] T005 Audit PM4 matching/correlation callers with `rg` across
  `wow-viewer/src`, `wow-viewer/tools`, and `wow-viewer/tests`; record whether
  `Pm4WmoGroupMatchService.cs`, object-match state, correlation reports, and saved-match persistence have
  non-UI owners before any deletion.

**Checkpoint**: Contracts, test expectations, and the safe removal boundary are explicit; no UI or
runtime story is considered complete yet.

## Phase 2: User Story 1 - Browse and focus decoded PM4 regions (Priority: P1) 🎯 MVP

**Goal**: Produce a deterministic resident region list and double-click camera focus without external
asset matching or whole-map loading.

**Independent Test**: With a fixture containing at least two non-empty regions, list each region once,
double-click each row, and verify selection, finite framing, and bounded residency behavior.

- [ ] T006 [US1] Extend the resident PM4 snapshot in
  `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` to aggregate `MshdRegionId`, object/tile/
  surface totals, union bounds, center, empty-stub state, and availability from `_pm4TileObjects`.
- [ ] T007 [US1] Add region selection, stale-state clearing, and a validated focus-request seam to
  `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`; reject missing/non-finite bounds without
  mutating camera or PM4 transforms.
- [ ] T008 [US1] Replace the correlation-first PM4 workbench content with a region list in
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs`; support deterministic sorting, selected
  state, empty/loading/unavailable messaging, and ImGui double-click detection.
- [ ] T009 [US1] Reuse/extend the existing `FocusCameraOnBounds` and bounds-union path in
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs`, using the authoritative camera
  position/rotation and normal AOI/residency update path with finite bounds clamping and a safe offset
  for flat/large regions.
- [ ] T010 [US1] Keep region selection/highlight synchronized across PM4 reload, tile unload, map change,
  and camera-driven residency changes in `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` and
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs`.
- [ ] T011 [US1] Make `Pm4RegionNavigationTests.cs` pass, including multi-tile aggregation, empty-stub
  rejection, stale refresh, and finite focus-request assertions.

**Checkpoint**: User Story 1 is independently demonstrable before the correlation UI retirement begins.

## Phase 3: User Story 2 - Inspect PM4 facts without correlation UI (Priority: P1)

**Goal**: Remove user-facing matching/correlation interaction and make PM4 presentation factual and
region-oriented.

**Independent Test**: Open PM4 Workbench, select PM4 geometry, and hover it; no correlation/matching
control, candidate, score, saved-match, or match wording remains in the PM4 presentation path.

- [ ] T012 [US2] Remove the `Correlation` workbench tab and obsolete PM4 matching window/tab state from
  `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` and
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs`.
- [ ] T013 [US2] Remove WMO/M2 match buttons, shape search, saved-match rows, match-detail panels, and
  candidate suggestions from `wow-viewer/src/viewer/WoWViewer/ViewerApp_Pm4Utilities.cs`, preserving
  decoded selected-object and region facts.
- [ ] T014 [US2] Remove matching/correlation labels and obsolete PM4 tab routing from
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Sidebars.cs`,
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Workspaces.cs`,
  `wow-viewer/src/viewer/WoWViewer/Workbench/WorkbenchNavigator.cs`, and the PM4 menu/status text in
  `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs`.
- [ ] T015 [US2] Replace hovered PM4 match-candidate rendering in
  `wow-viewer/src/viewer/WoWViewer/ViewerApp.cs` with decoded tooltip fields only: tile, region, CK24,
  part, surface count, bounds/center, and proven grouping/source context.
- [ ] T016 [US2] Apply the Phase 1 caller-audit result to `wow-viewer/src/viewer/WoWViewer/Pm4WmoGroupMatchService.cs`
  and related match models/state: remove orphaned viewer-only code, or document and isolate any retained
  non-UI research/export owner without leaving a reachable workbench control.
- [ ] T017 [US2] Run a source/UI regression audit over `wow-viewer/src/viewer/WoWViewer` and add/update
  focused assertions or test notes so PM4 presentation contains no correlation/matching fields while
  region navigation remains reachable.

**Checkpoint**: User Story 2 is independently demonstrable and the old matching path cannot distract from
decoded PM4 inspection.

## Phase 4: User Story 3 - Review and explicitly enable audio triggers (Priority: P1)

**Goal**: Make world-trigger playback silent by default, inspectable for every resident MCNK,
MCSE/current-area trigger, and explicitly controllable per trigger.

**Independent Test**: Load a 0.5.3 fixture with MCNK liquid/environment data and no MCSE, plus a later
fixture with MCSE and area audio data; verify all rows are off with zero automatic starts, verify MCSE
coordinates normalize into the owning tile/chunk, enable one supported row, then disable it and verify
stop/no-restart behavior.

- [x] T018 [US3] Add a canonical MCSE local-position normalization helper and focused tests, using the
  owning tile/chunk origin and existing renderer axis convention before range checks or OpenAL placement;
  preserve raw/local values for diagnostics in `AlphaTerrainAdapter`, `StandardTerrainAdapter`, and the
  audio runtime seam.
- [x] T019 [US3] Project decoded `TerrainChunkData.McnkFlags` and `LiquidChunkData` into bounded MCNK
  environmental/liquid trigger candidates, preserving raw liquid identity when available and leaving
  an explicit unresolved mapping diagnostic when no client-proven SoundEntries row exists.
- [ ] T020a [US3] Add a build-aware `ZoneMusic` reader/model and resolver so `AreaTable.ZoneMusic`
  follows `ZoneMusic row -> day/night Sounds[2] -> SoundEntries`; preserve ZoneMusic volume/files/
  scheduler fields as metadata and keep `MIDIAmbienceUnderwater` selection explicit. Add a regression
  fixture for the 0.5.3 row-1 mapping to SoundEntries 2523/2533.
- [ ] T020 [US3] Add default-off master and per-instance trigger enablement, stable MCNK/MCSE/area keys,
  and stop/restart guards to `wow-viewer/src/viewer/WoWViewer/Audio/WorldAudioRuntime.cs`; gate
  `TryStartEmitter`, MCNK/liquid starts, and `UpdateAreaMusic` without changing explicit preview behavior.
- [ ] T021 [US3] Extend `WorldScene` audio forwarding and diagnostic projection in
  `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs` so resident MCNK, MCSE, and applicable
  current-area/ZoneMusic rows share a typed inspectable trigger list and explicit disabled state.
- [ ] T022 [US3] Replace the diagnostic-only audio panel in
  `wow-viewer/src/viewer/WoWViewer/ViewerApp_Audio.cs` with a bounded trigger list showing source,
  context, MCNK flags/liquid identity, IDs/path, raw/normalized coordinates, provenance/stage status,
  enablement, per-row toggles, and a master world-trigger toggle defaulted off; retain deliberate
  SoundEntries preview and gain/mute controls separately.
- [ ] T023 [US3] Reset world-trigger enablement and owned sources during runtime configure, map/client
  replacement, tile removal, and dispose in `WorldAudioRuntime.cs` and `WorldScene.cs`.
- [ ] T024 [US3] Make `AudioTriggerControlTests.cs` pass for 0.5.3 MCNK-without-MCSE enumeration,
  liquid-family selection, normalized MCSE distance, ZoneMusic row-to-SoundEntries indirection,
  default-off initialization, duplicate SoundEntries IDs with distinct instance keys, master/per-row
  stop behavior, disabled diagnostics, duplicate-start prevention, and preview independence.
- [ ] T025 [US3] Update audio UI wording and status classifications so user-disabled, unresolved,
  unsupported, backend-unavailable, ready, active, and stopped are distinguishable in
  `AudioTriggerDiagnostic.cs` and `ViewerApp_Audio.cs`.

**Checkpoint**: User Story 3 is independently demonstrable without looping samples starting during map
inspection.

## Phase 4A: User Story 4 - Inspect resident zone and subzone boundaries (Priority: P2)

**Goal**: Make the decoded resident MCNK AreaNumber/AreaTable context visible as opt-in 3D chunk
footprints with distinct Zone/SubZone styling and world-space name tags.

**Independent Test**: Load resident chunks with multiple resolvable area values, enable the area
overlay, verify one finite labeled group per resolved Zone/SubZone identity, then stream a tile out and
verify its footprint and label contribution disappear without a stale whole-map result.

- [x] T032 [US4] Add a resident chunk snapshot/revision accessor to
  `wow-viewer/src/viewer/WoWViewer/Terrain/TerrainRenderer.cs` so the overlay can rebuild only after
  tile residency changes and never scan unloaded map data.
- [x] T033 [US4] Add `AreaOverlayRegion` and footprint-cell models plus a deterministic aggregation
  helper that uses `AreaTableService.ResolveArea`, `TryGetParent`, and the existing packed
  Zone/SubZone contract; omit unresolved rows and count them explicitly.
- [x] T034 [US4] Add opt-in area overlay state to
  `wow-viewer/src/viewer/WoWViewer/Terrain/WorldScene.cs`; render resident footprint boxes and group
  pins through the existing bounding-box batch, with separate Zone/SubZone colors.
- [x] T035 [US4] Add the area-boundary toggle and unresolved/resident summary to the existing spatial
  investigation UI in `wow-viewer/src/viewer/WoWViewer/ViewerApp_Investigation.cs`.
- [x] T036 [US4] Project one world-space name tag per resolved group from `ViewerApp` using the existing
  scene viewport and foreground draw-list projection path; keep labels disabled with the 3D overlay.
- [ ] T037 [US4] Add focused aggregation tests for packed AreaNumber identity, duplicate names across
  IDs/maps, finite bounds, deterministic colors, unresolved omission, and stale resident refresh.

**Checkpoint**: User Story 4 is independently demonstrable as an honest resident-footprint inspection
overlay and does not affect terrain, camera, lighting, or audio.

## Phase 4B: Resident audio speaker markers (Priority: P2)

**Goal**: Make the normalized positions of resident MCSE and legacy MCNK/liquid emitters visible as an
opt-in 3D diagnostic overlay without coupling visualization to playback.

- [x] T038 [US3] Add a cached resident-emitter snapshot in
  wow-viewer/src/viewer/WoWViewer/Audio/WorldAudioRuntime.cs, forward it through WorldScene,
  expose an off-by-default audio-panel toggle, and render source-colored pins through the existing
  BoundingBoxRenderer batch. Keep markers bounded to resident tiles and independent of audio file
  probing, world-trigger enablement, and OpenAL source state.

## Phase 5: Cross-cutting validation and handoff

**Purpose**: Validate the combined feature and prepare the user-owned runtime proof without claiming it
from compilation alone.

- [ ] T026 [P] Update `wow-viewer/specs/149-pm4-region-audio-controls/quickstart.md` with the final
  focused test filters, configured client/build recording fields, and exact PM4/audio proof steps.
- [ ] T027 [P] Update the owning PM4/audio continuity notes only after implementation proof changes the
  handoff; do not rewrite the existing negative Ghidra/MIDI/DLS findings.
- [ ] T028 Run `git diff --check` and the focused PM4/audio tests from the quickstart.
- [ ] T029 Run `dotnet build I:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug --no-restore` and
  report source/build proof separately from visual, streaming, and audible proof.
- [ ] T030 Perform the final source audit for remaining reachable correlation/matching UI and automatic
  world-trigger starts; preserve unrelated dirty worktree changes and stage only intended files if a
  later commit is requested.
- [ ] T031 Hand off the PowerShell-ready user-run PM4 camera and audio trigger checks with configured
  client root, build identity, and proof level; keep player/game-mode movement as a separate future
  Spec Kit feature.

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: No implementation dependency; blocks all story work with the model/test/caller boundary.
- **Phase 2 / User Story 1**: Depends on T001, T003, and T005; must pass its checkpoint before PM4 UI
  retirement is treated as complete.
- **Phase 3 / User Story 2**: Depends on Phase 2's region presentation and T005's caller audit.
- **Phase 4 / User Story 3**: Depends on T002 and T004; may proceed in parallel with Phase 3 once the
  shared runtime contract is stable.
- **Phase 4A / User Story 4**: Depends on the existing AreaTable resolver and resident terrain metadata;
  it may proceed independently of PM4 matching retirement and audio playback implementation.
- **Phase 5**: Depends on the desired story checkpoints and is the only phase that prepares final user
  runtime proof.

### Parallel Opportunities

- T001, T002, T003, and T004 are disjoint contract/test slices and can be prepared in parallel.
- T018, T019, and T012 can be implemented in parallel after their respective foundation/audit
  prerequisites; they touch separate coordinate/producer/PM4 presentation ownership areas.
- T026 and T027 are documentation-only and can be prepared in parallel with source validation.

## Implementation Strategy

1. Complete the contracts, failing tests, and caller audit.
2. Implement and validate PM4 region navigation as the first independently useful slice.
3. Remove matching UI only after the region path works and the caller boundary is known.
4. Implement and validate MCNK/liquid and normalized-MCSE trigger production, then default-off audio
   controls including area music gating.
5. Add opt-in resident speaker markers from the normalized emitter snapshot without changing playback.
6. Run focused source proof, then hand off real-client visual/streaming/audible proof to the user.

## Notes

- Do not implement player-height, walking/running, jumping, collision, or game mode under these tasks.
- Do not infer MIDI/DLS playback or native MCSE callback support from this UI/runtime policy change.
- Do not delete shared PM4 research code without the T005 caller audit.

## Implementation checkpoint — 2026-08-14

- T018 is implemented: Alpha and standard MCSE records now preserve raw local coordinates while their
  renderer-facing positions are anchored to the owning chunk corner through the shared core transform.
  MCNK liquid candidates now use the same terrain corner convention for their center position; the
  previous `+ halfChunk` placement put the audio marker on the opposite side of the liquid chunk.
  Focused coordinate/audio-contract coverage is 11/11, and the Windows viewer project builds with
  0 errors using isolated output because the live viewer owns the normal Debug binaries. This is not
  audible or visual real-client proof.
- T019 is implemented: resident MCNK/MCLQ/MH2O liquid state now produces inspectable candidates;
  exact-build `SoundWaterType` resolves supported rows and leaves missing mappings unresolved.
- T038 is implemented: the audio runtime now publishes a residency-change-only snapshot of normalized
  MCSE/MCNK emitter records. WorldScene exposes an opt-in, default-off 3D speaker overlay that uses
  the existing batched pin renderer with amber MCSE, cyan MCNK water, and purple MCNK environment
  markers. Marker rendering does not probe files, enable playback, or create OpenAL sources.
- Automatic ZoneMusic playback is now explicitly muted behind a tested world-audio policy. Area
  resolution and status diagnostics remain active, while MCNK/MCSE emitter playback remains separate.
- Alpha MCLQ preserves its packed 9x9 vertex records and 8x8 tile flags. Camera rotation no longer
  rebuilds residency, terrain keeps one capped unload-hysteresis ring, and WMO frustum-visible groups
  are admitted after portal evaluation to avoid spotty interiors.
- T020/T022/T024 remain open for per-trigger enablement, explicit disabled diagnostics, ZoneMusic row
  indirection, and focused runtime control tests. The active tile list now follows mouse-look without
  reopening the streaming lease; focused catalog/flag/audio contract tests pass 53/53. These source
  changes do not constitute visual, FPS, or audible real-client proof.
