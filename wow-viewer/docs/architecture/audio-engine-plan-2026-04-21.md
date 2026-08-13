# wow-viewer Audio Engine Plan

## Status

- status: active planning note
- intent: define the first real audio-engine ownership lane for the game-engine side of `wow-viewer`
- related micro-plans:
  - `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/GV-14A-audio-system-foundation.md`
  - `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/GV-14B-profile-audio-resolution-contracts.md`
  - `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/GV-14C-runtime-audio-scene-and-mixer.md`
  - `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/GV-14D-audio-asset-family-support-matrix.md`
  - `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/GV-17A-audio-backend-bridge.md`
  - `wow-viewer/docs/architecture/game-viewer-plan-pack-2026-05-14/GV-17B-midi-synth-and-instrument-bank-bridge.md`
- immediate trigger:
  - user direction is to start planning `wow-viewer` as a game-engine host, with audio as one of the first missing runtime subsystems
  - the user then explicitly reprioritized this lane toward `0.5.x` Alpha support first, especially MIDI ambience plus DLS soundbanks
  - ADT-family `MCSE` data is still a concrete world-format seam, but it is no longer the first universal starting point for the oldest recoverable audio path

## Why This Plan Exists

- `wow-viewer` already has bounded world bootstrap, tile terrain, liquid, placement, and visibility seams, but no first-class audio runtime yet
- the repo already carries enough shared access to make audio planning concrete:
  - root ADT and tile-family reads already flow through shared `MapFileSummaryReader`, `AdtSummaryReader`, `AdtMcnkSummaryReader`, `AdtLiquidReader`, and `WorldTerrainTileBuilder`
  - world-session and world-runtime consumers already reopen root ADTs in `WowViewer.App/WowViewerWorldRuntimeBridge.cs`
  - DB client tables are already reachable through shared `DbClientFileReader`
- the missing seam is not general file access anymore; it is typed audio ownership across:
  - Alpha area-audio lookup and asset discovery (the catalog and inspect proof now exist; playback does not)
  - ADT `MCSE` parsing across Alpha and later layouts
  - audio lookup tables and sound-entry resolution for later FMOD-era families
  - decoded-audio support for `wav`, `ogg`, and `flac`
  - MIDI sequencing plus instrument-bank support for `SFP0` and `DLS`
  - listener/emitter runtime state
  - backend playback and debug surfaces

## Current Concrete Boundary

Today `wow-viewer` has:

- world session bootstrap in `src/viewer/WowViewer.App`
- bounded world runtime tile reads in `WowViewerWorldRuntimeBridge`
- shared terrain, liquid, and placement readers in `src/core/WowViewer.Core.IO/Maps`
- shared DB client table access through `src/core/WowViewer.Core.IO/Files/DbClientFileReader.cs`
- a build-aware Alpha area-audio catalog and inspect command; see
  [`alpha-audio-catalog.md`](alpha-audio-catalog.md)
- loose-file and archive-backed probing for the catalog's referenced `.mid` and
  `.dls` assets

Today `wow-viewer` does not have:

- an `MCSE` chunk id or reader in shared ADT ownership
- typed world-sound emitter contracts in `WowViewer.Core`
- an audio-scene runtime in `WowViewer.Core.Runtime`
- a backend abstraction for playback, streaming, or listener updates
- a viewer-side audio diagnostics or capture surface

## Ownership Rule

- keep audio-format ownership in `wow-viewer/src/core/WowViewer.Core` and `wow-viewer/src/core/WowViewer.Core.IO`
- keep runtime audio-scene ownership in `wow-viewer/src/core/WowViewer.Core.Runtime`
- keep `WowViewer.App` as the first bounded consumer and diagnostics host, not the design owner of parsing or runtime contracts
- do not route new audio architecture back into `gillijimproject_refactor/src/MdxViewer` unless a bounded compatibility hotfix is explicitly requested
- in the future extracted repo shape, this subsystem belongs to `BASE` plus profile/personality libraries:
  - `BASE` owns engine-neutral audio runtime, mixer, backend, and diagnostics contracts
  - profile/personality libraries own audio lookup, schema, and asset-resolution differences

## Architectural Direction

Treat audio as the first explicit game-engine subsystem that sits beside the existing world-runtime seams.

The target stack should be:

1. shared format or lookup layer
  - Alpha area-audio lookup and asset discovery
   - root ADT `MCSE` parsing
  - DB client lookup readers for Alpha area MIDI bindings and later emitter/sound resolution
   - virtual-file resolution for real sound asset paths
  - decoded-audio family identification for `wav`, `ogg`, and `flac`
  - MIDI sequence plus bank resolution for `midi` + `SFP0` or `DLS`

2. runtime scene layer
   - listener state
   - resolved world emitters
   - play/stop/update policy
   - distance, loop, and culling state

3. backend layer
   - backend-neutral audio-device interface
   - sample or stream decode path
   - MIDI-to-rendered-audio synth bridge before backend playback
   - one initial desktop backend for bounded proof

4. app or engine integration layer
   - world camera drives listener transform
   - diagnostics expose active emitters and resolved assets
   - later engine-side consumers can reuse the same runtime without depending on the desktop app

## First Narrow Slice

The first slice should not try to play sound yet.

The first slice should prove that `wow-viewer` can read and report real Alpha area-audio bindings before it tries to model later-era emitter playback.

That keeps the proof cheap and falsifiable:

- if Alpha `AreaTable` or `AreaMIDIAmbiences` linkage is wrong, the inspect output will be wrong before any backend work exists
- if `.mid` and `.dls` asset discovery is wrong, the runtime will not even know which oldest assets it is supposed to restore
- once that is proven, Alpha-aware `MCSE` and later FMOD-era table resolution can land without guessing across eras

## Alpha-First Override

The earliest implementation order is now:

1. Alpha area-audio discovery and inspect proof
2. Alpha-aware `MCSE` reader and inspect proof
3. later-era sound table and asset resolution

This overrides the earlier `MCSE`-first ordering.

The rationale is repo-local evidence:

- `AreaTable.dbd` already shows `MIDIAmbience` and `MIDIAmbienceUnderwater` for `0.5.3.3368` and `0.5.5.3494`
- the checked-in `0.5.3` PDB dump already names `AreaMIDIAmbiencesRec` fields `m_DaySequence`, `m_NightSequence`, `m_DLSFile`, and `m_volume`
- existing reverse-engineering notes already describe `0.5.3` as a DirectMusic `.mid` plus `.dls` path, which is a stronger first restoration seam than jumping straight to later-style sound-entry resolution

## Slice 00 - Alpha Area MIDI/DLS Discovery And Inspect Proof

- target problem:
- the earliest client path is area-bound MIDI ambience plus DLS soundbanks; the catalog now owns the metadata join and asset-discovery proof, while runtime playback remains open
- implementation scope:
  - add shared Alpha area-audio contracts under `WowViewer.Core`, centered on `AreaTable` `MIDIAmbience` fields and `AreaMIDIAmbiences` records
  - add shared readers or resolver seams in `WowViewer.Core.IO` that can shape:
    - area id -> `MIDIAmbience`
    - `MIDIAmbience` -> day or night sequence + DLS bank + volume
  - add one thin inspect/report surface that proves those bindings on a real `0.5.3` client root or staged archive-backed copy
  - include virtual-file discovery for referenced `.mid` and `.dls` assets
- proof goal:
  - real-data inspect output on a fixed Alpha client root showing area-bound MIDI ambience resolution and concrete asset names
  - focused tests around typed record shaping where synthetic proof is practical
- out of scope:
  - no playback backend yet
  - no promise that later FMOD-era audio tables are already unified

## Ordered Slices

### Slice 01 - Shared `MCSE` Reader And Inspect Proof

- target problem:
  - `MCSE` is part of the ADT family, but `wow-viewer` does not currently expose it in shared code and Alpha-era payloads differ materially from later layouts
- implementation scope:
  - add `AdtChunkIds.Mcse`
  - add typed `MCSE` contracts under `WowViewer.Core.Maps` such as `AdtSoundEmitterFile`, `AdtSoundEmitterChunk`, and emitter entry models
  - add `AdtSoundEmitterReader` in `WowViewer.Core.IO/Maps`
  - add one thin inspect surface, likely under `WowViewer.Tool.Inspect map` or `adt inspect`, that reports emitter count, ids, positions, and raw flags or ranges
- proof goal:
  - real-data inspect output on a known Alpha ADT with `MCSE` payloads and explicit era-aware shape claims
  - focused tests with synthetic root-ADT payloads
- out of scope:
  - no playback
  - no later-era sound-entry linkage beyond raw ids unless it is trivial

### Slice 02 - Shared Sound Table Lookup Layer

- target problem:
  - Alpha and later clients do not share one clean audio lookup chain, so runtime playback cannot assume one resolver
- implementation scope:
  - add shared lookup readers for the minimum needed world-audio tables, keeping Alpha MIDI/DLS and later FMOD-era families explicit instead of flattening them together
  - first verify Alpha-era `AreaMIDIAmbiences` and related area bindings on `0.5.3` and `0.5.5`
  - then verify later-era emitter families such as `SoundEmitters`, `SoundEntries`, `SoundKit`, `WorldChunkSounds`, and `TerrainTypeSounds`
  - shape resolved assets so decoded-audio families and `midi` + bank families are explicit in the runtime-facing result
  - keep this as shared table-loading and record-shaping work, not tool-local parsing
- proof goal:
  - one shared resolver layer can take either an Alpha area-audio binding or a later emitter reference and produce a runtime-shaped audio recipe or a precise unresolved reason
- out of scope:
  - no backend playback yet
  - no pretending the cross-era schema is final until real-data proof exists for the target builds

### Slice 03 - Runtime Audio Scene Contracts

- target problem:
  - the world runtime has no place to carry audio state beside terrain, liquid, and object visibility
- implementation scope:
  - add runtime-owned contracts under `WowViewer.Core.Runtime`, for example:
    - `WorldAudioEmitterInstance`
    - `WorldAudioSceneFrame`
    - `WorldAudioListenerState`
    - `WorldAudioPlaybackRequest`
  - add a world-audio scene builder that consumes `MCSE` plus resolved sound-table data
  - keep the first version report-first: resolved emitters, audible/not-audible state, and reasons
- proof goal:
  - bounded runtime frame output that reports active audio emitters for a tile and listener position
- out of scope:
  - no real playback backend yet

### Slice 04 - Null Backend And App Diagnostics

- target problem:
  - before real playback, the app needs a runtime-facing place to prove the scene is updating correctly
- implementation scope:
  - add an `IAudioBackend` or equivalent runtime interface with a null backend implementation
  - wire `WowViewer.App` world session to display:
    - active emitter count
    - nearest audible emitters
    - resolved asset keys or sound-kit ids
    - loop or one-shot classification when known
  - add app-side listener state derived from the bounded world camera or inspection position
- proof goal:
  - app diagnostics change as the listener moves or the selected tile changes
- out of scope:
  - no waveform playback required

### Slice 05 - First Desktop Playback Backend

- target problem:
  - runtime resolution without sound output is not enough for engine ownership
- implementation scope:
  - introduce one desktop backend suitable for bounded proof on Windows first
  - keep the backend behind an interface so later engine or headless consumers do not depend on app-local implementation details
  - support the smallest viable asset set first, but plan explicitly for:
    - decoded `wav`
    - decoded `ogg`
    - decoded `flac`
    - rendered output from `midi` + `SFP0`
    - rendered output from `midi` + `DLS`
- proof goal:
  - bounded proof that a known world emitter can be heard at the expected listener position
  - backend can start, update, and stop one emitter without leaking resources
- out of scope:
  - no full music system
  - no reverb, occlusion, or environmental mixing claims yet

### Slice 06 - World Integration And Budgeting

- target problem:
  - once playback exists, the engine needs policy rather than raw emitter spam
- implementation scope:
  - add listener-follow integration to world camera/session state
  - add prioritization and voice budgeting
  - add max-distance and update-throttling rules
  - keep integration runtime-owned, not embedded into `WowViewerDesktopApp`
- proof goal:
  - stable audible set under listener movement without runaway voice growth
- out of scope:
  - no final native-client parity claims

### Slice 07 - Broader Audio Families

- target problem:
  - the first world-audio lane will only cover emitter-driven ADT proof
- implementation scope:
  - extend from MCSE-driven emitters into broader world-audio families as needed, such as zone ambience, terrain-type sounds, waterfall or liquid families, or model-driven effect playback
  - do this only after the emitter/runtime/backend core is already stable
- proof goal:
  - a broader game-engine audio surface that still reuses the same runtime contracts instead of spawning separate ad hoc systems

## Recommended Initial Runtime Contracts

The first stable contract set should stay small.

Candidate shapes:

- `WorldAudioEmitterDefinition`
  - raw emitter identity from ADT or later other sources
- `WorldAudioResolvedEmitter`
  - emitter definition plus resolved sound recipe
- `WorldAudioListenerState`
  - position, facing, and optional environment/context
- `WorldAudioSceneFrame`
  - resolved active emitters for one world update
- `WorldAudioBackendUpdate`
  - backend-neutral play, stop, and parameter changes

## Backend Guardrails

- start with one backend only; do not design three backends before the first emitter is proven end to end
- keep asset decode and playback behind interfaces so `WowViewer.App` is not forced to own audio device logic forever
- avoid binding the runtime to one desktop GUI toolkit or one specific shell loop
- be explicit that backend proof is engine-subsystem proof, not native-client parity proof

## Validation Plan

Use this order:

1. focused synthetic tests or typed-record checks for Alpha area-audio shaping where practical
2. real-data inspect proof on a fixed Alpha client root for `AreaTable` or `AreaMIDIAmbiences` plus `.mid` or `.dls` asset discovery
3. focused synthetic tests for Alpha-aware `MCSE` reader shape
4. real-data inspect proof on a fixed Alpha ADT with emitter payloads
5. shared resolver proof against real DB client tables for both Alpha and later supported eras
6. runtime-frame proof in `WowViewer.App` with null backend diagnostics
7. bounded audible proof with one backend and one known scene

## Real Risks

- Alpha `0.5.3` may need a dedicated DirectMusic-shaped runtime path rather than a thin compatibility layer over later FMOD-era assumptions
- `MCSE` payload interpretation may vary more across client eras than the current docs imply
- the exact lookup chain from `MCSE` to real playable assets may differ by build family, so the first shared resolver should stay explicit about supported eras
- sound assets may not all be simple wave playback; Alpha already proves `.mid` plus `.dls`, and later eras bring `.wav`, `.mp3`, and `.ogg`, so format handling should be proven from real client evidence before the backend scope expands
- supporting `midi` means the engine also owns the complexity of instrument-bank resolution and synthesis; `SFP0` and `DLS` should stay explicit as separate proof targets rather than being hand-waved as generic "soundfont support"
- occlusion, indoor/outdoor routing, reverb, and portal-aware audio are later world-runtime problems, not first-slice requirements

## Broader Game-Engine Direction

Audio is a good first subsystem because it forces the right engine shape without requiring full renderer parity first.

If this plan lands cleanly, the same pattern should later be reusable for other game-engine lanes:

- scene or entity ownership
- update scheduling
- input and camera systems
- gameplay-facing runtime services
- scripting/event routing
- physics/collision or navigation services

The key rule is the same as the current viewer/runtime cutover:

- shared file and table access in `Core` or `Core.IO`
- runtime state and system logic in `Core.Runtime`
- app shell as a consumer and proof surface only
