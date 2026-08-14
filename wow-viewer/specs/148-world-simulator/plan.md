# Implementation Plan: Artifact World Simulator Runtime

**Branch**: `148-world-simulator` | **Date**: 2026-08-14 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/148-world-simulator/spec.md`

## Summary

The current viewer has the pieces of an audio pipeline, but it collapses archive lookup,
SoundEntries resolution, coordinate conversion, distance admission, decoding, and OpenAL playback
into a path that is difficult to inspect. The first implementation phase will expose those stages
for every MCSE and area-music/ambience trigger relevant to the active world. Later phases will make
the camera an explicit actor, unify it with audio/collision/residency, and use the resulting
attribution to fix fog-bounded loading and WMO-internal doodad batching.

The plan is intentionally additive. Existing terrain, WMO, M2/MDX, LIT, DBC, and MPQ readers stay
the canonical owners. A visible camera model is not assumed to improve performance; the actor
contract is about one authoritative transform and lifecycle.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Existing `WowViewer.Core.*` libraries, DBCD schema loading, MPQ data
source, Silk.NET.OpenGL, existing OpenAL/NVorbis/NLayer audio path

**Storage**: User-selected client roots and runtime caches; no client data or soundbanks committed

**Testing**: Focused xUnit tests in `wow-viewer/tests/`, Debug build, then user-run configured
client/audio/render proof

**Target Platform**: Windows desktop first; cross-platform viewer build must remain compiling

**Project Type**: Desktop viewer/runtime with shared C# libraries and thin diagnostic UI

**Performance Goals**: Keep residency bounded by actor/fog/path leases; make near-field popping and
dense-WMO doodad cost attributable; compare controlled captures without claiming a target FPS until
the user's client-backed benchmark proves it

**Constraints**: Offline-capable, no hardcoded client paths, first-decade era/schema differences
must remain explicit, optional OpenAL/MIDI/DLS backends must fail visibly, no whole-map load

**Scale/Scope**: World maps with hundreds of tiles, thousands of placements, and dense WMO doodads;
the initial code slice is current-tile audio diagnostics, not a complete simulator/server

## Constitution Check

*GATE: passed before Phase 0 research; re-check after Phase 1 design.*

- **Repo Independence**: PASS. All code, tests, contracts, and docs stay under `wow-viewer/`.
- **Library-First**: PASS. Audio diagnostics extend shared runtime/data contracts; UI only renders
  results and does not parse MCSE/DBC/MPQ data itself.
- **Real-Data Validation**: PASS with a user-owned gate. Fixtures cover stage classification, while
  final archive/path/coordinate/audio proof records the configured client root, build, and hashes.
- **No Client Path Assumptions**: PASS. Client roots remain runtime configuration.
- **Format Ownership**: PASS. Existing readers are extended only where their public contract lacks
  raw/transformed diagnostics; no duplicate MCSE, SoundEntries, or MPQ parser is planned.
- **One Phase at a Time**: PASS. Phase 1 audio diagnostics must validate before actor/residency
  integration begins.
- **Memory/Documentation Discipline**: PASS. The owning spec and compact active-context/progress
  entries are updated with each phase.
- **Safety**: PASS. No proprietary client assets, executable, soundbank, or external project code
  enters the repository.

## Documentation (this feature)

```text
specs/148-world-simulator/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
└── tasks.md
```

## Source Code Structure

```text
src/core/WowViewer.Core.Runtime/
├── World/                 # CameraActor, session, residency and performance contracts
└── Audio/                 # Shared audio diagnostic/value contracts where appropriate
src/core/WowViewer.Core.IO/
├── Dbc/                   # Existing SoundEntries/area owners; provenance extensions only
└── Mpq/                   # Existing data-source owner; read provenance extension only
src/viewer/WoWViewer/
├── Audio/                 # Runtime stage collection and backend status
├── Terrain/               # Existing MCSE ownership and world-coordinate adapters
├── Rendering/             # Existing batch/residency owners and performance attribution
└── ViewerApp_Audio.cs     # Inspector rendering only
tests/
├── WowViewer.Core.Tests/  # Pure contract, coordinate, lease, and classification tests
└── WoWViewer.Tests/       # Viewer/runtime focused tests where existing project permits
```

**Structure Decision**: Keep shared contracts and format provenance in existing core/runtime
owners, keep MCSE collection in the Alpha/Standard terrain adapters, and keep ImGui presentation in
the viewer. Do not create a parallel simulator project or a second reader stack. Land small phases
so the existing viewer remains runnable after each phase.

## Phase Order and Stop Conditions

### Phase 1 — Audio truth surface

Add the diagnostic value contract and populate it from existing MCSE/SoundEntries/area-music paths.
Show raw and transformed coordinates, range admission, candidate paths, source/read/decode/backend
stages, and current-tile/WMO context. Add pure tests for every terminal state. Stop until focused
tests and a Debug build pass; user then runs one configured client and reports the diagnostic table.

### Phase 2 — Explicit camera actor

Introduce one authoritative actor transform/context seam and route existing camera input/path
playback through it. Audio listener, active-area lookup, collision queries, and residency requests
consume the same snapshot. Stop if any existing camera path or manual movement loses behavior.

### Phase 3 — Fog/path residency attribution

Unify fog coverage, path warmup, and inspection leases. Retain the union of justified tiles/objects,
show lease owners, and remove premature near-field unloads. Add fixed-selection tests without claiming
that the fog formula is correct for every era until real-client evidence is checked.

### Phase 4 — Batch and WMO doodad performance

Measure and then optimize shared preparation and instance submission across terrain and WMO-internal
doodads. Keep per-frame work flat and attributable. Compare controlled captures; do not replace the
working path based on a single FPS screenshot.

### Phase 5 — Optional audio backends and museum-session seams

Add a pluggable MIDI/DLS backend only after its native dependency and licensing/runtime behavior are
proven. Formalize local session/build provenance and future NPC/game-object/CPU-LLM seams without
turning this epic into a server implementation.

## First Implementation Slice

The first code slice is Phase 1 only:

1. Preserve the raw MCSE position before the existing coordinate transform and label both spaces.
2. Add a diagnostic projection in `WorldAudioRuntime` that resolves every relevant emitter without
   requiring successful OpenAL playback.
3. Add read/source provenance at the data-source boundary or an explicit “source unknown” state;
   do not infer that a catalog count proves `sounds.mpq` supplied a requested file.
4. Render a scrollable current-tile audio table in the existing audio panel.
5. Add focused tests for coordinate fields and each resolution/decode/backend terminal state.

This slice does not change the coordinate transform, add MIDI playback, make the camera visible, or
rewrite tile selection. Those require evidence from the diagnostic output and their later phase gates.

## Complexity Tracking

No constitution violations. The multi-phase runtime direction is deliberately split across existing
owners rather than introducing a new simulator project or duplicate data pipeline.
