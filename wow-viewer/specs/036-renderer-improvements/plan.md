# Implementation Plan: Renderer Improvements Convergence

**Branch**: `036-renderer-improvements` | **Date**: 2026-06-01 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/036-renderer-improvements/spec.md`

## Summary

Create a single renderer-improvements owner plan that converges the active work from specs 030, 031, and 032 into one dependency-ordered modernization roadmap for `wow-viewer`. The convergence plan keeps library ownership explicit, preserves the old specs as source slices, and sequences implementation around lighting/fog foundations, terrain topology, WMO pass architecture, runtime render pipelines, and thin viewer-host integration.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: WowViewer.Core.IO, WowViewer.Core.Runtime, WowViewer.App, Silk.NET.OpenGL

**Storage**: Markdown feature-pack artifacts only for this planning slice; runtime work later remains library/runtime code

**Testing**: `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug`; staged-client visual/runtime validation under `I:\parp\parp-tools\output\tmp\wowarchive-clients\`

**Target Platform**: Windows desktop viewer host with library-first runtime ownership

**Project Type**: Library + desktop viewer host

**Performance Goals**: Keep convergence phases aligned to eventual 60 FPS world rendering goals while deferring exact shader/runtime implementation proof to later execution phases

**Constraints**: No code outside `wow-viewer/`; `gillijimproject_refactor` remains read-only reference; one phase at a time; max 10 steps per phase; staged-client proof required before phase completion

**Scale/Scope**: Planning convergence across 3 source specs, 5 implementation phases, and shared validation boundaries for terrain, WMO, lighting, sky/fog, liquid, and viewer wiring

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Repo Independence | PASS | All implementation targets remain inside `wow-viewer/` |
| II. Library-First | PASS | Core.IO and Core.Runtime remain canonical owners; app host stays thin |
| III. Real-Data Validation | PASS | Every execution phase uses staged-client proof |
| IV. Residual Model Chain | N/A | No ML model training work in this convergence feature |
| V. Streaming-First | N/A | No dataset pipeline work |
| VI. No Game Client Path Assumptions | PASS | Staged client roots only |
| Read-Only Reference | PASS | Source behavior may be read from `gillijimproject_refactor` but not implemented there |
| One Phase at a Time | PASS | Convergence phases are ordered and independently validatable |
| Bite-Sized Plans | PASS | No phase exceeds 10 steps |

## Project Structure

### Documentation (this feature)

```text
specs/036-renderer-improvements/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   ├── README.md
│   ├── renderer-capability-slice.schema.json
│   └── renderer-validation-scenario.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
wow-viewer/src/core/WowViewer.Core.IO/
├── Maps/                     # Terrain-cell and topology readers from spec 031
└── Lighting/                 # LIT and lighting readers from spec 032

wow-viewer/src/core/WowViewer.Core.Runtime/
├── World/
│   ├── Terrain/              # Terrain topology, LOD, shadow, cell-aware state
│   ├── Wmo/                  # WMO dispatch, lightmap split, interior fog, batch flags
│   ├── Liquid/               # Water vs magma, animation, lighting-fed state
│   ├── Lighting/             # CurrentLight evaluation, local-light selection
│   ├── Sky/                  # Sky dome and clear color
│   ├── Fog/                  # Exterior/interior/WMO-area fog state
│   └── Passes/               # Frame-pass coordination
└── Rendering/                # Runtime render-pipeline families

wow-viewer/src/viewer/WowViewer.App/
└──                           # Thin viewer wiring: toggles, diagnostics, validation surfaces
```

**Structure Decision**: Preserve the existing library-first split and converge the three source plans by phase order, not by flattening everything into one renderer class or one viewer-owned feature surface.

## Implementation Phases

### Phase 1 — Ownership and Lighting Foundation
**Goal**: Establish the first executable modernization owner slices: lighting-state evaluation, sky/fog ownership, and documentation boundaries that all later terrain/WMO/liquid work depends on.

**Dependencies**: None.

**Approach**:
1. Treat the convergence feature as the owner plan while 030-032 remain source slices.
2. Land lighting-state and day/night evaluation first because terrain shading, WMO fog, water tint, and sky all depend on the same source of truth.
3. Keep this phase library-first and proof-oriented before tackling geometry or pass rewrites.

**Steps**:
1. Map source-plan sections from specs 030-032 to convergence phases and keep that mapping in the feature docs.
2. Port or define lighting-state contracts under `WowViewer.Core.IO` and `WowViewer.Core.Runtime`.
3. Define day/night evaluation and fog-source ownership in runtime lighting/fog surfaces.
4. Define sky-dome and clear-color ownership under runtime sky surfaces.
5. Define staged-client proof cases for noon, dusk, and night lighting states.
6. Validate lighting evaluation outputs against staged-client-driven expectations before moving on.

---

### Phase 2 — Terrain Cell and Topology Runtime Foundation
**Goal**: Converge the terrain data/topology work from spec 031 into the runtime modernization path required by later terrain rendering and LOD phases.

**Dependencies**: Phase 1.

**Approach**:
1. Keep `Core.IO` as the canonical owner for 145-vertex terrain-cell decoding.
2. Promote runtime cell/topology consumption only after the I/O contract is stable.
3. Treat hole-mask behavior and cell-level addressing as prerequisites, not polish.

**Steps**:
1. Define terrain cell, face-plane, and hole-mask contracts in `Core.IO`.
2. Define runtime terrain chunk/state surfaces that consume those contracts without re-decoding raw file data.
3. Sequence cell-aware spatial queries before LOD or render optimizations that depend on them.
4. Define native-accurate versus reconstructive hole handling as an explicit runtime choice.
5. Validate vertex counts, face-plane counts, and cell-address results on staged clients.
6. Validate runtime mesh/topology parity on a known terrain sample before moving on.

---

### Phase 3 — WMO Pass Architecture and Interior/Exterior Split
**Goal**: Converge the WMO pass architecture from spec 030 into the runtime modernization plan with correct dispatch, batch flags, lightmap split, and interior fog ownership.

**Dependencies**: Phase 1.

**Approach**:
1. Use the WMO architecture doc as source-of-truth input.
2. Keep pass dispatch, per-batch flags, and fog/lightmap selection in runtime library code.
3. Treat portal-walk, skip groups, and always-render groups as first-class routing behavior.

**Steps**:
1. Define WMO group dispatch ownership under runtime WMO surfaces.
2. Define per-batch MOMT flag evaluation and lightmap pass-selection surfaces.
3. Define interior fog and WMO-area fog ownership against the shared lighting/fog foundation.
4. Sequence liquid-in-WMO dispatch only after group dispatch semantics are stable.
5. Validate interior and exterior sample WMOs with staged-client comparison evidence.
6. Validate skip-group and always-render behavior before declaring the phase complete.

---

### Phase 4 — Terrain, Liquid, and World Render Pipelines
**Goal**: Converge the native-renderer-parity execution slices into bounded runtime pipelines for terrain layers, liquid routing, shadow overlays, and distance LOD.

**Dependencies**: Phases 1, 2, and 3.

**Approach**:
1. Build terrain, liquid, and frame-pass orchestration on top of the lighting and topology foundations.
2. Keep water animation, type dispatch, and shadow overlays as explicit runtime pipeline concerns.
3. Avoid viewer-host ownership drift by keeping orchestration in `Core.Runtime`.

**Steps**:
1. Define terrain layer pipeline ownership, including per-layer state and shadow overlay behavior.
2. Define terrain LOD selection and far-terrain ownership against the topology foundation.
3. Define liquid type dispatch, animation state, and lighting-fed water color ownership.
4. Define frame-pass coordination boundaries between terrain, WMO, liquid, and scene-level fog/sky work.
5. Validate close, mid, and far terrain behavior plus interior/exterior water on staged clients.
6. Record known parity gaps that remain after the bounded runtime pipeline pass.

---

### Phase 5 — Viewer Host Integration, Diagnostics, and Signoff
**Goal**: Add only the thin `WowViewer.App` wiring needed to drive the converged runtime capabilities, validation surfaces, and operator diagnostics.

**Dependencies**: Phases 1 through 4.

**Approach**:
1. Keep the app host thin: toggles, time-of-day controls, proof capture surfaces, and diagnostics only.
2. Reuse runtime-owned contracts rather than duplicating policy in the viewer.
3. Use staged-client proof surfaces to close the signoff loop.

**Steps**:
1. Add thin viewer controls for time-of-day, debug enables, and renderer validation surfaces.
2. Add diagnostics that expose active terrain/WMO/liquid/lighting state from runtime-owned contracts.
3. Define capture and screenshot proof flows for renderer parity review.
4. Validate that viewer host wiring does not become a second owner of renderer policy.
5. Run end-to-end staged-client validation across representative terrain and WMO samples.
6. Record final convergence evidence and remaining deferred work.

## Complexity Tracking

No constitution violations currently require justification. The main complexity control is scope: spec 035 M2 recovery remains adjacent and must not be silently absorbed into this convergence feature except where shared scene-proof surfaces overlap.
