# Implementation Plan: M2 and WMO Shader Permutation System

**Branch**: `198-m2-shader-permutations` | **Date**: 2026-09-01 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/198-m2-shader-permutations/spec.md`

## Summary

Replace the single hardcoded M2/WMO program with the client's indexed shader-pair selection.
Research established that selection is an **ordinal into two ordered name tables**
(`0x00eb4b10`, 35 pixel entries; `0x00eb4ba0`, 16 vertex entries) and decoded the pixel table
in full. What supplies the ordinal per batch is **not yet measured** and is the first
implementation task — everything else (registry, fallback, reporting, era gate) is built
first and is independent of it, so the unknown blocks only the selection step.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET OpenGL bindings; existing `WowViewer.Core.Renderer`
shader helpers (`TerrainShader`, `WmoShader` patterns), `IGpuInstancedModelRenderer`

**Storage**: N/A — no persisted artifacts. The selection report is in-memory per scene load.

**Testing**: xUnit (`WowViewer.Core.Tests`) for the registry, request validation, and report;
operator-run capture comparison for visual parity (SC-002, SC-005)

**Target Platform**: Windows desktop, OpenGL

**Project Type**: Desktop renderer inside an existing solution

**Performance Goals**: Reduce per-batch material setup CPU time in the reference scene
(SC-004). No frame-time regression on scenes that fall back entirely.

**Constraints**: Must not alter batching/culling (FR-008); must be switchable off to exact
pre-change output (FR-007, SC-005); must not touch 0.5.3/LK rendering (FR-006)

**Scale/Scope**: 16 x 35 = 560 possible pairs; the corpus determines the live subset.
Implementation is incremental and the unimplemented remainder is enumerable by construction.

## Constitution Check

*GATE: evaluated before Phase 0 and re-evaluated after Phase 1 design.*

| Principle | Status | Note |
|---|---|---|
| I. Repo Independence | **PASS** | All work inside `wow-viewer/`. No outside paths. |
| II. Library-First | **PASS** | Enums, `ShaderPermutation`, registry and report go in `WowViewer.Core.Renderer`; `M2Renderer`/`WmoRenderer` become consumers. No format reader is duplicated — the selector's source field is read by the existing M2 reader once T002 identifies it. |
| III. Real-Data Validation | **PASS** | SC-002/SC-005/SC-006 are all real-corpus gated. No mock-asset signoff. Visual proof is operator-run per the execution boundary. |
| IV. Evidence vs Architecture | **N/A** | No model training in this feature. |
| V. Streaming-First Dataset | **N/A** | No dataset pipeline involvement. |
| VI. No Client Path Assumptions | **PASS** | No client root referenced; the reference scene is named by configuration. |
| VII. Containers Are Inputs | **PASS** | Reads only; writes nothing. |

**Post-Phase-1 re-check**: unchanged. The design adds no new project, no new store, and no
new format owner. `PermutationRegistry` is the single owner of compiled programs, which
*reduces* the current duplication of inline GLSL across three renderer classes.

**No Complexity Tracking entries** — no violations to justify.

## Project Structure

### Documentation (this feature)

```text
specs/198-m2-shader-permutations/
├── plan.md              # This file
├── research.md          # Phase 0 — shader tables + the open selector question
├── data-model.md        # Phase 1 — entities and state transitions
├── quickstart.md        # Phase 1 — how to run and read the report
├── contracts/
│   └── permutation-registry.md
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 — speckit-tasks output, not created here
```

### Source Code

```text
wow-viewer/src/core/WowViewer.Core.Renderer/Shaders/
├── M2PixelShader.cs              # 35-value ordinal enum (research.md R2)
├── M2VertexShader.cs             # 16-value ordinal enum (T003)
├── WmoShader.cs                  # 6-value ordinal enum
├── ShaderPermutation.cs          # pair + native name + texture-unit count
├── PermutationRequest.cs         # resolved pair OR unresolved reason, never both
├── PermutationRegistry.cs        # compile-once cache, fallback, enable switch
└── PermutationSelectionReport.cs # per-scene counts and offender lists

wow-viewer/src/viewer/WoWViewer/Rendering/
├── M2Renderer.cs                 # consumes the registry; keeps its current program as fallback
└── WmoRenderer.cs                # same, US3

wow-viewer/tests/WowViewer.Core.Tests/
├── ShaderPermutationTests.cs     # enum ordinals pinned to the measured table
├── PermutationRegistryTests.cs   # compile-once, fallback, enable switch, state transitions
└── PermutationReportTests.cs     # counts, dedup, caps
```

**Structure Decision**: Core-library-first per Constitution II. The permutation types have no
GL dependency beyond the program handle, so they are testable without a GL context; the
renderers in `src/viewer/` are thin consumers. This mirrors how `TerrainShader` and
`WmoShader` already sit in `WowViewer.Core.Renderer`.

## Phasing

Each phase is independently valuable and independently verifiable, per the spec's user
stories. **Phase 2 is gated on Phase 1's T002 result** and must not begin before it.

| Phase | Delivers | Gate |
|---|---|---|
| **1. Tables and registry** | Ordinal enums pinned to the measured table, `ShaderPermutation`, registry with fallback and enable switch, report plumbing. Nothing selects yet — every batch falls back. | Unit tests green; renderer output pixel-identical to today (SC-005 in its trivial form) |
| **2. Selector provenance** | T002 resolves where the per-batch ordinal comes from; if it cannot be resolved, the phase ends with a documented negative result and the feature stops at Phase 1. | Measured evidence recorded in research.md R3, or an explicit "not resolved" |
| **3. Selection + era gate** | Batches resolve real permutations; unresolved and non-applicable eras fall back. Report populated. | SC-001, SC-006; 0.5.3 scene pixel-identical (FR-006) |
| **4. Combiner implementations** | Author programs for the permutations the corpus actually requests, highest-count first. | SC-002 operator capture per permutation implemented |
| **5. CPU measurement** | Before/after per-batch material CPU timings using spec 136's path. | SC-004 |
| **6. WMO set (US3)** | The six map-object programs. | US3 acceptance |

## Risks

- **T002 does not resolve.** Mitigated by phase ordering: Phases 1 and 3's fallback path are
  useful and shippable regardless, and the spec's FR-002 makes "unresolved" a legitimate,
  reportable outcome rather than a failure to paper over with a guess.
- **Shader count explodes.** Bounded by authoring only what the corpus requests (R5, SC-006)
  rather than the full 560-pair cross product.
- **Regressing the working 0.5.3 lane.** FR-006 era gate plus SC-005's pixel comparison, and
  Phase 1 ships with everything falling back so the gate is exercised from the start.
- **Conflating pass class with combiner.** R6 states these are orthogonal; data-model.md
  keeps `M2MaterialPassProfile` out of the permutation types deliberately.
