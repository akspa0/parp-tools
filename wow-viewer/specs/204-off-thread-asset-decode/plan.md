# Implementation Plan: Off-Thread Asset Decode

**Branch**: `204-off-thread-asset-decode` | **Date**: 2026-09-01 | **Spec**: [spec.md](spec.md)

## Summary

Split asset loading into **fetch → decode → upload**, run fetch and decode on workers, and leave
only GPU resource creation on the render thread. Then retire the unconditional oversized admission
that the frame budget currently needs, and fix the throughput throttle that only made sense while
one load could cost 68 ms.

The prize is bounded and known: `DeferredAssetLoads` owns 12 of the 13 recent hitches at 26.4–68.1
ms each, on a route whose median frame is 75.26 ms.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET OpenGL (single context, single thread); existing
`MpqDataSource` prefetch workers; `DeferredLoadBudget`

**Storage**: N/A

**Testing**: xUnit for the queue, budget and cancellation logic — all of which can be exercised
headless. Operator flights for frame-time and streaming-rate proof.

**Target Platform**: Windows desktop, OpenGL

**Project Type**: Renderer/asset-pipeline change inside an existing solution

**Performance Goals**: Remove `DeferredAssetLoads` from the hitch list (SC-001); at least 5x faster
cold-cache population (SC-003); p99 better than the recorded 235.37 ms (SC-004).

**Constraints**: single GL context (FR-001); no appearance change (FR-008); runtime toggle
(FR-009); deterministic completion still available for capture/harvest (FR-011)

**Scale/Scope**: 25,684 placements, 926 distinct resident models on the baseline route; two asset
kinds (MDX/M2, WMO) plus textures.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I. Repo Independence | **PASS** | all inside `wow-viewer/` |
| II. Library-First | **PASS** | one load pipeline, not a second parallel one. The synchronous path is retained only as a runtime toggle for comparison (FR-009) and is the *same* pipeline with the worker stage bypassed — not a fork. |
| III. Real-Data Validation | **PASS** | every success criterion is a real flight or a cold-cache timing |
| IV. Evidence vs Architecture | **N/A** (no model training) — in spirit: FR-012 forbids a single `DeferredAssetLoads` number that hides which of fetch/decode/upload is responsible, which is the attribution defect this spec had to reverse-engineer |
| V. Streaming-First Dataset | **N/A** |
| VI. No Client Path Assumptions | **PASS** |
| VII. Containers Are Inputs | **PASS** — MPQ read behaviour is unchanged |

**No Complexity Tracking entries.** The concurrency is genuinely required by the measurement; it is
not speculative generality.

## Project Structure

```text
specs/204-off-thread-asset-decode/
├── spec.md
├── research.md          # the two threads that already exist, the two that do not
├── plan.md              # this file
├── contracts/
│   └── asset-load-pipeline.md
├── checklists/
│   └── requirements.md
└── tasks.md
```

```text
wow-viewer/src/core/WowViewer.Core.Runtime/World/
├── DeferredLoadBudget.cs        # re-pointed at upload cost; oversized admission retired
├── AssetLoadRequest.cs          # NEW — key, kind, priority, cancellation
├── DecodedAssetPayload.cs       # NEW — CPU-side result, holds no GL handle
└── AssetLoadPipeline.cs         # NEW — headless queue/state machine, unit-testable

wow-viewer/src/viewer/WoWViewer/Terrain/
└── WorldAssetManager.cs         # drives the pipeline; upload stays here

wow-viewer/src/viewer/WoWViewer/Rendering/
├── ModelRenderer.cs             # constructor split: parse phase vs GPU-upload phase
├── M2Renderer.cs                # same
└── WmoRenderer.cs               # same

wow-viewer/tests/WowViewer.Core.Tests/
├── AssetLoadPipelineTests.cs
└── DeferredLoadBudgetTests.cs
```

**Structure Decision**: the queue/state machine goes in `Core.Runtime` so it is unit-testable
without a GL context, matching where `DeferredLoadBudget` and `WorldObjectPassCoordinator` already
live. The GL-touching upload stays in the viewer.

## Phasing

Each phase ends in something checkable. **Phase 1 is a hard gate**: nothing runs concurrently until
the thread-safety audit is done.

| Phase | Delivers | Gate |
|---|---|---|
| **0. Attribute** | Split the single `DeferredAssetLoads` timer into fetch / decode / upload (FR-012). Surface the budget counters. No behaviour change. | Three numbers where there was one; the decode share is measured, not assumed |
| **1. Audit** | Settle research.md R5. Every static and shared structure on the load path is either proven safe, made safe, or confined to the render thread. | A written list, each item with a verdict. **No concurrency before this exists.** |
| **2. Split construction** | `MdxRenderer` / `M2Renderer` / `WmoRenderer` gain a CPU-parse phase producing a payload with no GL handles, and a separate GPU-upload phase. Still called back-to-back on the render thread. | Behaviour and frame time unchanged; the seam exists and is tested |
| **3. Move decode off-thread** | Worker pool runs fetch+decode; render thread drains an upload queue under the frame budget. Runtime toggle (FR-009). | SC-001, SC-002, SC-005 |
| **4. Fix the throttle and the budget** | Retire the unconditional oversized admission; re-point the budget at upload; correct the R3 CPU throttle. | SC-002, SC-003 |
| **5. Textures out of the draw pass** | `ProcessDeferredTextureLoads` leaves `RenderGeosets`; texture decode joins the worker stage, upload joins the budgeted queue. | US3, SC-007 |
| **6. Prefetch follows the parse** | Group files and textures are requested once known. | US4 |
| **7. Era + determinism** | Verify 0.5.3, LK, Cata, MoP. Confirm capture/harvest still complete deterministically. | SC-004, SC-006, FR-011 |

**Why Phase 0 before the work.** The current `DeferredAssetLoads` number is fetch + decode + upload
in one figure, and R1 says fetch is probably already warm. If decode turns out to be a small share
of the 68 ms and *upload* is the bulk, then moving decode off-thread wins little and Phase 3 must be
re-scoped toward upload batching instead. That is a real possible outcome and Phase 0 is how it gets
caught before the concurrency is written.

**Why Phase 1 is a gate.** R5 found GL objects on static fields, a process-global diagnostic writer
re-opened per model from a constructor, and unaudited caches. Running that concurrently would
produce intermittent, non-reproducible corruption — the worst possible failure mode in a renderer,
and one that would be blamed on the batching work landing alongside it.

## Risks

- **Decode is not the dominant share.** Mitigated by Phase 0 measuring fetch/decode/upload before
  Phase 3 is written; the phase table has an explicit re-scope path.
- **Latent thread-unsafety in the parse layer.** The main risk. Phase 1 is a gate, not a step, and
  Phase 2 keeps everything on one thread so the seam can be validated before it is crossed.
- **Intermittent corruption is hard to attribute.** FR-009's runtime toggle exists so a suspected
  regression can be bisected in-session rather than across builds.
- **Upload spikes replace decode spikes.** Phase 4 re-points the budget at upload specifically; if
  a single upload still exceeds a frame, that is a smaller and more tractable problem (buffer
  sub-uploads) than a 68 ms decode.
- **Capture/harvest regressions.** These paths need completion, not streaming. FR-011 and Phase 7
  keep a deterministic drain available.
- **Eviction races with in-flight loads.** Covered by the contract's cancellation rules; the edge
  cases are enumerated in spec.md rather than discovered later.
