# Implementation Plan: Unified Model Batching

**Branch**: `202-unified-model-batching` | **Date**: 2026-09-01 | **Spec**: [spec.md](spec.md)

## Summary

Move batchability from "a property each renderer opts into" to "one decision taken from render
state", get per-model GPU instancing working on every route and era, and fix the two metric
conflations that currently make any before/after unreadable. Sequenced so that each phase is
measurable before the next begins.

The prize is quantified but **not assumed**: per-model instancing converges opaque draws on the
number of distinct visible models. If that is ~429 against 13,180 instances it is a ~30x
reduction; T001 measures it before a line of batching code is written.

## Technical Context

**Language/Version**: C# / .NET 10

**Primary Dependencies**: Silk.NET OpenGL; existing `IGpuInstancedModelRenderer`,
`WorldObjectPassCoordinator`, `M2RouteType`

**Storage**: N/A

**Testing**: xUnit for the decision logic and route planning (headless — the coordinator is
already in `Core.Runtime` and testable without GL); operator flights for frame-time proof

**Target Platform**: Windows desktop, OpenGL

**Project Type**: Renderer change inside an existing solution

**Performance Goals**: The stated problem is **sub-1 FPS in most areas of 5.0.1 terrain**, and
84.44 ms median / 7 FPS on the recorded flight. Target is a measurable improvement with the
residual explained, not a specific number pulled from the air.

**Constraints**: no appearance change (FR-004); no draw-order change (FR-005); runtime toggle
(FR-006); 0.5.3 frozen (FR-008)

**Scale/Scope**: 13,180 opaque + 260 transparent instances on the baseline route; five
`M2RouteType` values; four eras.

## Constitution Check

| Principle | Status | Note |
|---|---|---|
| I. Repo Independence | **PASS** | all inside `wow-viewer/` |
| II. Library-First | **PASS — restores it.** The batch decision moves into `Core.Runtime` beside `WorldObjectPassCoordinator`; renderers become executors. Spec 153 US3's mechanism is absorbed, not paralleled. |
| III. Real-Data Validation | **PASS** | every success criterion is a real flight or capture |
| IV. Evidence vs Architecture | **N/A** (no model training) — in spirit: FR-003 forbids an aggregate "batched" number hiding a path that never batches, which is the R1 defect |
| V. Streaming-First Dataset | **N/A** |
| VI. No Client Path Assumptions | **PASS** |
| VII. Containers Are Inputs | **PASS** |

**No Complexity Tracking entries.**

## Project Structure

```text
specs/202-unified-model-batching/
├── spec.md
├── research.md          # the two conflations, the three gates, the floor
├── plan.md              # this file
├── contracts/
│   └── batch-decision.md
└── tasks.md
```

```text
wow-viewer/src/core/WowViewer.Core.Runtime/World/Passes/
├── ModelBatchKey.cs             # the state that makes two draws combinable
├── ModelBatchDecision.cs        # batched / unbatched-by-state / unbatchable+reason
├── ModelBatchPlanner.cs         # ONE decision point, headless, testable
└── WorldObjectPassCoordinator.cs # consumes the planner

wow-viewer/src/viewer/WoWViewer/Rendering/
├── IGpuInstancedModelRenderer.cs # gains fade in the instance payload (R3, gate 3)
├── M2Renderer.cs                 # stops deciding; executes
├── ModelRenderer.cs              # same
└── WmoRenderer.cs                # doodad submissions follow the same planner

wow-viewer/tests/WowViewer.Core.Tests/
├── ModelBatchPlannerTests.cs
└── ModelBatchKeyTests.cs
```

**Structure Decision**: the planner lives in `Core.Runtime` because
`WorldObjectPassCoordinator` already does — which means the decision is unit-testable without a
GL context, and the same planner serves the defunct-app bridge that already calls
`ExecutePlannedOpaqueMdx`.

## Phasing

Each phase ends in a number. **No phase starts before the previous one's measurement exists.**

| Phase | Delivers | Gate |
|---|---|---|
| **0. Preconditions** | Spec 201's per-path attribution, plus R1's split of *instanced* from *state-hoisted* in the counters. Nothing else changes. | Baseline re-flown; batched/instanced/unbatched/unbatchable are four distinct numbers |
| **1. Measure the ceiling** | Distinct visible models per frame; the share of instances blocked by each of R3's three gates. | The theoretical floor is a number, and the largest gate is named |
| **2. One decision point** | `ModelBatchKey` + `ModelBatchDecision` + `ModelBatchPlanner`. Renderers stop deciding. Behaviour deliberately unchanged. | Unit tests green; a flown route is frame-time-identical and appearance-identical |
| **3. Close the gates** | Per-model instancing for every route (absorbs spec 201 Phase 2's native batch key); fade carried in the instance payload so faded instances stop dropping out. | SC-001: opaque unbatched falls toward the Phase 1 floor; residual attributed |
| **4. Era coverage** | Verify and fix per era: 0.5.3, LK, Cata, MoP. | SC-003, SC-004, SC-005 |
| **5. Transparent** | Batch transparent where sort order permits. | US3 |
| **6. Cross-model (optional)** | Only if Phase 1 showed the per-model floor is still too high. | Re-measured need, or explicitly dropped |

**Why Phase 0 and 1 come before any batching work.** The recorded frame says *0 batched /
13,180 unbatched*, and R4 establishes that this is ambiguous between "the toggle was off" and
"no route can batch". Building against an ambiguous baseline is how spec 153 US3 came to be
believed a complete fix. Phase 1 also prevents the opposite error — spending weeks on
cross-model batching when per-model instancing already gets to ~429 draws.

## Risks

- **The win is smaller than hoped.** Mitigated by Phase 1 measuring the floor first; SC-001
  requires the residual to be explained, so a modest result is reportable rather than a failure.
- **Appearance changes under instancing.** Per-instance state that is not in the batch key is
  exactly how this happens. FR-002 requires anything excluded from the key to be justified in
  writing, and Phase 2 ships behaviour-identical so the key is validated before it is relied on.
- **Fade in the instance payload changes blending.** Phase 3 gates on capture comparison, and
  the toggle (FR-006) keeps the old path reachable in-session.
- **Regressing 0.5.3.** FR-008 plus SC-005's pixel comparison; 0.5.3 is verified in Phase 4 as
  its own gate rather than assumed to come along.
- **Scope creep into cross-model batching.** Phase 6 is explicitly optional and gated on a
  re-measured need.
