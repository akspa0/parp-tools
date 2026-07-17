# Tasks: Minimap Lighting Calibration and Lighting-Aware Terrain Reconstruction

**Input**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md),
[contracts/minimap-lighting-calibration-contract.md](contracts/minimap-lighting-calibration-contract.md),
[quickstart.md](quickstart.md)

## Dependencies

```text
Setup + Foundational (extended MinimapLightingProvenance contract)
  -> US1 Determine each authored minimap's lighting condition
    -> US2 Rebalance synthetic training lighting to match reality
      -> US3 Retrain and evaluate (execution gated on explicit user go-ahead)
```

US1 is the independently valuable MVP: it produces a trustworthy real-dataset lighting-bucket
distribution even if US2/US3 never run. US2 depends on US1's real output. US3 depends on US2's
rebalanced plan and is additionally gated on a separate, in-session user authorization before any
training command executes.

## Phase 1: Setup

- [x] T001 Create file skeletons per `plan.md` Project Structure: `src/core/WowViewer.Core.IO/Maps/MinimapShadingMatch.cs`, `data-harvester/src/harvester/spec111/__init__.py`, `data-harvester/src/harvester/spec111/lighting_buckets.py`, `data-harvester/src/harvester/spec111/rebalance_lighting_variants.py`, `data-harvester/tests/spec111/__init__.py`

## Phase 2: Foundational (extend the shared provenance contract)

**⚠️ CRITICAL**: No user story work can begin until this phase is complete -- every story reads or
writes the extended `MinimapLightingProvenance` record.

- [x] T002 Add the additive shading-match fields (`ShadingMatchStatus`, `ShadingMatchedTimeOfDayHours`, `ShadingMatchConfidence`, `ShadingMatchEvidence`, `ShadingMatchExcludedMcshFraction`, `ShadingMatchBuildFingerprint`) from `data-model.md` to `src/core/WowViewer.Core/Maps/MinimapLightingProvenance.cs`, preserving every existing field/behavior unchanged
- [x] T003 [P] Add focused coverage in `tests/WowViewer.Core.Tests/MinimapLightingProvenanceTests.cs` pinning the new fields' default/not-evaluated states, without changing any existing tint-path assertion

**Checkpoint**: extended provenance record ready; the shading-match scorer (US1) can now target it.

## Phase 3: User Story 1 - Determine each authored minimap's lighting condition (Priority: P1) 🎯 MVP

**Goal**: Every 0.5.3.3368 dataset tile with both an authored minimap and ground-truth terrain
receives a shading-based lighting-bucket label or an explicit not-evaluated/low-confidence status,
using the production `TerrainMinimapCompositor`/`TerrainSolarDirection` path exclusively.

**Independent Test**: Run the bounded `minimap-lighting-calibrate --limit N` pass from
`quickstart.md`, inspect a handful of `matched` tiles by eye against their winning candidate, and
confirm the per-build distribution report's counts reconcile exactly.

### Tests for User Story 1

- [x] T004 [P] [US1] Add `tests/WowViewer.Core.Tests/MinimapShadingMatchTests.cs` covering: the candidate sweep renders exclusively through `TerrainMinimapCompositor`; the score is tint-independent (a same-shape, different-material-color fixture must not change the score); MCSH-correlated regions are excluded before scoring; near-tied candidates yield `low_confidence_ambiguous`; near-flat/low-signal terrain yields `low_confidence_flat_terrain`; a non-0.5.3.3368 build fingerprint yields `not_evaluated` without rendering any candidate (9/9 passing, including empirically-tuned thresholds verified against real compositor output, not just hand-picked numbers)

### Implementation for User Story 1

- [x] T005 [US1] Implement the candidate sweep and value-correlation scorer in `src/core/WowViewer.Core.IO/Maps/MinimapShadingMatch.cs`, rendering candidates only through `TerrainMinimapCompositor`/`TerrainSolarDirection` per `contracts/minimap-lighting-calibration-contract.md` §Shading-match inference contract (depends on T002)
- [x] T006 [US1] `MinimapShadingMatch.Evaluate` takes and returns the extended `MinimapLightingProvenance` directly (`with`-expression chaining), so wiring is the call site itself (depends on T005)
- [x] T007 [US1] Chain `MinimapShadingMatch.Evaluate` onto the existing `AnalyzeAuthoredMinimapLighting` tint-based `Infer()` call in `tools/harvest/WowViewer.Tool.Harvest/Program.cs` (both Full/V22 call sites now pass `buildVersion` through), reusing the existing tile-iteration and C#-to-Python length-prefixed streaming pathway into additive Zarr fields (no NPZ, no new parallel command) (depends on T006)
- [x] T008 [US1] Add `data-harvester/scripts/report_lighting_buckets.py` (thin CLI over `harvester/spec111/lighting_buckets.py`) producing the per-map and overall `LightingBucketDistributionReport` from the streamed Zarr fields, enforcing the `sum(BucketCounts) + NotEvaluatedCount + LowConfidenceCount == TotalEligibleTiles` validation rule from `data-model.md`; pre-spec-111 tiles missing the field entirely are surfaced as `tiles_without_shading_match_field`, never folded into not-evaluated
- [ ] T009 [US1] Code proof complete: focused C# sweep (`MinimapShadingMatchTests` + `MinimapLightingProvenanceTests` + `TerrainMinimapCompositorTests` + `TerrainSolarDirectionTests`) 42/42; Debug Harvest build 0 errors; `tests/spec111/` Python suite 16/16. **Remaining user-run proof**: one bounded real-0.5.3.3368 `harvest-stream --stream-profile v22` pass, the quickstart side-by-side eyeball check on a handful of `matched` tiles, then the whole-build pass and reconciled distribution report

**Checkpoint**: User Story 1 is independently complete -- every eligible 0.5.3.3368 tile has a trustworthy shading-match status and the distribution report is available.

## Phase 4: User Story 2 - Rebalance synthetic training lighting to match reality (Priority: P2)

**Goal**: Synthetic-lighting-variant sampling for training matches the real observed 0.5.3.3368
lighting distribution from User Story 1, and the drifted direction reimplementation found in
`terrain_lighting.py` no longer runs independently of the corrected production model.

**Independent Test**: Run the quickstart rebalancing `--dry-run` and confirm the printed per-bucket
weights match the Phase 3 distribution report, `no_real_baseline` buckets are flagged rather than
fabricated, and the existing leak-safety tags are unchanged.

### Tests for User Story 2

- [x] T010 [P] [US2] Add `data-harvester/tests/spec111/test_lighting_bucket_rebalancing.py` covering: weight computation from a synthetic `LightingBucketDistributionReport`; `no_real_baseline` flagging for zero-coverage buckets without fabricated weight; exact largest-remainder variant allocation; and the structural leak-safety guarantee (the rebalancer only ever sees aggregate bucket counts, never per-tile grouping fields)

### Implementation for User Story 2

- [x] T011 [US2] Corrected the drifted solar-direction reimplementation in `data-harvester/src/harvester/spec103/terrain_lighting.py`: `_terrain_solar_direction` is now a documented value-for-value port of the corrected C# `TerrainSolarDirection.Evaluate` (positive-X north lock, fixed north-west bearing, elevation-only cycling), with a regression test pinning both corrections; a true streamed (not ported) architecture remains a labeled follow-up in the function's docstring
- [x] T012 [P] [US2] Implement `data-harvester/src/harvester/spec111/lighting_buckets.py` ingesting the streamed shading-match fields and producing the reconciled `LightingBucketDistributionReport` (library owner; `scripts/report_lighting_buckets.py` is its thin wrapper)
- [x] T013 [US2] Implement `data-harvester/src/harvester/spec111/rebalance_lighting_variants.py` producing a `RebalancedTrainingSamplingPlan` plus a `rebalanced_lighting_times` list that feeds the existing `spec103_build_synthetic_store.py` `lighting_times=` entry point unchanged, and `scripts/rebalance_lighting_variants.py` as its thin `--dry-run`-capable CLI
- [x] T014 [US2] Input-contract check: `rebalanced_lighting_times` emits bare normalized floats only (pinned by test), so the rebalanced training data structurally cannot carry a lighting-bucket label, status, or confidence into the model's input path -- only the same game_time float the arbitrary sweep always used
- [x] T015 [US2] Focused Python proof: `tests/spec111/` 16/16 passed; `tests/spec103/test_terrain_lighting.py` 10/10 passed (including the new fixed-bearing regression); all three spec111 CLIs import and parse; the training gate smoke-run validated a fixture plan and refused to train without `--confirm-run`

**Checkpoint**: User Stories 1 and 2 both work independently -- rebalanced sampling plan is ready and leak-safe.

## Phase 5: User Story 3 - Retrain and evaluate against the current checkpoint (Priority: P3)

**Goal**: Determine whether training on the rebalanced dataset actually improves the deployed
reconstruction model's real-world robustness, without ever promoting a regression and without ever
launching GPU/cloud compute without explicit authorization.

**Independent Test**: The prepared training config and evaluation harness can be reviewed and dry-run
(config validation only, no GPU execution) independent of whether execution is ever authorized; once
authorized and run, the checkpoint comparison must show a recorded `Outcome` before any promotion.

### Implementation for User Story 3

- [x] T016 [US3] Confirmed at implementation time (2026-07-17): Spec 102 remains BLOCKED on its M0 target reharvest, so the active, unblocked stage is Spec 108 `WdlPriorNet` (`scripts/train_spec103_wdl_prior.py`). No baseline checkpoint is committed to the repo -- it is a user-run artifact, so `--baseline-checkpoint` is a required explicit path that fails closed when absent
- [x] T017 [US3] Implement `data-harvester/scripts/train_spec111_reconstruction.py`: validates the rebalanced plan (bare-float lighting_times, leak-safety assertion, existing store/baseline), prints the exact delegated `train_spec103_wdl_prior.py` command, and refuses to start any GPU run without `--confirm-run`
- [x] T018 [US3] Implement `data-harvester/src/harvester/spec111/checkpoint_comparison.py` producing the `ReconstructionCheckpointComparison` record; `promotion_decision` is True only for a clear improvement -- both regressed and inconclusive outcomes keep the deployed checkpoint (5/5 focused tests)
- [ ] T019 [US3] **STOP: confirm with the user before executing.** Only after explicit, separate authorization at the point of execution (`--confirm-run`), run the training pass (T017) and the evaluation (T018), and record the outcome per `contracts/minimap-lighting-calibration-contract.md` §Training/evaluation execution contract. Depends on the user-run portions of T009 (real bucketing pass) and a real rebalanced store

**Checkpoint**: all three user stories independently functional; go/no-go recorded for any executed training run.

## Phase 6: Polish and continuity

- [x] T020 [P] Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` with this feature's outcome, per the constitution's Memory Bank Discipline
- [x] T021 Update task states and exact proof commands in `specs/111-minimap-lighting-calibration/tasks.md` as each phase's checkpoint is reached

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies.
- **Foundational (Phase 2)**: depends on Setup; blocks every user story.
- **US1 (Phase 3)**: depends on Foundational only. This is the MVP -- valuable and shippable on its own.
- **US2 (Phase 4)**: depends on US1's real distribution report existing (T009's output), not just its code.
- **US3 (Phase 5)**: depends on US2's rebalanced plan (T013's output); its final execution task (T019) additionally depends on a separate, explicit user go-ahead that is not satisfied by completing any other task.

### Parallel Opportunities

- T003 and T004 (test scaffolding) can be written in parallel with each other once T002 lands.
- T012 (bucket ingestion) can be implemented in parallel with T011 (retiring the drifted module), since neither depends on the other's output, though both must land before T013.
- T020 can run in parallel with T021.

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) and Phase 2 (Foundational).
2. Complete Phase 3 (US1): shading-match inference and the real 0.5.3.3368 distribution report.
3. **STOP and VALIDATE**: eyeball the bounded run's `matched` tiles against real minimaps, exactly as
   today's real-vs-synthesized side-by-side caught the original sun-direction bug.
4. The distribution report alone is a useful, shippable artifact even before US2/US3 exist.

### Incremental Delivery

1. Setup + Foundational -> extended provenance contract ready.
2. US1 -> real lighting-bucket distribution (MVP, independently valuable).
3. US2 -> rebalanced, leak-safe synthetic-lighting-variant sampling, and the drifted Python
   reimplementation retired.
4. US3 -> retrain-and-evaluate, gated on explicit authorization at the execution step (T019), with a
   regression always keeping the current deployed checkpoint.
