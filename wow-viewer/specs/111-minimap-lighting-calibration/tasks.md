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

- [ ] T001 Create file skeletons per `plan.md` Project Structure: `src/core/WowViewer.Core/Maps/MinimapShadingMatch.cs`, `data-harvester/src/harvester/spec111/__init__.py`, `data-harvester/src/harvester/spec111/lighting_buckets.py`, `data-harvester/src/harvester/spec111/rebalance_lighting_variants.py`, `data-harvester/tests/spec111/__init__.py`

## Phase 2: Foundational (extend the shared provenance contract)

**⚠️ CRITICAL**: No user story work can begin until this phase is complete -- every story reads or
writes the extended `MinimapLightingProvenance` record.

- [ ] T002 Add the additive shading-match fields (`ShadingMatchStatus`, `ShadingMatchedTimeOfDayHours`, `ShadingMatchConfidence`, `ShadingMatchEvidence`, `ShadingMatchExcludedMcshFraction`, `ShadingMatchBuildFingerprint`) from `data-model.md` to `src/core/WowViewer.Core/Maps/MinimapLightingProvenance.cs`, preserving every existing field/behavior unchanged
- [ ] T003 [P] Add focused coverage in `tests/WowViewer.Core.Tests/MinimapLightingProvenanceTests.cs` pinning the new fields' default/not-evaluated states, without changing any existing tint-path assertion

**Checkpoint**: extended provenance record ready; the shading-match scorer (US1) can now target it.

## Phase 3: User Story 1 - Determine each authored minimap's lighting condition (Priority: P1) 🎯 MVP

**Goal**: Every 0.5.3.3368 dataset tile with both an authored minimap and ground-truth terrain
receives a shading-based lighting-bucket label or an explicit not-evaluated/low-confidence status,
using the production `TerrainMinimapCompositor`/`TerrainSolarDirection` path exclusively.

**Independent Test**: Run the bounded `minimap-lighting-calibrate --limit N` pass from
`quickstart.md`, inspect a handful of `matched` tiles by eye against their winning candidate, and
confirm the per-build distribution report's counts reconcile exactly.

### Tests for User Story 1

- [ ] T004 [P] [US1] Add `tests/WowViewer.Core.Tests/MinimapShadingMatchTests.cs` covering: the candidate sweep renders exclusively through `TerrainMinimapCompositor`; the score is tint-independent (a same-shape, different-material-color fixture must not change the score); MCSH-correlated regions are down-weighted before scoring; near-tied candidates yield `low_confidence_ambiguous`; near-flat terrain yields `low_confidence_flat_terrain`; a non-0.5.3.3368 build fingerprint yields `not_evaluated` without rendering any candidate

### Implementation for User Story 1

- [ ] T005 [US1] Implement the candidate sweep and directional-structure scorer in `src/core/WowViewer.Core/Maps/MinimapShadingMatch.cs`, rendering candidates only through `TerrainMinimapCompositor`/`TerrainSolarDirection` per `contracts/minimap-lighting-calibration-contract.md` §Shading-match inference contract (depends on T002)
- [ ] T006 [US1] Wire `MinimapShadingMatch` output into the extended `MinimapLightingProvenance` fields (depends on T005)
- [ ] T007 [US1] Add a `minimap-lighting-calibrate` command to `tools/harvest/WowViewer.Tool.Harvest/Program.cs` that iterates the configured 0.5.3.3368 dataset's eligible tiles, invokes the scorer, and streams results through the existing C#-to-Python length-prefixed protocol into additive Zarr fields (no NPZ) (depends on T006)
- [ ] T008 [US1] Add `data-harvester/scripts/report_lighting_buckets.py` producing the per-map and overall `LightingBucketDistributionReport` from the streamed Zarr fields, enforcing the `sum(BucketCounts) + NotEvaluatedCount + LowConfidenceCount == TotalEligibleTiles` validation rule from `data-model.md`
- [ ] T009 [US1] Run the focused C# tests (T003, T004), a bounded real-0.5.3.3368-client `minimap-lighting-calibrate --limit N` pass, and the quickstart eyeball check on a handful of `matched` tiles before removing `--limit`

**Checkpoint**: User Story 1 is independently complete -- every eligible 0.5.3.3368 tile has a trustworthy shading-match status and the distribution report is available.

## Phase 4: User Story 2 - Rebalance synthetic training lighting to match reality (Priority: P2)

**Goal**: Synthetic-lighting-variant sampling for training matches the real observed 0.5.3.3368
lighting distribution from User Story 1, and the drifted direction reimplementation found in
`terrain_lighting.py` no longer runs independently of the corrected production model.

**Independent Test**: Run the quickstart rebalancing `--dry-run` and confirm the printed per-bucket
weights match the Phase 3 distribution report, `no_real_baseline` buckets are flagged rather than
fabricated, and the existing leak-safety tags are unchanged.

### Tests for User Story 2

- [ ] T010 [P] [US2] Add `data-harvester/tests/spec111/test_lighting_bucket_rebalancing.py` covering: weight computation from a synthetic `LightingBucketDistributionReport`; `no_real_baseline` flagging for zero/near-zero-coverage buckets; `source_group_id`/`lighting_variant_id` tag preservation; rejection of a plan that would alter tagging

### Implementation for User Story 2

- [ ] T011 [US2] Retire the drifted solar-direction reimplementation in `data-harvester/src/harvester/spec103/terrain_lighting.py` per `research.md`'s "retire the drifted Python sweep" decision, re-labeling its remaining non-direction responsibilities (color/fog interpolation, MCSH bake authoring) so they cannot be mistaken for a second lighting-direction source of truth
- [ ] T012 [P] [US2] Implement `data-harvester/src/harvester/spec111/lighting_buckets.py` to ingest a `LightingBucketDistributionReport` (depends on T008's output shape)
- [ ] T013 [US2] Implement `data-harvester/src/harvester/spec111/rebalance_lighting_variants.py` producing a `RebalancedTrainingSamplingPlan` and wire its weights into the existing synthetic-lighting-variant generation entry point (depends on T011, T012)
- [ ] T014 [US2] Add an explicit input-contract check confirming rebalanced training rows carry the lighting-bucket label only as a sampling/metadata signal, never as a field the model consumes at input time, per `contracts/minimap-lighting-calibration-contract.md` §Rebalancing contract
- [ ] T015 [US2] Run the focused Python tests (T010) and the quickstart `--dry-run` rebalancing check

**Checkpoint**: User Stories 1 and 2 both work independently -- rebalanced sampling plan is ready and leak-safe.

## Phase 5: User Story 3 - Retrain and evaluate against the current checkpoint (Priority: P3)

**Goal**: Determine whether training on the rebalanced dataset actually improves the deployed
reconstruction model's real-world robustness, without ever promoting a regression and without ever
launching GPU/cloud compute without explicit authorization.

**Independent Test**: The prepared training config and evaluation harness can be reviewed and dry-run
(config validation only, no GPU execution) independent of whether execution is ever authorized; once
authorized and run, the checkpoint comparison must show a recorded `Outcome` before any promotion.

### Implementation for User Story 3

- [ ] T016 [US3] Confirm the currently active, unblocked reconstruction stage (Spec 108 `WdlPriorNet` or the currently active Spec 102 residual-chain stage) and its currently deployed checkpoint identity -- do not assume the stage identified at this plan's authoring time is still current
- [ ] T017 [US3] Implement `data-harvester/scripts/train_spec111_reconstruction.py` targeting the stage confirmed in T016 and the existing Spec 108 group-held-out split contract (`research.md`), consuming T013's rebalanced plan
- [ ] T018 [US3] Implement the checkpoint-comparison evaluation producing a `ReconstructionCheckpointComparison` record (`Outcome`, `PromotionDecision`) per `data-model.md`, with `PromotionDecision = false` enforced whenever `Outcome = regressed`
- [ ] T019 [US3] **STOP: confirm with the user before executing.** Only after explicit, separate authorization at the point of execution, run the training pass (T017) and the evaluation (T018), and record the outcome per `contracts/minimap-lighting-calibration-contract.md` §Training/evaluation execution contract

**Checkpoint**: all three user stories independently functional; go/no-go recorded for any executed training run.

## Phase 6: Polish and continuity

- [ ] T020 [P] Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` with this feature's outcome, per the constitution's Memory Bank Discipline
- [ ] T021 Update task states and exact proof commands in `specs/111-minimap-lighting-calibration/tasks.md` as each phase's checkpoint is reached

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
