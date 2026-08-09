# Tasks: V60 Controlled Terrain Reconstruction Experiment

**Input**: Design documents from `specs/134-v60-unified-dataset-model/`
**Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Phase Overview

| Phase | Goal | Status |
|---|---|---|
| 1 | Setup and contracts | Complete |
| 2 | Foundational control validation | In progress |
| 3 | Small deterministic synthetic control corpus | In progress |
| 4A | Synthetic object sieve and contamination decomposition | In progress |
| 4B | Limited control-data model experiment | Pending |
| 5 | Albedo normalization and textureless gate | Pending |
| 6 | Tiny real transfer and expansion decision | Pending |
| 7 | Later signals and client adapters | Deferred |

## Phase 1: Setup

- [x] T001 [P] Record the control, albedo-gate, and experiment contracts in `specs/134-v60-unified-dataset-model/contracts/`.
- [x] T002 [P] Record the entities and ownership boundaries in `specs/134-v60-unified-dataset-model/data-model.md`.
- [x] T003 [P] Record implementation decisions and rejected harvest-first alternatives in `specs/134-v60-unified-dataset-model/research.md`.
- [x] T004 [P] Write PowerShell-ready staged execution instructions in `specs/134-v60-unified-dataset-model/quickstart.md`.

## Phase 2: Foundational

- [x] T005 [P] Define exact control input/target shapes and hash validation in `data-harvester/src/harvester/v60/control_corpus.py`.
- [x] T006 [P] Define the procedural control command and family holdout policy in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [x] T007 [P] Add focused control manifest/hash/split tests in `data-harvester/tests/v60/test_control_corpus.py`.
- [ ] T008 Add JSON contract validation coverage for `specs/134-v60-unified-dataset-model/contracts/*.schema.json`.

**Checkpoint**: The contract surface is explicit and lightweight checks can run without client data,
GPU training, or a full harvest.

## Phase 3: User Story 1 — Small synthetic control corpus (P1) 🎯 MVP

**Goal**: Generate and validate a small exact control corpus with family-level holdouts.

**Independent Test**: Two user-run generations from the same command produce identical row hashes,
complete family splits, valid shapes/ranges, and no recursively harvested client output.

- [ ] T009 [P] [US1] Add explicit source-manifest parsing and `0.x`/`1.x` client-era rejection in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [x] T010 [P] [US1] Emit deterministic procedural families and variants in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [x] T011 [US1] Reuse `TerrainMinimapCompositor.ComposeShadowArray` for `terrain_shadow_256` and pair it with exact `height_257` in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [x] T012 [US1] Emit family split membership and per-array hashes in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [x] T013 [US1] Validate missing arrays, shape/range/finite errors, family leakage, and hashes in `data-harvester/src/harvester/v60/control_corpus.py`.
- [x] T014 [P] [US1] Add the `easy`/`medium`/`hard`/`pathological` complexity vocabulary and family-bucket validation in `data-harvester/src/harvester/v60/control_corpus.py`.
- [x] T015 [US1] Implement family, variant, and stitched cross-tile visual atlases in `data-harvester/scripts/v60_visualize_control_corpus.py`.
- [ ] T016 [P] [US1] Add repeat-run, bucket-coverage, cross-tile completeness, and visual-output fixture tests in `data-harvester/tests/v60/test_control_corpus.py` and `data-harvester/tests/v60/test_visualize_control_corpus.py`.
- [ ] T017 [US1] **USER RUNS** the bounded full-taxonomy control command from `specs/134-v60-unified-dataset-model/quickstart.md`.
- [ ] T018 [US1] Validate the generated manifest and write `validation.json` with `data-harvester/scripts/v60_validate_control_corpus.py`.

**Checkpoint**: A valid 108-row project-owned control corpus is available with fractal, cross-tile,
mountainous, sheer-dropoff, and zone-style-blend coverage; its generator also emits a 540-row
object-sieve derivative. No v50 store or full-client harvest is needed.

## Phase 4A: User Story 2 — Synthetic object sieve and contamination decomposition (P1)

**Goal**: Build a small object-overlay control lane that teaches removal of screen-space object
contamination without leaking ground-truth masks into inference.

**Independent Test**: The object-sieve report contains clean-output and mask metrics for no-object,
sparse, dense, overlapping, and boundary-crossing controls, with separate ablation variants.

- [x] T019a [P] [US2] Define the `objectified_terrain_shadow_256`, clean `terrain_shadow_256`, and
  `object_contamination_mask_256` contract in `specs/134-v60-unified-dataset-model/contracts/`.
- [x] T020a [P] [US2] Define deterministic procedural object families, density regimes, overlap, and
  tile-boundary placement metadata in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [x] T021a [US2] Emit synthetic object-overlay rows by reusing the canonical terrain shadow as the
  base and writing exact clean-target/mask arrays; do not add a second terrain-lighting equation.
- [x] T022a [US2] Add object-control validation and visual review for contamination coverage,
  placement regimes, and boundary-crossing cases in `data-harvester/src/harvester/v60/` and
  `data-harvester/scripts/`.
- [x] T023a [US2] Implement clean-only, auxiliary-mask-loss, and predicted-mask-guided sieve model
  variants; the guided variant must consume predicted masks during training and inference.
- [ ] T024a [P] [US2] Add separate clean-output, mask, density, placement, and held-out-family
  metrics plus a ground-truth-mask-input leakage assertion.
- [ ] T025a [US2] **USER RUNS** the bounded object-sieve control experiment after its corpus passes
  validation; GPU work remains user-owned.
- [x] T026a [P] [US2] Inspect the existing v50.1 object-mask curriculum and record populated real
  mask arrays, authored-row count, source-group split state, and empty geometry-visible evidence.
- [x] T027a [P] [US2] Define the real v50 object-mask dataset and experiment contracts in
  `specs/134-v60-unified-dataset-model/`.
- [x] T028a [US2] Define the real-mask model/loss variants and minimum-requested-target checkpoint
  selection rule in `data-harvester/src/harvester/v60/`.
- [x] T029a [US2] Implement the lazy v50 Zarr loader, authored/map-holdout split audit, and
  real-mask trainer/evaluator in `data-harvester/scripts/v60_train_real_object_masks.py`.
- [x] T030a [P] [US2] Add real-mask target projection, model/loss, provenance, and leakage tests in
  `data-harvester/tests/v60/`.
- [x] T031a [P] [US2] Add same-tile authored/synthetic pair selection, domain-distance report, and
  visual atlas in `data-harvester/src/harvester/v60/real_synthetic_pairs.py` and
  `data-harvester/scripts/v60_validate_real_synthetic_pairs.py`.
- [x] T032a [US2] Reclassify the legacy v50 synthetic minimap as a flat fake-maptexture diagnostic;
  remove it from the real-mask trainer's terrain input contract.
- [ ] T034a [US2] **USER RUNS** the real v50 object-mask experiment; GPU work remains user-owned.
- [ ] T033a [US2] **USER RUNS** a bounded post-fix `harvest-map-mpq` tile set that emits
  `terrain_shadow_256` for the same pair identities; GPU work remains user-owned.

**Checkpoint**: The object sieve has independent mask and clean-terrain evidence. Do not pass its
ground-truth mask into the height model or authorize real-client expansion from this result alone.

## Phase 4B: User Story 3 — Limited control-data model experiment (P1)

**Goal**: Test the first input/height relationship with a small learning curve and family holdout.

**Independent Test**: The report contains fixed holdout-family metrics, a tile-mean baseline,
per-training-size results, retexturing controls, and ambiguity labels.

- [ ] T019 [P] [US3] Implement a control-v1 loader that reads `terrain_shadow_256` and `height_257` without changing historical contracts in `data-harvester/src/harvester/v60/control_experiment.py`.
- [ ] T020 [P] [US3] Implement fixed-family split selection and limited training-size schedules in `data-harvester/src/harvester/v60/control_experiment.py`.
- [ ] T021 [US3] Implement tile-mean baseline and per-family/per-variant metrics in `data-harvester/src/harvester/v60/control_experiment.py`.
- [ ] T022 [US3] Add the bounded experiment CLI and `experiment-report-v1` writer in `data-harvester/scripts/v60_run_experiment.py`.
- [ ] T023 [P] [US3] Add loader, split, baseline, and unchanged-target retexturing tests in `data-harvester/tests/v60/test_control_experiment.py`.
- [ ] T024 [US3] **USER RUNS** limited clean-height control training/evaluation after T025a and T018
  pass; GPU work remains user-owned.
- [ ] T025 [US3] Record the held-out result and ambiguity cases in the experiment report before real-data work in `specs/134-v60-unified-dataset-model/`.

**Checkpoint**: Control evidence says whether the first relationship is learnable. Do not expand the
corpus merely because the score is disappointing.

## Phase 5: User Story 4 — Albedo normalization and textureless gate (P1)

**Goal**: Convert a tiny explicit real sample into canonical textureless inputs and fail closed on
textured or failed outputs.

**Independent Test**: Positive controls, textured negatives, missing outputs, and non-finite outputs
produce persisted accepted/rejected/quarantined decisions with thresholds and reasons.

- [ ] T026 [P] [US4] Define the versioned albedo-operation input/output contract in `data-harvester/src/harvester/v60/albedo_normalization.py`.
- [ ] T027 [US4] Implement candidate albedo estimation/removal and residual metrics in `data-harvester/src/harvester/v60/albedo_normalization.py`; do not substitute synthetic ground truth.
- [ ] T028 [US4] Implement threshold calibration from positive and deliberately textured/failed controls in `data-harvester/src/harvester/v60/albedo_gate.py`.
- [ ] T029 [US4] Implement fail-closed accepted/rejected/quarantined decisions and report writing in `data-harvester/src/harvester/v60/albedo_gate.py`.
- [ ] T030 [US4] Add the PowerShell-ready tiny 0.x/1.x normalization CLI in `data-harvester/scripts/v60_normalize_albedo.py`.
- [ ] T031 [P] [US4] Add missing, non-finite, textured, threshold, and artifact tests in `data-harvester/tests/v60/test_albedo_normalization.py`.
- [ ] T032 [US4] **USER RUNS** the tiny explicit real-sample normalization after the control gate and source manifest are ready.

**Checkpoint**: Only accepted rows enter the model input directory; every other row remains visible
in the gate report.

## Phase 6: User Story 5 — Tiny real transfer and expansion decision (P2)

**Goal**: Compare accepted real inputs with controls and make an explicit hold/diagnose/expand call.

**Independent Test**: The transfer report references both runs, compares distributions and metrics,
and contains an explicit decision.

- [ ] T033 [P] [US5] Implement control-vs-real input distribution and failure-signature comparison in `data-harvester/src/harvester/v60/transfer_gate.py`.
- [ ] T034 [US5] Reuse the control evaluator for accepted normalized rows in `data-harvester/src/harvester/v60/transfer_gate.py`.
- [ ] T035 [US5] Write the `TransferGate` decision and expansion block in `data-harvester/src/harvester/v60/transfer_gate.py`.
- [ ] T036 [P] [US5] Add hold/diagnose/expand decision tests in `data-harvester/tests/v60/test_transfer_gate.py`.
- [ ] T037 [US5] **USER RUNS** the tiny transfer comparison after T032; broader processing is not authorized by a synthetic score alone.
- [ ] T038 [US5] Record the transfer decision and next bounded batch in `specs/134-v60-unified-dataset-model/quickstart.md` or a run-local report.

**Checkpoint**: Broader processing is allowed only when the report says `expand`; otherwise diagnose
normalization or domain shift first.

## Phase 7: User Story 6 — Deferred extensions (P3)

- [ ] T039 [P] [US6] Add one additional exact signal behind a versioned contract in `data-harvester/src/harvester/v60/`.
- [ ] T040 [P] [US6] Add later client-era adapters behind the same source manifest in `tools/harvest/WowViewer.Tool.Harvest/Program.cs`.
- [ ] T041 [US6] Expand controls only in response to a recorded failure mode in `specs/134-v60-unified-dataset-model/`.

## Phase 8: Polish and cross-cutting validation

- [ ] T042 [P] Add JSON schema validation to the focused v60 test suite in `data-harvester/tests/v60/`.
- [ ] T043 [P] Update `specs/134-v60-unified-dataset-model/quickstart.md` whenever an executable CLI lands.
- [ ] T044 Run `git diff --check` and focused Python/C# checks after each bounded implementation slice.
- [ ] T045 Keep `specs/134-v60-unified-dataset-model/spec.md`, `plan.md`, and `tasks.md` aligned with the current gate status.

## Dependencies and execution order

```text
Phase 1 setup
    -> Phase 2 foundation
        -> US1 control corpus
            -> US2 object sieve
                -> US3 height control experiment
                    -> US4 albedo normalization/gate
                        -> US5 tiny transfer gate
                            -> US6 extensions
```

- US1 blocks US2 because the object-sieve evaluator needs a validated terrain/control manifest.
- US2 blocks US3 because the height evaluator should consume the proven clean signal contract.
- US3 blocks US4 because the real-input gate must target a measured canonical input contract.
- US4 blocks US5 because only accepted normalized rows may transfer.
- US5 blocks broader processing and all later-era expansion.

## Parallel execution opportunities

- T001–T004 are independent documentation/contract tasks.
- T009, T014, and T017–T021 can be developed in parallel only after their respective preceding
  contract surfaces are stable; the user-run tasks remain sequential gates.
- T024, T029, and T031/T034 are parallel within their story when their shared report contracts are
  fixed.

## MVP strategy

1. Finish the existing control command and run/validate the 32-row corpus.
2. Implement and run the limited control experiment; stop and inspect the evidence.
3. Implement albedo normalization and the textureless gate against a tiny 0.x/1.x sample.
4. Run one transfer decision. Only then decide whether to expand real data or diagnose the route.

No full v50/v60 harvest is part of this MVP.
