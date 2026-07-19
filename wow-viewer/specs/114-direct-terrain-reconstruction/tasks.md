# Tasks: Direct Minimap-to-Terrain Reconstruction

**Input**: Design documents from `specs/114-direct-terrain-reconstruction/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`,
`contracts/model-stage-and-curriculum.schema.json`, `quickstart.md`

**Execution ownership**: The assistant implements/tests lightweight code and prepares exact CLI
commands. The user alone launches rendering, corpus builds, CUDA training, or other heavy runs.

**Phase rule**: Finish and validate each phase before starting the next. In particular, do not begin
object-mask work while direct geometry is unproven.

## Format: `[ID] [P?] [Story] Description`

- **[P]** means different files and no incomplete dependency in the current phase.
- Every story task names an exact planned file.
- Tests are required because every model stage has a fail-closed contract and promotion gate.

## Phase 1: Setup and source gate

**Purpose**: Bind the feature to corrected source evidence and keep the old baseline reproducible.

- [ ] T001 Record the completed Spec 113 T010b fixed-noon-white visual evidence and fresh synthetic store/manifest hashes in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T002 Audit Kalimdor/Azeroth v50.1 signal coverage and confirm numeric height/normal/liquid/material/alpha plus authored RGB were not invalidated by the synthetic-lighting fix in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T003 Freeze the strongest completed Spec 112 run identity and metrics as `direct_cnn_v112` comparison evidence in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T004 [P] Add schema fixture documents for all three contract variants under `data-harvester/tests/fixtures/v50/spec114/`
- [ ] T005 [P] Add JSON-schema contract tests in `data-harvester/tests/v50/test_model_stage_contract.py`

**Checkpoint**: Source provenance is corrected and contract fixtures pass; no corpus or trainer has
been launched.

---

## Phase 2: Foundational curriculum and run contracts

**Purpose**: Build shared identity/lineage infrastructure that blocks every model story.

- [ ] T006 Implement immutable curriculum/model-stage validation in `data-harvester/src/harvester/v50/model_stage_contract.py`
- [ ] T007 [P] Write failing grouped-split, missing-signal, stale-lighting, and generated-input provenance tests in `data-harvester/tests/v50/test_reconstruction_curriculum.py`
- [ ] T008 Implement dual-view row selection and summary generation in `data-harvester/src/harvester/v50/reconstruction_curriculum.py`
- [ ] T009 Add the thin non-training builder/dry-run CLI in `data-harvester/scripts/v50_build_reconstruction_curriculum.py`
- [ ] T010 Run CPU fixtures and hand the user the exact real corpus-build command, output paths, coverage expectations, and time/disk estimate in `specs/114-direct-terrain-reconstruction/quickstart.md`

**Checkpoint**: The curriculum builder refuses group leakage, zero-filled missing signals, and any
synthetic view without `NoonWhiteGlobal` provenance.

---

## Phase 3: User Story 1 — Direct relative geometry (Priority: P1) MVP

**Goal**: Compare the Spec 112 lean CNN with one compact MiT-B0 continuous regressor on the exact
same direct RGB-to-`relative_height_257` curriculum, without WDL.

**Independent Test**: Frozen held-out metrics beat the flat/tile-mean and strongest Spec 112
baselines by SC-001, border error passes SC-002, and the user accepts the held-out visual sheet.

- [x] T011 [P] [US1] Prove shape, finite-output, offset-invariance, and the one-RGB-input contract in `data-harvester/tests/v50/test_height_relative_model.py` and `data-harvester/tests/v50/test_height_relative_train.py`
- [x] T012 [P] [US1] Add direct-geometry source-filter, dry-run-plan, run-summary, and epoch-1 structural-failure tests in `data-harvester/tests/v50/test_height_relative_train.py`
- [x] T013 [US1] Pin the proven Spec 112 lean CNN as the 1,561,537-parameter `direct_cnn_v112` architecture in `data-harvester/src/harvester/v50/height_relative_model.py`
- [ ] T014 [US1] Add the compact `mit_b0_regression` architecture with one continuous relative-height output in `data-harvester/src/harvester/v50/direct_geometry_model.py`
- [ ] T015 [US1] Implement same-split training/evaluation, flat/tile-mean baselines, border metrics, and held-out visual sheets in `data-harvester/src/harvester/v50/direct_geometry_train.py`
- [x] T016 [US1] Add explicit `--source`, stale-synthetic refusal, no-training dry run, and `--confirm-run` to `data-harvester/scripts/v50_train_height_relative.py` through its library owner
- [x] T017 [US1] USER RUN: train tonight's authored-only `direct_cnn_v112` baseline with the exact proven command in `specs/114-direct-terrain-reconstruction/quickstart.md`; the 100-epoch run completed with best epoch 92, validation MAE 0.149267 versus tile-mean 0.138747, so it failed promotion honestly.
- [x] T017a [US1] Add deterministic per-best prediction sheets plus final all-validation per-row MAE,
  gradient MAE, border MAE, baseline comparison, error-quantile sheet, and worst-case sheet. Add a
  separate evaluator that can backfill those artifacts from the completed immutable checkpoint.
- [ ] T017b [US1] Port the previously proven bounded trainer stack in one separately validated
  change: AMP, EMA deploy weights, warmup/cosine decay, gradient clipping, multiscale height loss,
  training-only normal guidance through valid non-liquid terrain, detached hard-error weighting,
  peak-VRAM/history evidence, and identity-only geometric augmentation for baked-light RGB.
- [ ] T018 [US1] Validate the authored baseline summary first, then compare the later corrected dual-view/CNN/MiT runs against SC-001/SC-002 and record the user's visual verdict in `specs/114-direct-terrain-reconstruction/research.md`

**Checkpoint**: Stop. Geometry MVP must be validated before any trusted-object work begins.

---

## Phase 4: User Story 2 — Trusted object visibility (Priority: P2)

**Goal**: Produce trusted renderer-aligned object labels, train one semantic mask model, then measure
whether generated mask guidance improves the already-proven geometry model.

**Independent Test**: The mask beats empty/all-object baselines and generated-mask guidance meets
SC-003 without teacher-forced object truth.

- [ ] T019 [US2] Audit existing verified placement/runtime geometry and freeze the bounded top-down visibility export seam in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T020 [P] [US2] Write failing populated/empty/unavailable and alignment-fixture tests in `data-harvester/tests/v50/test_object_visibility_labels.py`
- [ ] T021 [US2] Implement trusted object-visibility label packing and provenance validation in `data-harvester/src/harvester/v50/object_visibility_labels.py`
- [ ] T022 [US2] Expose the bounded visibility export through `tools/harvest/WowViewer.Tool.Harvest/Program.cs` and extend renderer-contract proof in `tests/WowViewer.Core.Tests/TerrainVisibleObjectMaskRasterizerTests.cs`
- [ ] T023 [P] [US2] Write failing mask shape/loss/baseline tests in `data-harvester/tests/v50/test_object_mask_model.py`
- [ ] T024 [US2] Implement one compact semantic object-mask architecture in `data-harvester/src/harvester/v50/object_mask_model.py` and its trainer in `data-harvester/src/harvester/v50/object_mask_train.py`
- [ ] T025 [US2] Add user-run label-build and mask-training CLIs in `data-harvester/scripts/v50_build_object_visibility.py` and `data-harvester/scripts/v50_train_object_mask.py`
- [ ] T026 [US2] USER RUN: render labels, train the mask model, and persist generated masks for the frozen geometry split using commands documented in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T027 [US2] Retrain/evaluate geometry with generated masks only and record the SC-003 promotion verdict in `specs/114-direct-terrain-reconstruction/research.md`

**Checkpoint**: Object labels are renderer-trusted, missing rows remain unavailable, and mask
guidance is either proven or explicitly rejected without blocking the RGB-only geometry model.

---

## Phase 5: User Story 3 — Reusable terrain features (Priority: P3)

**Goal**: Freeze a deterministic feature vocabulary and train one independent image-to-feature map.

**Independent Test**: Family-safe validation beats the majority baseline by SC-004 and reports all
unknown/unsupported coverage.

- [ ] T028 [P] [US3] Write failing deterministic-rule, unknown-state, and family-leak tests in `data-harvester/tests/v50/test_terrain_feature_library.py`
- [ ] T029 [US3] Define the first height/slope/curvature/liquid/material/alpha-derived vocabulary and derivation evidence in `data-harvester/src/harvester/v50/terrain_feature_library.py`
- [ ] T030 [US3] Add the library build/audit CLI in `data-harvester/scripts/v50_build_terrain_feature_library.py`
- [ ] T031 [P] [US3] Write failing semantic output, confidence, and macro-metric tests in `data-harvester/tests/v50/test_terrain_feature_model.py`
- [ ] T032 [US3] Implement one compact semantic feature classifier in `data-harvester/src/harvester/v50/terrain_feature_model.py`
- [ ] T033 [US3] Add the fail-closed user-run trainer CLI in `data-harvester/scripts/v50_train_terrain_features.py`
- [ ] T034 [US3] USER RUN: build the family-safe library/corpus and train the feature classifier with commands documented in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T035 [US3] Record per-class coverage, macro-F1, unknowns, overlays, and the SC-004 verdict in `specs/114-direct-terrain-reconstruction/research.md`

**Checkpoint**: One versioned feature library and classifier are proven independently; geometry
weights were not shared or retrained.

---

## Phase 6: User Story 4A — Texture-family selection (Priority: P4)

**Goal**: Predict ordered canonical texture-family identities without predicting alpha.

**Independent Test**: The family-safe split beats per-map majority, reports unknowns, and passes
SC-005 with visually credible recomposition choices.

- [ ] T036 [P] [US4] Write failing alias, unknown, ordering, and family-leak tests in `data-harvester/tests/v50/test_texture_family_library.py`
- [ ] T037 [US4] Implement build-specific texture-to-canonical-family mapping in `data-harvester/src/harvester/v50/texture_family_library.py`
- [ ] T038 [US4] Add the library build/audit CLI in `data-harvester/scripts/v50_build_texture_family_library.py`
- [ ] T039 [P] [US4] Write failing ordered-selection, confidence, and majority-baseline tests in `data-harvester/tests/v50/test_texture_family_model.py`
- [ ] T040 [US4] Implement one independent ordered family selector using generated terrain-feature context in `data-harvester/src/harvester/v50/texture_family_model.py`
- [ ] T041 [US4] Add the fail-closed user-run trainer CLI in `data-harvester/scripts/v50_train_texture_families.py`
- [ ] T042 [US4] USER RUN: build the canonical family corpus and train the selector using commands documented in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T043 [US4] Record majority comparison, family-safe audit, unknown coverage, recomposition review, and SC-005 verdict in `specs/114-direct-terrain-reconstruction/research.md`

**Checkpoint**: Ordered family IDs are independently generated and frozen before alpha work.

---

## Phase 7: User Story 4B — Alpha-stack reconstruction (Priority: P4)

**Goal**: Predict one bounded ordered alpha stack conditioned on generated family selections.

**Independent Test**: Numeric blend and recomposition metrics beat base-only/uniform baselines by
SC-006 on Alpha and LK fixtures without renderer/parser changes.

- [ ] T044 [P] [US4] Write failing shape, bounds, layer-presence, and compositor-order tests in `data-harvester/tests/v50/test_alpha_stack_model.py`
- [ ] T045 [US4] Freeze the four-layer target, base-only/missing-MCAL states, and generated-family input policy in `data-harvester/src/harvester/v50/alpha_stack_model.py`
- [ ] T046 [US4] Implement one lean U-Net/FPN alpha-stack regressor in `data-harvester/src/harvester/v50/alpha_stack_model.py`
- [ ] T047 [US4] Implement numeric alpha metrics, base-only/uniform baselines, and existing-compositor recomposition evaluation in `data-harvester/src/harvester/v50/alpha_stack_train.py`
- [ ] T048 [US4] Add the fail-closed user-run trainer CLI in `data-harvester/scripts/v50_train_alpha_stack.py`
- [ ] T049 [US4] Prove CPU forward/backward plus Alpha/LK compositor fixtures without editing MCAL decode or `src/core/WowViewer.Core.IO/Maps/AlphaWdtWriter.cs`
- [ ] T050 [US4] USER RUN: train/evaluate alpha reconstruction with generated family selections using commands documented in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T051 [US4] Record blend-field metrics, recomposition metrics, visual review, and SC-006 verdict in `specs/114-direct-terrain-reconstruction/research.md`

**Checkpoint**: Texture family and alpha are separately promotable checkpoints; Spec 113 still
owns visual-detail SR.

---

## Phase 8: Documentation and end-to-end audit

**Purpose**: Prove the modular inference graph and preserve its current state.

- [ ] T052 [P] Audit authored-minimap-to-output lineage and prove zero ground-truth inference inputs for SC-007 in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T053 [P] Prove independent checkpoint replacement without unrelated retraining for SC-008 in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T054 Synchronize exact commands/status in `data-harvester/README.md`, `docs/dataset-preparation-userguide.md`, `memory-bank/activeContext.md`, and `memory-bank/progress.md`
- [ ] T055 Run focused Python/C# tests, schema validation, `git diff --check`, and the final user visual gate; record exact proof in `specs/114-direct-terrain-reconstruction/quickstart.md`

## Dependencies and execution order

```text
Phase 1 source gate
  -> Phase 2 contracts/curriculum
  -> Phase 3 direct geometry MVP
  -> Phase 4 trusted objects and mask-guided geometry
  -> Phase 5 terrain features
  -> Phase 6 texture families
  -> Phase 7 alpha stack
  -> Phase 8 end-to-end audit
```

- No phase may consume a later phase's ground truth at inference.
- Generated upstream signals must be materialized for downstream training/evaluation.
- A failed optional candidate does not invalidate a proven earlier checkpoint.
- Spec 112 supplies the direct-CNN baseline; Spec 113 exclusively owns RGB super-resolution.

## MVP scope

The MVP is Phases 1-3 only: corrected dual-view corpus plus a proven direct relative-height model.
Stop there, review the geometry result, and only then authorize the object-mask phase.
