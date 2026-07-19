# Tasks: Universal Image-to-Terrain Reconstruction

**Input**: Design documents from `specs/114-direct-terrain-reconstruction/`

**Execution ownership**: The assistant implements and runs lightweight tests/dry runs. The user
alone launches model downloads, teacher labeling, corpus builds, CUDA training, or other heavy work.

**Phase rule**: Finish and validate universal relief before beginning WoW-specific cleanup,
semantics, texture families, or alpha. The failed WoW-only CNN is negative evidence, not the next
training recipe.

## Phase 1: Setup and corrected evidence

- [ ] T001 Record Spec 113 fixed-noon-white visual evidence and fresh synthetic store hashes in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T002 Audit unaffected v50 numeric height/normal/liquid/material/alpha signals in `specs/114-direct-terrain-reconstruction/research.md`
- [x] T003 Freeze the rejected `direct_cnn_v112-authored-v1` identity and metrics in `specs/114-direct-terrain-reconstruction/research.md`
- [x] T004 Add checkpoint-backfill validation artifacts in `data-harvester/src/harvester/v50/height_relative_evaluate.py` and `data-harvester/scripts/v50_evaluate_height_relative.py`
- [x] T005 Correct the deployment boundary from WoW minimaps to arbitrary rasters across `specs/114-direct-terrain-reconstruction/`

**Checkpoint**: The old run is reviewable and rejected; the spec no longer authorizes a narrow retry.

## Phase 2: Foundational universal contracts and curriculum

- [x] T006 [P] Write raster-mode/aspect/blank/finite-mesh/UV contract tests in `data-harvester/tests/v50/test_universal_relief_contract.py`
- [x] T007 Implement arbitrary-raster preprocessing, relief stitching, and deterministic mesh export in `data-harvester/src/harvester/v50/universal_relief_contract.py`
- [ ] T008 [P] Add universal curriculum and model-stage schema fixtures under `data-harvester/tests/fixtures/v50/spec114/`
- [ ] T009 Implement fail-closed schema and source/group/family validation in `data-harvester/src/harvester/v50/model_stage_contract.py`
- [ ] T010 [P] Write teacher identity/orientation/normalization/license tests in `data-harvester/tests/v50/test_relief_teacher_labels.py`
- [ ] T011 Implement pinned non-DepthAnything teacher labeling in `data-harvester/src/harvester/v50/relief_teacher_labels.py`
- [ ] T012 [P] Write whole-family split, derived-view leakage, and target-authority tests in `data-harvester/tests/v50/test_universal_relief_curriculum.py`
- [ ] T013 Implement the immutable universal curriculum/index builder and dry-run CLI in `data-harvester/src/harvester/v50/universal_relief_curriculum.py` and `data-harvester/scripts/v50_build_universal_relief_curriculum.py`

**Checkpoint**: Any training row is exact numeric or pinned teacher pseudo-relief; at least five
visual families exist; source-group and held-out-family leak counts are zero.

## Phase 3: User Story 1 — Any raster to terrain (P1 MVP)

**Independent test**: The compatibility suite accepts every valid raster and emits a finite mesh;
whole-family paired holdouts beat constant and luminance baselines; the user accepts the arbitrary-
image review sheet.

- [ ] T014 [P] [US1] Write one-output, arbitrary-size, finite-forward, and frozen-backbone tests in `data-harvester/tests/v50/test_universal_relief_model.py`
- [ ] T015 [US1] Implement the pinned DINOv2-small relief student and one continuous decoder in `data-harvester/src/harvester/v50/universal_relief_model.py`
- [ ] T016 [P] [US1] Write loss masking, exact/pseudo authority, EMA, scheduler, AMP, clipping, and artifact tests in `data-harvester/tests/v50/test_universal_relief_train.py`
- [ ] T017 [US1] Implement the guided optimization/evaluation stack in `data-harvester/src/harvester/v50/universal_relief_train.py`
- [ ] T018 [P] [US1] Add the any-image relief/OBJ/height-preview inference CLI in `data-harvester/scripts/v50_image_to_terrain.py`
- [ ] T019 [US1] Add dry-run, immutable-source identity, and explicit user-confirmation training CLI in `data-harvester/scripts/v50_train_universal_relief.py`
- [ ] T020 [US1] Run CPU fixtures, lint, compile, and real no-training dry runs; record proof and exact commands in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T021 [US1] USER RUN: build broad teacher relief labels using the exact command in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T022 [US1] USER RUN: train the universal relief student using the exact command in `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T023 [US1] Evaluate SC-001 through SC-004 and record the user visual verdict in `specs/114-direct-terrain-reconstruction/research.md`

**Checkpoint**: Stop. No downstream model begins unless arbitrary-image geometry promotes.

## Phase 4: User Story 2 — Optional WoW object cleanup (P2)

- [ ] T024 [P] [US2] Audit and freeze the renderer-aligned visibility-label seam in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T025 [P] [US2] Write populated/empty/unavailable and alignment tests in `data-harvester/tests/v50/test_object_visibility_labels.py`
- [ ] T026 [US2] Implement trusted object-visibility packing in `data-harvester/src/harvester/v50/object_visibility_labels.py`
- [ ] T027 [P] [US2] Write mask shape/loss/baseline tests in `data-harvester/tests/v50/test_object_mask_model.py`
- [ ] T028 [US2] Implement the independent object-mask model/trainer in `data-harvester/src/harvester/v50/object_mask_model.py` and `data-harvester/src/harvester/v50/object_mask_train.py`
- [ ] T029 [US2] Add user-run label-build and mask-training CLIs in `data-harvester/scripts/v50_build_object_visibility.py` and `data-harvester/scripts/v50_train_object_mask.py`
- [ ] T030 [US2] USER RUN: build labels, train the mask, and persist generated masks using `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T031 [US2] Evaluate the generated-mask relief ablation and record SC-005 in `specs/114-direct-terrain-reconstruction/research.md`

## Phase 5: User Story 3 — Reusable terrain features (P3)

- [ ] T032 [P] [US3] Write deterministic-rule, unknown-state, and family-leak tests in `data-harvester/tests/v50/test_terrain_feature_library.py`
- [ ] T033 [US3] Implement the versioned feature vocabulary in `data-harvester/src/harvester/v50/terrain_feature_library.py`
- [ ] T034 [P] [US3] Write semantic output/confidence/macro-metric tests in `data-harvester/tests/v50/test_terrain_feature_model.py`
- [ ] T035 [US3] Implement the independent feature classifier/trainer in `data-harvester/src/harvester/v50/terrain_feature_model.py`
- [ ] T036 [US3] Add the user-run library and trainer CLIs in `data-harvester/scripts/v50_build_terrain_feature_library.py` and `data-harvester/scripts/v50_train_terrain_features.py`
- [ ] T037 [US3] USER RUN: build the family-safe library and train the classifier using `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T038 [US3] Record per-class coverage, unknowns, overlays, and SC-006 in `specs/114-direct-terrain-reconstruction/research.md`

## Phase 6: User Story 4 — Editable texture families and alpha (P4)

- [ ] T039 [P] [US4] Write texture alias/unknown/order/family-leak tests in `data-harvester/tests/v50/test_texture_family_library.py`
- [ ] T040 [US4] Implement canonical texture-family mapping in `data-harvester/src/harvester/v50/texture_family_library.py`
- [ ] T041 [P] [US4] Write selector confidence/majority-baseline tests in `data-harvester/tests/v50/test_texture_family_model.py`
- [ ] T042 [US4] Implement the independent family selector/trainer in `data-harvester/src/harvester/v50/texture_family_model.py`
- [ ] T043 [US4] USER RUN: build/train/evaluate texture families using `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T044 [P] [US4] Write alpha shape/bounds/layer/compositor tests in `data-harvester/tests/v50/test_alpha_stack_model.py`
- [ ] T045 [US4] Implement the independent ordered alpha-stack model in `data-harvester/src/harvester/v50/alpha_stack_model.py`
- [ ] T046 [US4] Implement alpha numeric/recomposition evaluation in `data-harvester/src/harvester/v50/alpha_stack_train.py`
- [ ] T047 [US4] USER RUN: train/evaluate alpha reconstruction using `specs/114-direct-terrain-reconstruction/quickstart.md`
- [ ] T048 [US4] Record texture/alpha SC-007 and SC-008 verdicts in `specs/114-direct-terrain-reconstruction/research.md`

## Phase 7: Documentation and end-to-end audit

- [ ] T049 [P] Audit source-raster-to-mesh/material lineage and zero teacher/truth deployment inputs in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T050 [P] Prove independent checkpoint replacement without unrelated retraining in `specs/114-direct-terrain-reconstruction/research.md`
- [ ] T051 Synchronize commands/status in `data-harvester/README.md` and `docs/dataset-preparation-userguide.md`
- [ ] T052 Compress current truth into `memory-bank/activeContext.md` and `memory-bank/progress.md`
- [ ] T053 Run focused tests, schema validation, `git diff --check`, and final user visual gates; record proof in `specs/114-direct-terrain-reconstruction/quickstart.md`

## Dependencies and MVP

```text
Phase 1 evidence
  -> Phase 2 universal contracts/curriculum
  -> Phase 3 universal relief MVP
  -> Phase 4 optional WoW cleanup
  -> Phase 5 terrain semantics
  -> Phase 6 editable texture/alpha
  -> Phase 7 audit
```

The MVP is Phases 1–3 only. Source-image UV projection gives an immediate textured mesh; semantic
material families and alpha are later independently promotable stages. No later phase may consume
teacher or ground-truth signals at deployment.
