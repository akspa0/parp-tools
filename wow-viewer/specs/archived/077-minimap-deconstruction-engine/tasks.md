# Tasks: Minimap Deconstruction Engine

**Input**: Design documents from `/specs/077-minimap-deconstruction-engine/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`

**Tests**: Include bounded xUnit, pytest, and smoke-validation tasks where the spec requires them.

**Organization**: Tasks are grouped by user story and phase so each phase can be implemented and validated independently.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Direction Lock and Documentation (Shared Foundation)

**Purpose**: Freeze the new direction before any code changes.

- [x] T001 [US1] Add `wow-viewer/specs/077-minimap-deconstruction-engine/spec.md` with the successor feature contract and explicit non-goals.
- [x] T002 [P] [US1] Add `wow-viewer/specs/077-minimap-deconstruction-engine/research.md` auditing the existing capture, placement, and mask-owner surfaces.
- [x] T003 [P] [US1] Add `wow-viewer/specs/077-minimap-deconstruction-engine/data-model.md` defining object-library, teacher-prior, and height-only sample schemas.
- [x] T004 [US1] Add `wow-viewer/specs/077-minimap-deconstruction-engine/plan.md` with phase gates that forbid normals or restoration before height proof.
- [x] T005 [US1] Add `wow-viewer/docs/architecture/minimap-deconstruction-engine-2026-06-28.md` summarizing the staged deconstruction pipeline and its proof owners.

**Checkpoint**: The new planning surface exists and is explicit about tiny single-purpose stages.

---

## Phase 2: User Story 1 - Per-Object Capture Library (Priority: P1)
**Goal**: Build a reusable object-library contract and one-object-at-a-time capture lane.

**Independent Test**: A bounded capture run writes a Zarr-backed object library with image, mask, and metadata rows.

### Tests for User Story 1

- [x] T006 [P] [US1] Add shared-library tests for `ObjectLibraryEntry` and `ObjectCaptureVariant` serialization under `wow-viewer/tests/WowViewer.Core.Tests/`.
- [ ] T007 [P] [US1] Add bounded capture-tool tests for object-library artifact writing under `wow-viewer/tests/WowViewer.Core.Tests/`.

### Implementation for User Story 1

- [x] T008 [US1] Add shared data contracts for object-library entries and capture variants under `wow-viewer/src/core/WowViewer.Core/Maps/` or a nearby canonical owner.
- [x] T009 [US1] Add object-library Zarr/Parquet writer support under `wow-viewer/src/core/WowViewer.Core.IO/Maps/`.
- [ ] T010 [US1] Extend the existing capture lane in `wow-viewer/tools/validation-capture/` to support one-object-at-a-time capture requests keyed by asset path.
- [x] T011 [US1] Add capture-job enumeration from harvested placement name tables under `wow-viewer/data-harvester/scripts/` or a thin CLI wrapper, without inventing a parallel metadata source.
- [x] T012 [US1] Persist captured image, mask, capture status, and metadata rows into `wow-viewer/output/datasets/object-library/`.
- [x] T013 [US1] Emit bounded review artifacts for captured objects under `wow-viewer/output/analysis/object-library/`.

**Checkpoint**: The repo can build a bounded per-object library from already-harvested asset references.

---

## Phase 3: User Story 2 - Teacher Deconstruction Priors (Priority: P1)

**Goal**: Generate ADT-backed processed minimap priors using filtered precise teacher signals.

**Independent Test**: A bounded prior-generation run writes raw minimap, teacher mask, confidence, and processed prior arrays plus review artifacts.

### Tests for User Story 2

- [x] T014 [P] [US2] Add pytest coverage for no-object pass-through and object-heavy suppression behavior in `wow-viewer/data-harvester/tests/`.
- [x] T015 [P] [US2] Add tests for metadata/index parity between source V18 tiles and generated teacher-prior rows.
- [x] T015a [P] [US2] Add tests for precise-first mask priority, compact-row review, and mask/minimap visibility audit bucketing.

### Implementation for User Story 2

- [x] T016 [US2] Add a Python library module for teacher-prior generation under `wow-viewer/data-harvester/src/harvester/`.
- [x] T017 [US2] Add a CLI script under `wow-viewer/data-harvester/scripts/` that reads V18 Zarr stores and writes the teacher-prior dataset.
- [x] T018 [US2] Make the prior generator explicitly prefer `object_precise_mask`, with `object_filtered_mask` and `object_mask` as documented fallbacks/ablations.
- [x] T019 [US2] Add processed-prior review-artifact rendering under `wow-viewer/data-harvester/scripts/`.
- [x] T020 [US2] Document the exact phase-1 prior channels in the generated dataset metadata.
- [x] T020a [US2] Add targeted teacher-prior review by original `tile_id` and compact `row_index`, including source V18 masks.
- [x] T020b [US2] Add mask/minimap visibility audit outputting `visibility_audit.parquet`, `summary.json`, and `kept_tiles.parquet` for second-stage curation.
- [ ] T021 [US2] Run and record a bounded proof on one object-rich anchor map using staged-client-backed V18 data.

**Checkpoint**: A reusable teacher-prior dataset exists and is auditable tile by tile.

---

## Phase 4: User Story 3 - Height-Only Terrain Reboot (Priority: P1) 🎯 MVP

**Goal**: Train the smallest viable processed-prior to `height_257` model and prove the contract end to end.

**Independent Test**: A smoke training run completes using processed priors and emits previews and checkpoints.

### Tests for User Story 3

- [x] T022 [P] [US3] Add dataset-loader tests for the new processed-prior height dataset in `wow-viewer/data-harvester/tests/`.
- [x] T023 [P] [US3] Add a trainer smoke test or deterministic dry-run assertion for the height-only lane.

### Implementation for User Story 3

- [x] T024 [US3] Audit the smallest existing height model/trainer owner to reuse under `wow-viewer/data-harvester/src/harvester/` and `scripts/`.
- [x] T025 [US3] Add a processed-prior height dataset loader under `wow-viewer/data-harvester/src/harvester/`.
- [x] T026 [US3] Add or adapt a height-only training script under `wow-viewer/data-harvester/scripts/` that predicts only `height_257`.
- [x] T027 [US3] Preserve existing filtered terrain-valid weighting and authoritative raw height targets in the new lane.
- [x] T028 [US3] Emit preview artifacts showing processed prior, prediction, and ground truth.
- [x] T028a [US3] Emit validation preview grids showing raw minimap, teacher mask/confidence, processed prior, prediction, truth, error, and loss weight.
- [x] T028b [US3] Train from visibility-audited curation manifests so weak/mismatched teacher-mask rows can be excluded.
- [x] T028c [US3] Add optional V18 normal-guidance loss derived from predicted height, without adding a normal output head.
- [x] T028d [US3] Use validation loss for LR plateau scheduling, resume LR override, and best-checkpoint selection without backpropagating validation data.
- [x] T028e [US3] Add optional training-batch hard-error weighting from detached absolute height residuals; keep validation abs-error diagnostic only.
- [x] T029 [US3] Run a bounded smoke training proof and record the outputs in the spec or architecture note.
- [x] T029a [US3] Correct the augmentation contract for baked minimap RGB: default to shadow-safe identity-only augmentation, keep D4 only as an explicit ablation, and update operator docs.
- [x] T029c [US3] Add a precomputed albedo guidance sidecar builder and train-time `--albedo-path` consumption so full albedo runs use reviewed fixed inputs.
- [x] T029d [US3] Add opt-in anti-grid base-model controls (`--model-norm group`, `--decoder-upsample nearest`) after the albedo run plateaued, while preserving legacy defaults for old checkpoints.
- [x] T029e [US3] Add a RunPod cloud-training package builder that copies Python training code plus derived teacher-prior/V18/albedo/curation artifacts only, excluding game-client roots.
- [x] T029f [US3] Add a RunPod REST setup helper for the RTX 4000 Ada training Pod, default network-volume storage, and `runpodctl` send/receive bootstrap using `RUNPOD_API_KEY` only for compute/volume creation; GPU fallbacks are opt-in only.
- [x] T029g [US3] Fix height trainer loss gates to prefer `object_precise_mask` and require precise masks in RunPod V18 slim bundles so cloud training cannot silently use non-precise object gates.
- [x] T029b1 [US3] Define the H0/H1 coarse-to-fine residual chain in `spec.md`, `plan.md`, `data-model.md`, the architecture note, and this task list.
- [x] T029b2 [US3] Add `V18HeightCoarseModel` and `V18HeightResidualModel` aliases/exports under `data-harvester/src/harvester/` without changing legacy `V18HeightModel` checkpoints.
- [x] T029b3 [US3] Add shared residual-chain helpers for model-input assembly, coarse target downsampling, coarse upsampling, residual target construction, and composed-height reconstruction.
- [x] T029b4 [US3] Add `scripts/train_height_coarse_prior.py` for H0 `height_coarse_65` training with its own metrics/checkpoints.
- [x] T029b5 [US3] Add `scripts/train_height_residual_prior.py` for H1 `height_delta_257` training from a frozen H0 checkpoint.
- [x] T029b6 [US3] Add pytest coverage for H0/H1 shapes, residual composition, and checkpoint metadata assumptions.
- [x] T029b7 [US3] Update RunPod packaging/user-guide commands so cloud bundles include H0/H1 scripts and shell wrappers.
- [ ] T029b8 [US3] Run a bounded H0/H1 smoke validation and record outputs or blockers.

**Checkpoint**: The first terrain proof is a tiny height-only lane, not a combined terrain model.

---

## Phase 5: User Story 4 - ADT-Free Object Explanation (Priority: P2)

**Goal**: Explain object coverage and asset candidates from raw minimap without ADT placements.

**Independent Test**: A bounded development-map inference run emits predicted object masks, asset candidates, and processed priors without reading ADT placement arrays.

### Tests for User Story 4

- [x] T030 [P] [US4] Add pytest coverage for minimap-only object-mask dataset generation under `wow-viewer/data-harvester/tests/`.
- [x] T031 [P] [US4] Add tests for asset-candidate matching output structure and confidence sorting.

### Implementation for User Story 4

- [x] T032 [US4] Define the minimap-only object-mask output contract in a Python module under `wow-viewer/data-harvester/src/harvester/`.
- [x] T033 [US4] Define the asset-candidate matching/classification contract keyed to the object library.
- [ ] T034 [US4] Add the smallest viable object-mask training/inference lane under `wow-viewer/data-harvester/scripts/`.
- [x] T035 [US4] Add the smallest viable asset-candidate lane under `wow-viewer/data-harvester/scripts/`.
- [x] T036 [US4] Restrict first pose outputs to XY and yaw in the inference contracts and review artifacts.
- [x] T037 [US4] Generate processed priors from minimap-only object predictions.
- [ ] T038 [US4] Run a bounded development-map proof using minimap plus PM4 context and record review artifacts.

**Checkpoint**: The engine can produce a processed prior without any ADT teacher at runtime.

---

## Phase 6: User Story 5 - Normal Follow-On (Priority: P3)

**Goal**: Add normals only after the height proof is accepted.

**Independent Test**: Analytic normals are generated from predicted height, and any later normal model has its own separate smoke proof.

### Tests for User Story 5

- [x] T039 [P] [US5] Add tests for deterministic height-to-normal derivation utilities.
- [x] T040 [P] [US5] Add tests for any separate normal-refinement dataset contract if that phase is opened.

### Implementation for User Story 5

- [x] T041 [US5] Add a documented analytic-normal baseline path for height predictions.
- [x] T042 [US5] Decide whether analytic normals are sufficient before any model work begins.
- [ ] T043 [US5] If needed, add a separate normal-refinement dataset and trainer under `wow-viewer/data-harvester/`.
- [ ] T044 [US5] Validate the normal follow-on lane independently from the height lane.

**Checkpoint**: Normals remain a separate, optional lane and never force a joint terrain model.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: no dependencies
- **Phase 2**: depends on Phase 1 documentation freeze
- **Phase 3**: depends on usable object-library outputs from Phase 2
- **Phase 4**: depends on teacher-prior dataset from Phase 3
- **Phase 5**: depends on accepted height-only contract from Phase 4
- **Phase 6**: blocked until Phase 4 is validated

### User Story Dependencies

- **US1** blocks every later story
- **US2** blocks height-only training and minimap-only supervision generation
- **US3** is the first MVP terrain proof
- **US4** depends on US1 and US2 outputs but not on normals
- **US5** is explicitly deferred until US3 is validated

### Parallel Opportunities

- Documentation tasks in Phase 1 marked `[P]` can run in parallel
- Test-writing tasks within each phase marked `[P]` can run in parallel
- Within US1, schema tests and capture-artifact tests can run in parallel
- Within US2, dataset parity tests and suppression-behavior tests can run in parallel
- Within US4, object-mask and asset-candidate test scaffolds can run in parallel

## Implementation Strategy

### MVP First

1. Complete Phase 1 documentation freeze.
2. Complete US1 per-object library.
3. Complete US2 teacher-prior generation.
4. Complete US3 height-only terrain proof.
5. Stop and validate before touching minimap-only inference or normals.
