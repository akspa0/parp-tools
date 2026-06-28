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

### Implementation for User Story 2

- [x] T016 [US2] Add a Python library module for teacher-prior generation under `wow-viewer/data-harvester/src/harvester/`.
- [x] T017 [US2] Add a CLI script under `wow-viewer/data-harvester/scripts/` that reads V18 Zarr stores and writes the teacher-prior dataset.
- [x] T018 [US2] Make the prior generator explicitly prefer filtered precise/object-filtered masks over coarse rectangle roof masks where available.
- [x] T019 [US2] Add processed-prior review-artifact rendering under `wow-viewer/data-harvester/scripts/`.
- [x] T020 [US2] Document the exact phase-1 prior channels in the generated dataset metadata.
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
- [ ] T029 [US3] Run a bounded smoke training proof and record the outputs in the spec or architecture note.

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
