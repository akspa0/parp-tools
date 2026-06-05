# Tasks: V18 Focused Two-Build Minimap-to-Terrain Loop

**Input**: Design documents from `/specs/047-v18-distill-corpus-open-source-loop/`

**Active Scope**: two focused builds, minimap-only input, plain height and
normal supervision. Renderer truth, distillation, and student release are
deferred.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no incomplete dependency)
- **[Story]**: Which user story this task belongs to

---

## Phase 1: Landed Simplification

**Purpose**: record the simplification work already in the repo.

- [x] T001 [P] Reset spec 047 to the focused two-build lane instead of the wider distill/release loop.
- [x] T002 [P] Simplify `wow-viewer/data-harvester/scripts/train_v16_1_common.py` so the active height and normal losses are plain supervision.
- [x] T003 [P] Keep the normal default on the minimap-only contract (`v16_1_1_base`) instead of the old height/refiner/object-roof variants.
- [x] T004 [P] Sync continuity/docs so spec 047 now means minimap-only height and normal training on two builds.

**Checkpoint**: the repo contract says the same thing the active trainer does.

---

## Phase 2: Focused Corpus Validation

**Purpose**: make sure the basic training lane has honest source data.

- [ ] T005 [US1] Confirm the staged client roots exist under `output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft/` and `output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft/`.
- [ ] T006 [US1] Validate `wow-viewer/output/datasets/v18/0_5_3_3368.zarr` for the minimap/height/normal training contract.
- [ ] T007 [US1] Validate `wow-viewer/output/datasets/v18/3_3_5_12340.zarr` for the minimap/height/normal training contract.
- [ ] T008 [US1] Record the exact focused-corpus evidence: staged roots, store paths, and validation summaries.

**Checkpoint**: the two focused stores are honest inputs for the minimap-only lane.

---

## Phase 3: User Story 2 - Height Training (Priority: P1)

**Goal**: prove `minimap_rgb -> height_257` works on the focused corpus.

**Independent Test**: run a bounded height-training pass and verify checkpoint
plus validation previews land under the V18 model root.

- [x] T010 [US2] Run a bounded focused-corpus height training pass with `wow-viewer/data-harvester/scripts/train_v16_1_height.py`.
- [x] T011 [US2] Save the height checkpoint and evidence under `wow-viewer/models/v18/height/runs/`.
- [x] T012 [US2] Record the exact command, seed, config snapshot, and best validation outputs for rerun.

**Checkpoint**: a reproducible minimap-only height run exists.

---

## Phase 4: User Story 3 - Normal Training (Priority: P1)

**Goal**: prove `minimap_rgb -> normal_xyz` works on the focused corpus.

**Independent Test**: run a bounded normal-training pass and verify checkpoint
plus validation previews land under the V18 model root.

- [x] T013 [US3] Run a bounded focused-corpus normal training pass with `wow-viewer/data-harvester/scripts/train_v16_1_normal.py`.
- [x] T014 [US3] Save the normal checkpoint and evidence under `wow-viewer/models/v18/normal/runs/`.
- [x] T015 [US3] Record the exact command, seed, config snapshot, and best validation outputs for rerun.

**Checkpoint**: a reproducible minimap-only normal run exists.

---

## Phase N: Continuity

**Purpose**: keep the docs honest after proof lands.

- [x] T020 [P] Update `wow-viewer/docs/architecture/v18-distill-corpus-open-source-loop-2026-06-04.md` with the actual focused validation and training evidence.
- [x] T021 [P] Update `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md` after the first bounded height/normal proof lands.
- [ ] T022 [P] Update any README command surfaces that still describe the older renderer/object-mask lane.

**Checkpoint**: future sessions inherit the basic minimap-only lane instead of
the abandoned complicated one.
