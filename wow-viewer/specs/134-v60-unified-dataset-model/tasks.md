# Tasks: V60 Unified Dataset and Shadow-First Terrain Model

**Branch**: `134-v60-unified-dataset-model` | **Date**: 2026-08-05 | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

## Phase Overview

| Phase | Story | Priority | Goal | Status |
|-------|-------|----------|------|--------|
| 1 | US1 | P1 | Unified v60 dataset | Pending |
| 2 | US2 | P1 | Curation fix | Pending |
| 3 | US3 | P1 | Shadow→height model | Pending |
| 4 | US4 | P2 | Release v0.5.2 | Pending |

## Dependency Graph

```
Phase 1 (v60 store) ──> Phase 2 (curation) ──> Phase 3 (model)

Phase 4 (release) ── independent, last
```

## Phase 1 — Unified v60 dataset (US1, P1)

**Goal**: Single v60 Zarr store with all signals, including the new `terrain_shadow_256`.

**Independent test**: A single v60 store exists with all signals from both Kalimdor and Azeroth, deterministic across two runs.

**Tasks**:

- [ ] T001 [P] Update the frozen signal catalog (`docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`) to add `terrain_shadow_256`, `signal_class`, `surviving_height_levels`, and bump the release to v60.1.
- [ ] T002 [P] Create `wow-viewer/data-harvester/scripts/v60_build_unified_store.py` — CLI that reads all existing v50.1 Zarr stores + archaeology stores, merges them into a single v60 Zarr store with a unified index.parquet. Handles schema differences and missing signals gracefully (records unavailable-with-reason, never silently zero-fills).
- [ ] T003 [P] Create `wow-viewer/data-harvester/src/harvester/v50/v60_store.py` — library module for the v60 store builder and manifest (reuses `store.py` conventions: staging dir, atomic replace, finalize).
- [ ] T004 [P] Create `wow-viewer/data-harvester/tests/v50/test_v60_store.py` — unit tests: merge two small fixture stores, schema reconciliation, missing-signal handling, determinism.
- [ ] T005 [US1] **USER RUNS** re-harvest: run the spec-133-updated harvest tool on Kalimdor and Azeroth to produce NPZ shards containing `terrain_shadow_256`.
- [ ] T006 [US1] Build the v60 store from the re-harvested NPZ shards (user runs `v60_build_unified_store.py`).
- [ ] T007 [US1] Update `v50_tile_classify.py` / `v50_tile_inventory.py` output paths to read from the v60 store.

**Checkpoint**: A single v60 store exists with `terrain_shadow_256`, `signal_class`, and `surviving_height_levels` for every tile in Kalimdor and Azeroth.

## Phase 2 — Curation fix (US2, P1)

**Goal**: Training curriculum with surviving_height_levels gating.

**Independent test**: The curriculum excludes ≤64-level tiles and admits compressed-rich tiles, deterministically.

**Tasks**:

- [ ] T008 [P] Add `--min-height-levels` / `--max-height-levels` options to `wow-viewer/data-harvester/src/harvester/v50/training_curriculum.py` that filter tiles by `surviving_height_levels`.
- [ ] T009 [P] Add `--curation-height-levels` gating to the curriculum builder script (`v50_build_training_curriculum.py`).
- [ ] T010 [P] Create `wow-viewer/data-harvester/tests/v50/test_curriculum_height_levels.py` — unit tests for the gating logic.
- [ ] T011 [US2] **USER RUNS** rebuild the curriculum from the v60 store with the curation fix applied.

**Checkpoint**: A v60 curriculum exists with the curation fix; the excluded/admitted tile lists are correct by inspection.

## Phase 3 — Shadow→height model (US3, P1)

**Goal**: A model that takes `terrain_shadow_256 → height_257` and beats the tile-mean baseline.

**Independent test**: Trained checkpoint with val_mae < 0.142 (5% below the 0.1493 baseline).

**Tasks**:

- [ ] T012 [P] Create `wow-viewer/data-harvester/scripts/v60_train_shadow_height.py` — training script that reads `terrain_shadow_256` (1 channel, 256x256) as input and `height_257` as target. Reuses `direct_cnn_v112` with `in_channels=1` and the v112.1 relative-height contract.
- [ ] T013 [P] Add a 1-channel input builder to `wow-viewer/data-harvester/src/harvester/v50/height_relative_evaluate.py` (or a new `build_model_input_channels` path for shadow-only input).
- [ ] T014 [P] Create `wow-viewer/data-harvester/tests/v50/test_shadow_height_model.py` — unit tests: 1-channel forward pass, target contract, determinism.
- [ ] T015 [US3] **USER RUNS** training (exact command in plan.md Phase 3).
- [ ] T016 [US3] Evaluate against the 0.1493 baseline and record the result.

**Checkpoint**: A trained checkpoint with val_mae < 0.142, reproducibly.

## Phase 4 — Release v0.5.2 (US4, P2)

**Goal**: Release v0.5.2, merge branches, update docs, start new dev branch.

**Independent test**: GitHub Release published; main contains all committed work; README current.

**Tasks**:

- [ ] T017 [P] Update `wow-viewer/README.md` — current focus section now reflects the v60 dataset, shadow→height model, and unbaked minimap decomposition.
- [ ] T018 [P] Update `wow-viewer/docs/WoWViewer/USERGUIDE.md` — add the new signals and the shadow→height model usage.
- [ ] T019 [P] Update `wow-viewer/docs/releases/v0.5.2.md` — finalize release notes with the current state.
- [ ] T020 [P] Merge branches 131, 132, 133 into main (after each is validated).
- [ ] T021 [P] Tag `v0.5.2` and push the tag to trigger CI release.
- [ ] T022 [P] Create a new dev branch off main for continued work.

**Checkpoint**: GitHub Release published with v0.5.2 binaries; README and docs current; main contains all work.

---

## Parallel execution opportunities

- T001-T004 (catalog + store builder + tests) can be done in parallel.
- T008-T010 (curation) depend on T002 (v60 store) existing but can be written in parallel with the store builder.
- T012-T014 (model) depend on the curriculum (Phase 2) for training but the code can be written in parallel.
- T017-T022 (release) are independent of Phases 1-3 and can run in parallel.

## MVP scope

Phases 1-3 (v60 store + curation + shadow→height model) are the P1 deliverables. Phase 4 (release) is P2. The model milestone (val_mae < 0.142) is the primary success criterion.