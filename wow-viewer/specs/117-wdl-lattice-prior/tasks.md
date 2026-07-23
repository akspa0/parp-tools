---
description: "Task list for Spec 117 WDL-lattice coarse prior for terrain geometry"
---

# Tasks: WDL-Lattice Coarse Prior for Terrain Geometry

**Input**: Design documents from `/specs/117-wdl-lattice-prior/`

**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included — project convention is focused tests per slice, run with `uv run python -m pytest tests/spec117/ -q`.

**Organization**: Tasks grouped by user story in spec priority order (US1 P1 export, US2 P1 learnability, US3 P2 integration). Each story is independently testable per its own Independent Test in spec.md. The user runs every training/heavy step (FR-011); tasks that produce user-run CLIs hand off the invocation, they do not launch it.

**Implementation-time finding carried into these tasks**: the C# harvester (`RawArraySerializer.WriteTerrainVertexArrays`) already computes and streams the 545-point WDL lattice as `wdl_outer_17`/`wdl_inner_16`/`wdl_outer_present`/`wdl_inner_present` in every stream profile (Full/V16/V22) — `TerrainWdlLattice` is already wired into `AdtTensorPackBuilder`. No new C# code is required for US1. The only real gap is that these four arrays are not yet in the v50 store's frozen signal catalog, so the existing 1:1-name store builder (`scripts/v50_build_dataset.py::_cmd_build`) never selects them. US1's tasks are therefore catalog/config wiring, not a harvester change, and the store array names used throughout are the REAL stream names above (not the placeholder `wdl_lattice_outer17` etc. named in data-model.md before this was discovered — corrected in the Polish phase).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- All paths are relative to `wow-viewer/` unless stated otherwise (Python paths relative to `wow-viewer/data-harvester/`)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `harvester.spec117` package skeleton.

- [x] T001 Create package `data-harvester/src/harvester/spec117/__init__.py` (empty namespace, mirrors `harvester.spec116` layout)
- [x] T002 [P] Create test package `data-harvester/tests/spec117/__init__.py`

**Checkpoint**: Package importable.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Widen the reused `v50-model-stage-run-v1` schema and build the spec-scoped contract helpers every later story needs. MUST complete before any user story.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [x] T003 Widen `STAGES` in `data-harvester/src/harvester/v50/model_stage_contract.py` to add `"lattice_prior"` (data-model.md Run-Record Schema reuses `v50-model-stage-run-v1` verbatim per research.md D-01's "no new schema" philosophy; the stage enum is the one thing that must actually change for that reuse to validate)
- [x] T004 [P] Test the widened enum in `data-harvester/tests/spec117/test_model_stage_contract_lattice_prior.py`: a well-formed `stage="lattice_prior"` document now validates via `harvester.v50.model_stage_contract.validate_model_stage_run`; an unlisted stage still rejects
- [x] T005 Implement `data-harvester/src/harvester/spec117/lattice_contract.py`: constants (`STAGE="lattice_prior"`, `OUTPUT_SIGNAL="wdl_lattice_545"`, `OUTER_DIM=17`, `INNER_DIM=16`, `SAMPLE_COUNT=545`), `architecture_identity(model)` (id/config_sha256 via `sha256_json`/parameter_count, mirrors `direct_geometry_model.architecture_identity`), `build_lattice_stage_run(...)` assembling + self-validating a `v50-model-stage-run-v1` document, and re-exported `sha256_file`/`identity_for_path` from `harvester.v50.model_stage_contract` (single canonical owner, no duplication)
- [x] T006 [P] Test `data-harvester/tests/spec117/test_lattice_contract.py`: a well-formed run summary built by `build_lattice_stage_run` validates; a bad `stage`/`output_signal` is rejected with a clear error

**Checkpoint**: Foundation ready — schema and identity helpers work; user story implementation can begin.

---

## Phase 3: User Story 1 — export the lattice as a real signal (Priority: P1) 🎯 MVP

**Goal**: The 545-point WDL lattice becomes a first-class, readable v50 store signal for every tile with real height ground truth. No model.

**Independent Test**: Export against the corrected v50 curriculum store and confirm every tile with real `height_257` produces 545 finite lattice samples, with unexportable tiles excluded and counted (spec US1 acceptance 1–2).

### Implementation for User Story 1

- [x] T007 [US1] Add four rows to the frozen catalog table in `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`, using the REAL harvest-stream array names already emitted by `RawArraySerializer.WriteTerrainVertexArrays`: `wdl_outer_17` float32 (17,17) copy-if-verified not-required; `wdl_inner_16` float32 (16,16) copy-if-verified not-required; `wdl_outer_present` bool (17,17) copy-if-verified not-required; `wdl_inner_present` bool (16,16) copy-if-verified not-required. Note in the table that presence arrays mark per-sample gaps (never fabricated) per spec FR-001/Edge Cases.
- [x] T008 [US1] Regenerate both derived config files from the updated catalog (no hand-editing): `uv run python scripts/v50_generate_manifest_template.py --catalog-doc docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md --build-id 0_5_3_3368 --release v50.1 --output v50_configs/v50-manifest-template-0_5_3_3368.json --signals-output v50_configs/v50-signals-0_5_3_3368.json` from `data-harvester/`. This makes the existing 1:1-name extraction in `scripts/v50_build_dataset.py::_cmd_build` pick up the four new arrays with zero new ingestion code — the store writer, row-lineage tracking, and "excluded and counted, never fabricated" behavior are already generic and already tested.
- [x] T009 [US1] [P] Test `data-harvester/tests/spec117/test_wdl_lattice_signal.py`: `harvester.v50.signal_catalog.parse_catalog_table` over the real doc yields all four new entries with the documented dtype/shape/policy; confirm the existing `tests/v50/test_manifest_template_matches_catalog.py::test_committed_053_template_matches_the_frozen_catalog` still passes unmodified after T008's regeneration (i.e. run it, do not edit it — it is the drift guard this task must satisfy, not duplicate)

**Checkpoint**: US1 delivers the exported signal end-to-end through the existing generic store pipeline (SC-001). No model trained. A real rebuild against `H:\CLIENTS` is the user's next heavy step (not run here).

---

## Phase 4: User Story 2 — standalone minimap-only lattice predictor (Priority: P1)

**Goal**: Prove the 545-point lattice is learnable from minimap RGB alone before any chain-integration work is attempted, scored only on the honest held-out split.

**Independent Test**: Train the standalone predictor, score held-out lattice-point MAE against the trivial per-tile-mean lattice baseline, and read a plain learnable/not-learnable verdict (spec US2 acceptance 1–3).

### Implementation for User Story 2

- [x] T010 [US2] Implement `data-harvester/src/harvester/spec117/lattice_model.py`:
  - `encode_lattice_target(outer, inner, outer_present, inner_present) -> (target[545] float32 in [0,1], mask[545] float32 {0,1}, tile_min, tile_max)`: per-tile min/max computed ONLY over present samples (never fabricates absent ones), same `RANGE_FLOOR=1.0` floor convention as `height_relative_model.encode_relative_height` for consistency across the v112.1 target family; raises if zero samples are present (caller must exclude that row first, never silently degenerate).
  - `decode_lattice_target(...)`: inverse of the above.
  - `compute_lattice_tile_mean_baseline(targets_and_masks) -> float`: masked MAE of predicting each tile's own mean-of-present-samples for its present samples only (D-02's tile-mean concept, adapted for sparse presence — `height_relative_train.compute_tile_mean_baseline` assumes a fully finite array and cannot be reused as-is).
  - `select_lattice_rows(group, rows) -> (usable_rows, excluded_count)`: drops rows where `wdl_outer_present`/`wdl_inner_present` are all False (spec Edge Cases: "a held-out tile with no exportable lattice at all — excluded ... and counted").
  - `lattice_loss(predicted, target, mask)`: masked smooth-L1.
  - `LatticeNet(base=24)`: lean conv encoder (256x256x3 RGB in) + two small pooled heads producing the flattened (17,17) and (16,16) sigmoid outputs, concatenated to 545. Independently weighted from every other stage (constitution IV / FR-007). Record its real parameter count in the architecture identity rather than aiming for a specific number.
- [x] T011 [US2] [P] Test `data-harvester/tests/spec117/test_lattice_model.py`: encode/decode round-trips exactly at present samples and ignores absent ones; `tile_min`/`tile_max` are computed only from present samples on a fixture with a deliberate gap; `LatticeNet` forward returns shape `(B, 545)` bounded in `[0,1]`; `select_lattice_rows` excludes an all-absent fixture tile and reports it in the excluded count, not as zero rows silently
- [x] T012 [US2] Implement `data-harvester/src/harvester/spec117/lattice_train.py`: dry-run-first CLI. Reuses (imports, does not reimplement) `harvester.v50.height_relative_train.{validate_curriculum_contract, validate_source_selection, select_training_rows, SOURCE_CHOICES, curriculum_identity, require_new_output, TrainerContractError}` and `harvester.v50.direct_geometry_train.apply_held_out_split`. Unlike the existing geometry trainers, `--held-out-split` is REQUIRED with no `--val-key`/`--val-value` fallback (FR-004: "MUST refuse to run against a leaky or unspecified split"). Refuses closed with a clear message when the store lacks `wdl_outer_17`/`wdl_inner_16`/`wdl_outer_present`/`wdl_inner_present` (expected until a US1 rebuild lands). Computes `compute_lattice_tile_mean_baseline` on held-out rows before training. Trains `LatticeNet`; on completion writes `training_plan.json`, `run_identity.json`, `checkpoint_best.pt`, and `model_stage_run.json` via `lattice_contract.build_lattice_stage_run` (`promotion_verdict="pending"`, `upstream_models=[]`).
- [x] T013 [US2] [P] Test `data-harvester/tests/spec117/test_lattice_train.py` (CPU-only, no CUDA requirement — uses a tiny in-memory fixture store): dry-run plan prints valid JSON and exits 0 without `--confirm-run`; refuses a held-out split with `verified_violation_count != 0`; refuses when `--held-out-split` is omitted; refuses a store missing the wdl arrays with a clear message; `compute_lattice_tile_mean_baseline` matches a hand-computed value on a small fixture
- [x] T014 [US2] Implement thin CLI `data-harvester/scripts/spec117_train_lattice.py` (sys.path insert + `main()` passthrough, mirrors `scripts/v50_train_wdl_prior.py`)

**Checkpoint**: US2 delivers the learnable/not-learnable verdict (SC-002) before any integration code exists.

---

## Phase 5: User Story 3 — feed the generated lattice into the existing chain (Priority: P2)

**Goal**: Bridge the frozen predictor's generated (never ground-truth) output into the existing `--feature-store` contract so the already-validated coarse/detailer trainers can consume it with zero trainer changes (research.md D-01).

**Independent Test**: Materialize a frozen checkpoint's predictions into a `v115-feature-map-v1`-shaped store and confirm the existing trainers accept it unmodified; report relief-region MAE per feed point against the pre-existing real baseline (spec US3 acceptance 1–3).

### Implementation for User Story 3

- [x] T015 [US3] Implement `data-harvester/src/harvester/spec117/lattice_bridge.py`: `lattice_to_feature_map(store, checkpoint, output, write=False)` runs the frozen `LatticeNet` over every row's `minimap_rgb`, builds a dense `(256,256)` field per tile by independently bilinear-upsampling the `(17,17)` outer and `(16,16)` inner sigmoid outputs to `256x256` (`align_corners=True`) and averaging them (documented approximation — the true lattice is a quincunx-offset sparse grid, not one regular grid; averaging both grids' full-extent reconstructions is the simple, dependency-free choice consistent with this project's time-to-signal preference), writes a `(N,1,256,256)` float16 `feature_map` array under `schema="v115-feature-map-v1"`, `class_count=1`, `attrs.source_signal="wdl_lattice"`, checkpoint path+sha256 binding (mirrors `structure_feature_bridge.py`'s pattern exactly). Source store is never mutated; output is refused if non-empty.
- [x] T016 [US3] [P] Test `data-harvester/tests/spec117/test_lattice_bridge.py`: dry-run (`write=False`) returns a plan only, writes nothing; `write=True` produces the exact schema/shape/attrs and a row-aligned `index.parquet`; checkpoint sha256 in attrs matches the real file; source store bytes are unchanged after the call; refuses to overwrite a non-empty output directory
- [x] T017 [US3] Implement thin CLI `data-harvester/scripts/spec117_lattice_to_feature_map.py` (mirrors `scripts/spec116_structure_to_feature_map.py`)
- [x] T018 [US3] Regression-prove the US3(ii) handoff needs no trainer changes: point `v50_train_direct_geometry.py --feature-store <T015 output>` and `v50_train_geometry_detailer.py --feature-store <T015 output>` (both dry-run, no `--confirm-run`) at a tiny fixture store built by T016 and confirm both accept the shape/schema with zero code edits to either trainer (proves D-01, does not add a new capability to either script)

**Checkpoint**: US3(i) bridge is real and tested. US3(ii)'s paired with/without training comparison is entirely user-run against the existing trainers per quickstart.md §4 — nothing further to implement.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Reconcile the spec docs (written ahead of code) with what was actually built, and prove no regressions.

- [x] T019 [P] Update `specs/117-wdl-lattice-prior/data-model.md`, `contracts/cli-contract.md`, and `quickstart.md` to use the REAL store array names (`wdl_outer_17`/`wdl_inner_16`/`wdl_outer_present`/`wdl_inner_present`, discovered already-live in the C# stream — not the originally-drafted `wdl_lattice_outer17` etc.) and the REAL argparse of `spec117_train_lattice.py`/`spec117_lattice_to_feature_map.py`, per this project's "verify docs against real argparse" convention
- [x] T020 Run full validation from `data-harvester/`: `uv run python -m pytest tests/spec117/ tests/v50/ -q`, `uv run ruff check src/harvester/spec117 src/harvester/v50/model_stage_contract.py scripts/spec117_*.py`, `uv run python -m py_compile src/harvester/spec117/*.py scripts/spec117_*.py`, then the full suite (`uv run python -m pytest -q`) to confirm no regressions beyond the pre-existing unrelated failures
- [x] T021 Update `memory-bank/activeContext.md` and `memory-bank/progress.md` with the Spec 117 US1–US3(i) implementation summary, including the naming-convention correction and the "no C# change needed" finding

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories (US2/US3 both need the widened schema + `lattice_contract.py`).
- **US1 (Phase 3)**: Depends on Foundational only. Independent of US2/US3 — pure catalog/config wiring.
- **US2 (Phase 4)**: Depends on Foundational. Does NOT depend on US1's catalog change to be *implemented* (the code path is written and tested against fixtures either way), but a REAL run requires a store rebuilt after US1 lands.
- **US3 (Phase 5)**: Depends on Foundational and on US2's `LatticeNet`/checkpoint format (`lattice_bridge.py` loads a `lattice_train.py`-produced checkpoint). Not on a real US1 rebuild to be *implemented/tested* (fixtures cover it), only to be run for real.
- **Polish (Phase 6)**: Depends on US1–US3 all being implemented.

### Parallel Opportunities

- T002, T004, T006 (test-package/foundational tests) can run in parallel with their sibling implementation tasks once the corresponding implementation lands.
- T009 (US1 test) is parallel with T010–T014 (US2) — different files, no shared dependency.
- T011, T013 (US2 tests) parallel with each other once T010/T012 land respectively.
- T016 (US3 test) parallel with T019 (doc reconciliation).

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 + Phase 2 (Setup + Foundational).
2. Complete Phase 3 (US1): catalog/config wiring — the signal exists end-to-end through the generic pipeline.
3. **STOP and VALIDATE**: run T009's test; the four arrays parse correctly and the drift guard still passes.

### Incremental Delivery

1. Setup + Foundational → schema/contract ready.
2. US1 → signal plumbed (data only, no model) → validate independently.
3. US2 → standalone predictor + trainer → validate independently (learnable/not-learnable verdict) before touching US3.
4. US3(i) → bridge implemented + tested → US3(ii) handed off to the user as documented CLI invocations (no further code).
5. Polish → docs reconciled, full suite green, memory updated.
