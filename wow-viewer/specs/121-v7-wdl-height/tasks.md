# Tasks: V7-Style WDL-Prior Height Reconstruction (Small Model Lane)

**Input**: Design documents from `specs/121-v7-wdl-height/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli-contract.md, quickstart.md

**Execution ownership**: ALL training/materialization on real data is USER RUN (RULE 0). Agent
tasks are code + tests + dry-run verification only.

**Tests**: included per house precedent (specs 116–119 landed with pytest coverage).

## Phase 1: Setup

- [x] T001 Create `harvester/spec121/__init__.py` package skeleton in `wow-viewer/data-harvester/src/harvester/spec121/`
- [x] T002 Create `wow-viewer/data-harvester/tests/spec121/__init__.py` test package skeleton

## Phase 2: Foundational

- [x] T003 Verify v50 store signal prerequisites (script prints presence of `wdl_outer_17`, `wdl_inner_16`, `wdl_outer_present`, `wdl_inner_present`, `object_geometry_visible_mask_257`) as a read-only helper in `wow-viewer/data-harvester/src/harvester/spec121/store_check.py`
- [x] T004 [P] Unit test store_check against a fixture store in `wow-viewer/data-harvester/tests/spec121/test_store_check.py`

## Phase 3: User Story 1 — Stage A: Minimap → WDL Lattice Prior (P1)

**Goal**: `MitB0LatticeNet` + trainer clearing SC-001 (≥15% below tile-mean) on the frozen split.
**Independent Test**: dry-run prints plan; fixture-store training smoke runs one step; run record
contains baseline + param count in band.

- [x] T005 [US1] Implement `MitB0LatticeNet` (SegFormer-B0 encoder + outer/inner 545-value heads, config-only reconstructable, optional `from_pretrained` loader) in `wow-viewer/data-harvester/src/harvester/spec121/lattice_backbone_model.py` — DONE 2026-07-24: 3,469,922 params at default B0 config (inside 3–30M band)
- [x] T006 [P] [US1] Tests: shapes, masked-loss parity with Spec 117 contract, config reconstruction, 3–30M param band, absent-sample exclusion in `wow-viewer/data-harvester/tests/spec121/test_lattice_backbone_model.py`
- [x] T007 [US1] Implement tile-level object-mask coverage weighting (`1 − w·coverage`, warn+disable when array absent) in `wow-viewer/data-harvester/src/harvester/spec121/object_mask_tile_loss.py`
- [x] T008 [P] [US1] Tests: weight math, missing-array degradation, all-object tile finite loss in `wow-viewer/data-harvester/tests/spec121/test_object_mask_tile_loss.py`
- [x] T009 [US1] Implement trainer (required `--held-out-split`, tile-mean baseline, onecycle warmup-aware stale counter via `harvester/v50/lr_schedule.py`, per-epoch previews + final sheets, `--architecture`, `--object-mask-weight`, `--gradient-weight`, dry-run-first `--confirm-run`, `v50-model-stage-run-v1` record stage `"lattice_prior"`) in `wow-viewer/data-harvester/src/harvester/spec121/lattice_backbone_train.py` — default `--release v50.2` (v50.1 signals + lattice + object-mask arrays)
- [x] T010 [US1] Thin CLI in `wow-viewer/data-harvester/scripts/spec121_train_lattice_prior.py`
- [x] T011 [P] [US1] Tests: dry-run plan contents, refusal without split, baseline recorded, mask-weight echo in record in `wow-viewer/data-harvester/tests/spec121/test_lattice_backbone_train.py`
- [x] T012 [US1] Dry-run smoke on fixture store + full data-harvester pytest run (ruff clean) — no real training — DONE 2026-07-24: 30/30 spec121 tests; full suite 1136 passed / 3 pre-existing failures (v24 export-map, 2× v25 h1_coarse — unchanged); ruff+compileall clean; real dry-run on `curriculum-0_5_3_3368-dual_v3.zarr` + spec116 split: violation_count=0, lattice arrays present, object-mask absent → graceful `signal_present=false`, exits without training
- [ ] T013 [US1] USER RUN: real Stage A training per quickstart §1; record G1 verdict (SC-001) in `wow-viewer/specs/121-v7-wdl-height/research.md`

## Phase 4: User Story 2 — Stage B: Residual Detailer Over the Prior (P1)

**Goal**: predicted prior → coarse store → existing detailer trainer; SC-002 ≥9% below prior-only.
**Independent Test**: bridge output passes `validate_coarse_store`; detailer dry-run accepts it
with zero trainer changes (U-Net path).

- [ ] T014 [US2] Implement prior→coarse-store bridge (batch Stage A inference, bilinear outer/inner upsample+average per Spec 117 bridge rule, coarse-store schema + attrs with checkpoint sha256) in `wow-viewer/data-harvester/src/harvester/spec121/prior_coarse_bridge.py`
- [ ] T015 [US2] Thin CLI (`--write` gate) in `wow-viewer/data-harvester/scripts/spec121_bridge_prior_to_coarse.py`
- [ ] T016 [P] [US2] Tests: schema acceptance by `validate_coarse_store`, provenance attrs, dry-run report in `wow-viewer/data-harvester/tests/spec121/test_prior_coarse_bridge.py`
- [ ] T017 [US2] Implement `DetailerMitB0Net` + `--architecture {detailer_unet_v1,detailer_mit_b0_v1}` (default unchanged = parity) in `wow-viewer/data-harvester/src/harvester/v50/geometry_detailer_model.py` and `wow-viewer/data-harvester/src/harvester/v50/geometry_detailer_train.py`
- [ ] T018 [P] [US2] Tests: both archs config-reconstructable, zero-init residual passthrough, param band, default parity in `wow-viewer/data-harvester/tests/v50/test_geometry_detailer_model.py`
- [ ] T019 [US2] Integration proof: fixture store + random-init Stage A checkpoint → bridge `--write` → dry-run `v50_train_geometry_detailer.py --coarse-store` accepted (both archs)
- [ ] T020 [US2] USER RUN: real bridge + Stage B training per quickstart §2–3; record G2 verdict (SC-002) + GT-prior ablation in `wow-viewer/specs/121-v7-wdl-height/research.md`

## Phase 5: User Story 3 — Object Masks as Loss Signal, Paired Comparison (P2)

**Goal**: mask-weight 0 vs 1 paired runs per stage; verdict recorded; null = valid close (SC-003).
**Independent Test**: paired run records differ only in the weight flag; touched/untouched MAE
present in each.

- [ ] T021 [US3] USER RUN: Stage A paired run (`--object-mask-weight 1.0`) per quickstart §4
- [ ] T022 [US3] USER RUN: Stage B paired run (`--object-mask-weight 1.0`) per quickstart §4
- [ ] T023 [US3] Write comparison verdict table (helps/hurts/null per stage, touched vs untouched MAE) into `wow-viewer/specs/121-v7-wdl-height/research.md`

## Phase 6: User Story 4 — End-to-End Chain + Visual Gate (P3)

**Goal**: minimap-only chain materializer; sheets for held-out map + OOD tile; user visual verdict
(SC-005).
**Independent Test**: chain runs reading no ground-truth array; audit JSON names both checkpoint
sha256s.

- [ ] T024 [US4] Implement chain materializer (Stage A → bridge field → Stage B → final height sheets, `--store` batch mode XOR `--inputs` OOD mode, `--write` gate) in `wow-viewer/data-harvester/src/harvester/spec121/chain_materialize.py`
- [ ] T025 [US4] Thin CLI in `wow-viewer/data-harvester/scripts/spec121_materialize_chain.py`
- [ ] T026 [P] [US4] Tests: no-ground-truth read assertion, provenance audit fields, OOD mode flag in `wow-viewer/data-harvester/tests/spec121/test_chain_materialize.py`
- [ ] T027 [US4] USER RUN: chain sheets per quickstart §5; user issues visual verdict; flip `promotion_verdict` only on pass

## Phase 7: Polish & Bookkeeping

- [ ] T028 [P] Move `wow-viewer/specs/119-object-library-classifier/` to `wow-viewer/specs/archived/` with a closure note citing the retrieval PoC negative (p50=10px instances)
- [ ] T029 [P] Move `wow-viewer/specs/120-minimap-placement-retrieval/` to `wow-viewer/specs/archived/` with a closure note citing the same scale-physics negative
- [ ] T030 Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` (spec 121 active, 119/120 archived, gate verdicts)
- [ ] T031 Full data-harvester pytest + ruff + compileall clean; record counts in research.md

## Dependencies

- T005–T013 (US1) block T014–T020 (US2: bridge needs a Stage A checkpoint class; real runs need G1)
- T017 is independent of T014–T016 and may run in parallel
- T021–T023 (US3) depend on T013 + T020 (need trained checkpoints)
- T024–T027 (US4) depend on T020 (need Stage B checkpoint)
- T028–T031 are independent of all story phases

## Parallel Execution Examples

- T006 ∥ T008 ∥ T011 (test files, different modules)
- T014+T015 ∥ T017+T018 (bridge vs detailer trunk, different files)
- T028 ∥ T029 ∥ T030 (bookkeeping, different files)

## Implementation Strategy

MVP = Phase 3 alone (US1): if Stage A clears G1 the lane is alive; if not, a recorded negative
closes it cheaply before any Stage B work. Each subsequent phase is independently testable per
the criteria above. No phase starts before its gate verdict is recorded (One Phase at a Time).
