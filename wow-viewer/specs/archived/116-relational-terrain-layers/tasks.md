---
description: "Task list for Spec 116 relational terrain layer reconstruction"
---

# Tasks: Relational Terrain Layer Reconstruction

**Input**: Design documents from `/specs/116-relational-terrain-layers/`

**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included — the project convention is focused tests per slice (every acceptance scenario is testable). Tests are written alongside each slice and run with `uv run python -m pytest tests/spec116/ -q`.

**Organization**: Tasks grouped by user story in spec priority order (US1 P1, US2 P1, US4 P2, US3 P2, US5 P3). Each story is independently testable. The user runs every training/heavy step (FR-018); tasks that produce user-run CLIs hand off the invocation, they do not launch it.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- All paths are relative to `wow-viewer/data-harvester/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `harvester.spec116` package and shared contract validator.

- [ ] T001 Create package `src/harvester/spec116/__init__.py` (empty namespace, mirrors `harvester.v50` layout)
- [ ] T002 [P] Create test package `tests/spec116/__init__.py` and a `conftest.py` with a tiny in-memory v50-style Zarr fixture (mcly_texture_ids, mcly_layer_mask, height_257, minimap_rgb, index.parquet) reusable across all story tests
- [ ] T003 Add `scikit-learn` and `scipy` to `data-harvester/pyproject.toml` dev/optional deps if absent (US2 non-linear fit + bimodality test); run `uv sync` and record the lockfile change

**Checkpoint**: Package importable; fixture builds; deps resolve.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared relational extraction + JSON schema validators used by multiple stories. MUST complete before any user story.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [ ] T004 Implement `structure_contract.py` in `src/harvester/spec116/structure_contract.py`: validators for `v50-held-out-split-v1`, `v116-analysis-report-v1`, `v50-structure-run-v1`, `v50-structure-infer-v1`, `v50-structure-geometry-comparison-v1` against the schemas in `contracts/`, plus `sha256` identity helpers (reuse `harvester.v50.model_stage_contract` pattern)
- [ ] T005 [P] Test `structure_contract.py` in `tests/spec116/test_structure_contract.py`: each schema accepts a valid fixture and rejects a tampered one (wrong schema const, bad sha256, nonzero violation count, aggregate-accuracy gate)
- [ ] T006 Implement `relational_extract.py` in `src/harvester/spec116/relational_extract.py`: extract layer-entry rows `(tile_row, chunk_y, chunk_x, slot, local_texture_id, coverage, family)` from a v50 store by joining `mcly_texture_ids` → texture-name dump → `harvester.v50.terrain_feature_labels.classify_texture_name`; exclude rows with no dump entry / empty MTEX table and count them (never emit all-unknown)
- [ ] T007 Test `relational_extract.py` in `tests/spec116/test_relational_extract.py`: row count matches store rows; base slot marked opaque; excluded rows counted; family ordinals match the `v115.1` taxonomy on a known fixture

**Checkpoint**: Foundation ready — relational extraction and contract validation work; user story implementation can begin.

---

## Phase 3: User Story 1 — family→slot consistency (Priority: P1) 🎯 MVP

**Goal**: Measure how consistently each surface family occupies each layer slot; emit a durable vocabulary decision (`slot_keyed`/`family_keyed`) consumed by US3. No model trained.

**Independent Test**: Run the CLI over the real v50 store and read the reported family→slot distribution + summary consistency score + recommendation (spec US1 acceptance 1–2).

### Implementation for User Story 1

- [ ] T008 [US1] Implement `family_slot_consistency.py` in `src/harvester/spec116/family_slot_consistency.py`: per-family slot distribution `P(s|f)`, summary consistency score = mean over families of `max_s P(s|f)`, threshold (default 0.70), recommendation; build the `v116-analysis-report-v1` artifact (report_kind=`family_slot_consistency`) with store/taxonomy identity
- [ ] T009 [US1] Test `family_slot_consistency.py` in `tests/spec116/test_family_slot_consistency.py`: a fixture where one family always lands in one slot yields `slot_keyed`; a fixture where families spread evenly yields `family_keyed`; artifact validates against the schema
- [ ] T010 [US1] Implement thin CLI `scripts/spec116_family_slot_consistency.py`: argparse wrapper, dry-run/print by default, `--write` to emit the JSON report; print the recommendation

**Checkpoint**: US1 delivers a go/no-go vocabulary decision with no model trained (SC-001).

---

## Phase 4: User Story 2 — shape→coverage coupling (Priority: P1)

**Goal**: Determine whether layer masks derive from terrain shape (elevation + slope) or are authored independently; report explained variance per tile/layer and a bimodality finding. No model trained.

**Independent Test**: Run the CLI over the real v50 store and read per-(tile,layer) explained variance, the dip-test p-value, mixture BIC, and the high-coupling tile share (spec US2 acceptance 1–3).

### Implementation for User Story 2

- [ ] T011 [US2] Implement `shape_coverage_coupling.py` in `src/harvester/spec116/shape_coverage_coupling.py`: downsample `height_257` to 16×16 chunk elevation + slope; per (tile, detail layer) fit `GradientBoostingRegressor` `{elevation, slope} → coverage`; explained variance `1 - SS_res/SS_tot`; Hartigan dip test + two-vs-one component Gaussian-mixture BIC on the explained-variance distribution; high-coupling tile share; linear-underpowered note; build the `v116-analysis-report-v1` artifact (report_kind=`shape_coverage_coupling`, decision=`coverage_derivable`/`coverage_independent`)
- [ ] T012 [US2] Test `shape_coverage_coupling.py` in `tests/spec116/test_shape_coverage_coupling.py`: a synthetic strong-coupling fixture yields high explained variance + bimodality; a random-coverage fixture yields low explained variance; artifact validates against the schema
- [ ] T013 [US2] Implement thin CLI `scripts/spec116_shape_coverage_coupling.py`: dry-run/print by default, `--write` to emit the JSON report

**Checkpoint**: US2 delivers the derivability decision (SC-002), superseding the inconclusive linear analysis.

---

## Phase 5: User Story 4 — trustworthy evaluation (Priority: P2)

**Goal**: Build a spatially-isolated held-out set (zero 8-neighbour adjacency) and relief-stratified evaluation with the trivial baseline, so every later result is trustworthy. Delivers value immediately by re-scoring an existing model.

**Independent Test**: Build the split, verify `verified_violation_count == 0`, and re-score an existing geometry model stratified by relief (spec US4 acceptance 1–4).

### Implementation for User Story 4

- [ ] T014 [US4] Implement `held_out_split.py` in `src/harvester/spec116/held_out_split.py`: deterministic 8-neighbour-isolated split over tile coords from `index.parquet`; grow held-out regions with a ≥1-ring training buffer; compute and verify `verified_violation_count` (must be 0); write `split.parquet` + `split.json` (`v50-held-out-split-v1`) with store hash, build id, taxonomy revision, `absolute_comparison_to_prior_invalid=true`, and `baseline_requiring_rerun`
- [ ] T015 [US4] Test `held_out_split.py` in `tests/spec116/test_held_out_split.py`: a grid fixture produces zero 8-neighbour violations; a too-small corpus exits non-zero rather than emitting a leaky split; artifact validates against the schema
- [ ] T016 [US4] Implement thin CLI `scripts/spec116_build_held_out_split.py`: dry-run/print counts + violation count by default, `--write` to emit the split
- [ ] T017 [US4] Implement `relief_stratification.py` in `src/harvester/spec116/relief_stratification.py`: partition locations into flat vs relief-bearing by chunk height std above a reported constant threshold; compute per-stratum error + the trivial (tile-mean) baseline; reused-piece train/held-out overlap via `harvester.v50.minimap_alignment` restricted to cross-set pairs (FR-013/SC-008)
- [ ] T018 [US4] Test `relief_stratification.py` in `tests/spec116/test_relief_stratification.py`: a flat fixture routes to the flat stratum; a relief fixture routes to relief-bearing; trivial baseline is reported per stratum; overlap counter is zero for disjoint fixtures
- [ ] T019 [US4] Wire relief-stratified re-score into `scripts/spec116_train_structure.py` `--rescore-checkpoint`/`--print-only` path: load an existing geometry checkpoint, evaluate on the held-out split, report flat vs relief error + trivial baseline (no training)

**Checkpoint**: US4 makes evaluation trustworthy (SC-005, SC-006, SC-008); an existing model can be re-scored immediately.

---

## Phase 6: User Story 3 — predict layer structure from minimap alone (Priority: P2)

**Goal**: From a raw minimap image, predict terrain layer structure as rows (family per detail slot), respecting schema constraints; gate on per-class IoU/recall, never aggregate accuracy; legality check/repair; OOD audit.

**Independent Test**: Train (user-run) on the Phase 5 split, score per-class recall/IoU; run inference on an OOD hand-painted image and confirm a legal, non-degenerate, auditable result (spec US3 acceptance 1–4).

### Implementation for User Story 3

- [ ] T020 [US3] Implement `structure_model.py` in `src/harvester/spec116/structure_model.py`: `StructureSlotNet` — one independent U-Net-lite classifier (mirrors `TerrainFeatureNet` capacity class) predicting one detail slot's per-chunk family over 5 classes; `build_structure_model(slot=...)` + schema identity block; refuse multi-slot/multi-head construction (constitution IV)
- [ ] T021 [US3] Test `structure_model.py` in `tests/spec116/test_structure_model.py`: output shape is `(B, 5, 16, 16)`; param count in the small-model class; constructing a multi-head variant raises
- [ ] T022 [US3] Implement `structure_train.py` in `src/harvester/spec116/structure_train.py`: dry-run-first (print plan + time/mem estimate, exit without `--confirm-run`); class-weighted CE (capped inverse-frequency, default max 15.0); majority-class baseline; per-class IoU/recall gate (D-08); `promotion_verdict=pending`; write `checkpoint_best.pt` + `v50-structure-run-v1` record; refuse if `held_out_split.verified_violation_count != 0` or vocabulary decision missing
- [ ] T023 [US3] Test `structure_train.py` in `tests/spec116/test_structure_train.py`: dry run writes nothing and prints a plan; a tiny CPU fit on the fixture produces a valid run record; aggregate accuracy is recorded but the gate field never references it
- [ ] T024 [US3] Implement thin CLI `scripts/spec116_train_structure.py`: argparse wrapper binding `--store/--split/--dumps/--slot/--vocabulary/--epochs/.../--confirm-run/--device`; the `--rescore-checkpoint` path reuses T019
- [ ] T025 [US3] Implement `structure_infer.py` in `src/harvester/spec116/structure_infer.py`: predict family probabilities per chunk/slot; legality resolver picks a legal same-family local id when a tile MTEX table is supplied (SC-004 = 100% legal); reject + mark low-confidence when none; OOD (no table) sets `legal_table_available=false` and never fabricates a reference (D-05); emit `v50-structure-infer-v1` audit record
- [ ] T026 [US3] Test `structure_infer.py` in `tests/spec116/test_structure_infer.py`: with a table, all emitted references are legal; with no table, no reference is fabricated and the audit record flags it; a low-confidence chunk is reported
- [ ] T027 [US3] Implement thin CLI `scripts/spec116_infer_structure.py`: `--checkpoint/--inputs/--tile-table/--output`; produce the audit JSON + review sheet

**Checkpoint**: US3 delivers structure prediction (SC-003, SC-004, SC-009); the user runs the CUDA training.

---

## Phase 7: User Story 5 — feed predicted structure into geometry (Priority: P3)

**Goal**: Supply predicted (never ground-truth) structure to terrain height reconstruction and evaluate whether it reduces relief-region error.

**Independent Test**: Train height reconstruction with and without predicted structure on the same held-out set and compare relief-region error (spec US5 acceptance 1–2).

### Implementation for User Story 5

- [ ] T028 [US5] Implement `structure_materialize.py` in `src/harvester/spec116/structure_materialize.py`: run a frozen US3 checkpoint over selected curriculum rows → derived structure store (`structure_family`, `structure_confidence`, `structure_legal`, row-aligned `index.parquet`); bind to checkpoint sha256; source stores immutable (mirror `harvester.v50.feature_map_materialize`)
- [ ] T029 [US5] Test `structure_materialize.py` in `tests/spec116/test_structure_materialize.py`: derived store is row-aligned; source store untouched; checkpoint hash recorded; taxonomy mismatch refused
- [ ] T030 [US5] Implement thin CLI `scripts/spec116_materialize_structure.py`: dry-run/print by default, `--write` to emit the derived store
- [ ] T031 [US5] Document the paired geometry comparison in `quickstart.md` section 5b: run the existing Spec 114 geometry trainer with and without `--feature-store <derived store>` on the same split; record `v50-structure-geometry-comparison-v1` (relief MAE both ways + trivial baseline + `sc007_beats_trivial_on_relief` + `absolute_comparison_to_prior_runs_invalid=true`). Hand off the exact CLI; do not launch training.

**Checkpoint**: US5 reports whether predicted structure helps relief-region height error (SC-007) or an honest negative finding.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Hygiene across all stories.

- [ ] T032 [P] Run `uv run ruff check src/harvester/spec116 scripts/spec116_*.py` and fix; run `uv run python -m py_compile src/harvester/spec116/*.py`
- [ ] T033 [P] Run full data-harvester suite `uv run python -m pytest -q` and confirm no regressions beyond known pre-existing failures
- [ ] T034 Update `wow-viewer/memory-bank/{activeContext,progress}.md` with the implemented state (RULE 11); compress aggressively
- [ ] T035 Run `quickstart.md` validation paths end-to-end (dry runs only — no training launched by the assistant)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup; BLOCKS all user stories.
- **US1 (Phase 3)**: Depends on Foundational. No dependency on other stories. **MVP.**
- **US2 (Phase 4)**: Depends on Foundational. Independent of US1.
- **US4 (Phase 5)**: Depends on Foundational. Independent of US1/US2; delivers value immediately.
- **US3 (Phase 6)**: Depends on Foundational + **US1** (vocabulary decision) + **US4** (held-out split). The constitution's one-phase-at-a-time rule means US3 starts only after US1/US2/US4 are validated.
- **US5 (Phase 7)**: Depends on **US3** (a promoted structure checkpoint) + **US4** (held-out split).
- **Polish (Phase 8)**: After the desired stories are complete.

### User Story Dependencies

- **US1 (P1)**: Foundational only. Delivers the vocabulary decision US3 consumes.
- **US2 (P1)**: Foundational only. Delivers the derivability decision (whether US3 needs coverage regressors).
- **US4 (P2)**: Foundational only. Delivers the trustworthy held-out set + stratification US3/US5 require.
- **US3 (P2)**: Foundational + US1 + US4. Decomposed into one independent model per detail slot (D-04).
- **US5 (P3)**: US3 + US4. Consumes predicted (never ground-truth) structure.

### Within Each User Story

- Library module before its CLI wrapper.
- Tests alongside each slice (project convention).
- Story validated against the real v50 store before the next priority begins (one phase at a time).

### Parallel Opportunities

- T002/T003 (Setup) run in parallel.
- T005 (contract tests) and T006/T007 (relational extract) run in parallel within Foundational.
- US1 (Phase 3) and US2 (Phase 4) are fully independent and can proceed in parallel after Foundational.
- Within US3, the model (T020) and the legality resolver (T025) touch different files and can be drafted in parallel once T022's contract is fixed.

---

## Implementation Strategy (MVP first)

1. **MVP = US1**: a single analysis script that delivers the vocabulary decision with no model trained. Ship and validate this first.
2. Then US2 (derivability) and US4 (trustworthy eval) — both analysis/infra, no GPU.
3. Then US3 (the core model), decomposed per slot, user-run training.
4. Then US5 (the payoff), user-run paired geometry comparison.

Every training and heavy step is user-run from `quickstart.md`; the assistant prepares scripts, states time/memory, and hands off the CLI.