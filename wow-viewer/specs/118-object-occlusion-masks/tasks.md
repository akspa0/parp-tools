---
description: "Task list for Spec 118 per-object occlusion-aware masks for object-deconfounded terrain height"
---

# Tasks: Per-Object Occlusion-Aware Masks for Object-Deconfounded Terrain Height

**Input**: Design documents from `/specs/118-object-occlusion-masks/`

**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included — project convention is focused tests per slice, run with `uv run python -m pytest tests/spec118/ -q` (Python) and `dotnet test` with a name filter (C#).

**Organization**: Tasks grouped by user story in spec priority order (US1 P1 signal, US2 P2 loss proof, US3 P3 segmenter). Each story is independently testable per its own Independent Test in spec.md. The user runs every training/heavy step (FR-012); tasks that produce user-run CLIs hand off the invocation, they do not launch it.

**Implementation-time finding carried into these tasks**: the C# harvester already computes the visibility-correct object mask (`TerrainVisibleObjectMaskRasterizer` + `AdtTensorPackBuilder.BuildStrictTerrainVisibleObjectMask`: transformed M2/WMO triangles retained only above the raw MCVT surface +0.25 clearance, liquid-aware, front-most overlap rule) and already streams `object_geometry_visible_mask_257` / `object_geometry_visible_source_257` in the Full and V16 profiles (NOT V22). The ONLY new C# code is one dense per-tile instance-id array (research D-03). US1 is otherwise catalog wiring, mirroring Spec 117 US1 exactly.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- All paths are relative to `wow-viewer/` unless stated otherwise (Python paths relative to `wow-viewer/data-harvester/`)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the `harvester.spec118` package skeleton.

- [x] T001 Create package `data-harvester/src/harvester/spec118/__init__.py` (empty namespace, mirrors `harvester.spec117` layout)
- [x] T002 [P] Create test package `data-harvester/tests/spec118/__init__.py`

**Checkpoint**: Package importable.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Widen the reused `v50-model-stage-run-v1` schema and build the spec-scoped contract helpers every later story needs. MUST complete before any user story.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [x] T003 Widen `STAGES` in `data-harvester/src/harvester/v50/model_stage_contract.py` to add `"object_segmentation"` (data-model.md Run-Record Schema reuses `v50-model-stage-run-v1` verbatim per research.md D-06; the stage enum is the one schema change, same shape as Spec 117's `"lattice_prior"` addition)
- [x] T004 [P] Test the widened enum in `data-harvester/tests/spec118/test_model_stage_contract_object_segmentation.py`: a well-formed `stage="object_segmentation"` document validates via `harvester.v50.model_stage_contract.validate_model_stage_run`; an unlisted stage still rejects
- [x] T005 Implement `data-harvester/src/harvester/spec118/object_contract.py`: constants (`STAGE="object_segmentation"`, `OUTPUT_SIGNAL="object_class_3"`, `CLASS_NAMES=("none","doodad","building")`, `CLASS_COUNT=3`, `BRIDGE_CLASS_COUNT=2`), `architecture_identity(model)` (id/config_sha256 via `sha256_json`/parameter_count, mirrors `spec117.lattice_contract`), `build_object_stage_run(...)` assembling + self-validating a `v50-model-stage-run-v1` document, and re-exported `sha256_file`/`identity_for_path` from `harvester.v50.model_stage_contract` (single canonical owner, no duplication)
- [x] T006 [P] Test `data-harvester/tests/spec118/test_object_contract.py`: a well-formed run summary built by `build_object_stage_run` validates; a bad `stage`/`output_signal` is rejected with a clear error

**Checkpoint**: Foundation ready — schema and identity helpers work; user story implementation can begin.

---

## Phase 3: User Story 1 — occlusion-aware per-object mask + class signal in the dataset (Priority: P1) 🎯 MVP

**Goal**: The visible-only object mask, per-pixel class, and per-object instance id become first-class, readable v50 store signals for every eligible tile. No model.

**Independent Test**: Audit the rebuilt store: marked pixels coincide with visible objects (never full footprint), underground objects contribute ≈0 pixels, each object has a distinct instance id and a class, no-object tiles are exactly empty (spec US1 acceptance 1–4).

### Implementation for User Story 1

- [x] T007 [US1] Extend `src/core/WowViewer.Core.IO/Maps/TerrainVisibleObjectMaskRasterizer.cs`: add an optional visible-instance paint to `PaintTriangleWithTrace` (new optional parameters `int[,]? visibleInstance` + `int instanceId`); inside the existing front-most block (the `if (visibleMask[y, x] <= 0f || objectElevation > visibleTopElevation[y, x])` guard that already updates top-elevation/source) also write `visibleInstance[y, x] = instanceId`. Keep `PaintTriangle` (non-trace overload) behavior unchanged; no changes to visibility, clearance, or liquid semantics
- [x] T008 [US1] Extend `src/core/WowViewer.Core.IO/Maps/AdtTensorPackBuilder.cs`: in `BuildStrictTerrainVisibleObjectMask`, allocate `int[,] visibleInstance`, assign per-tile compact instance ids (1..K, deterministic iteration order over resolved placements, M2 then WMO loops), pass id + array into each `RecordTriangle` call, and return the array + an instance table (`instance_id`, `placement_unique_id`, `asset_index`, `source`, `visible_pixel_count` — count computed after painting) on `StrictObjectMaskBuildResult`; wire the array onto `TerrainTileTensorPack` as `ObjectGeometryVisibleInstance257` and append `object_geometry_visible_instances` to the per-tile metadata writer next to `object_geometry_target_assets`. Null-array eligibility paths stay null (excluded-and-counted semantics unchanged)
- [x] T009 [US1] Stream the new array: `WriteArray(outputStream, "object_geometry_visible_instance_257", pack.ObjectGeometryVisibleInstance257)` in `RawArraySerializer.WriteV16Arrays` + `WriteFullArrays` (NOT `WriteV22Arrays` — matches the existing strict-array omission there) and the NPZ equivalent (`"<i4"`) in `NpzTileSerializer.cs`
- [x] T010 [US1] [P] C# tests in `tests/WowViewer.Core.Tests/TerrainVisibleObjectMaskRasterizerTests.cs` (+ a serializer test): two overlapping visible triangles from different placements resolve the instance id to the front-most (highest-elevation) fragment; an occluded triangle paints no instance id; instance id 0 remains exactly where the mask is 0; the V16 round-trip carries `object_geometry_visible_instance_257` byte-identically
- [x] T011 [US1] Add three rows to the frozen catalog table in `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md` (`object_geometry_visible_mask_257` float32 (257,257); `object_geometry_visible_source_257` uint8 (257,257); `object_geometry_visible_instance_257` int32 (257,257); all copy-if-verified, no has-flag) with a note that the strict arrays are Full/V16-only, then regenerate both configs (no hand-editing): `uv run python scripts/v50_generate_manifest_template.py --catalog-doc docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md --build-id 0_5_3_3368 --release v50.1 --output v50_configs/v50-manifest-template-0_5_3_3368.json --signals-output v50_configs/v50-signals-0_5_3_3368.json` from `data-harvester/`
- [x] T012 [US1] [P] Test `data-harvester/tests/spec118/test_object_mask_signal.py`: `harvester.v50.signal_catalog.parse_catalog_table` over the real doc yields all three new entries with documented dtype/shape/policy; the existing drift guard `tests/v50/test_manifest_template_matches_catalog.py::test_committed_053_template_matches_the_frozen_catalog` still passes unmodified (run it, do not edit it)
- [x] T013 [US1] Implement `data-harvester/src/harvester/spec118/object_mask_audit.py` + thin CLI `data-harvester/scripts/spec118_audit_object_masks.py`: read-only audit emitting `v118-object-mask-audit-v1` (data-model.md): marked-fraction p05/p50/p95 per map + corpus, exclusion counts by reason, per-instance visible-pixel distribution, class-per-instance consistency violations (mode-of-source per instance; tolerance 0 beyond documented mixed pixels), visible-vs-`object_mask_257` reduction factor where the footprint mask is present, instance-ids-only-where-mask consistency check. Dry-run prints JSON; `--write` persists. Fails closed with a clear message when the store lacks `object_geometry_visible_mask_257`. Test `data-harvester/tests/spec118/test_object_mask_audit.py` on a fixture store (hand-computed fractions, violation detection, missing-array refusal)

**Checkpoint**: US1 delivers the signal end-to-end through the existing generic store pipeline (SC-001/SC-002 measurable by the audit). No model trained. A real rebuild against `H:\CLIENTS` is the user's next heavy step (not run here).

---

## Phase 4: User Story 2 — object-masked terrain-height loss (Priority: P2)

**Goal**: The existing geometry trainers can down-weight visible-object pixels in the loss, and the paired with/without comparison is fully wired for the user to run.

**Independent Test**: Dry-run both trainers with `--object-mask-weight 1.0` against a fixture store and confirm the plan records the flag + object-touched subset; user-run paired training then reads relief-stratified MAE on object-touched held-out tiles (spec US2 acceptance 1–3).

### Implementation for User Story 2

- [x] T014 [US2] Implement `data-harvester/src/harvester/spec118/object_loss.py`: `load_object_mask(store_group, rows) -> (mask_or_none, missing_count)` (reads `object_geometry_visible_mask_257`, crops 257→256 with the same convention as the liquid arrays, returns None-with-warning when absent — mirrors the `--liquid-mask-weight` missing-signal behavior), `object_point_weight(mask, w) -> ndarray` (`1 - w * mask`, w clamped to [0,1], `w=0` returns exact ones for bit-parity), `object_touched_rows(mask) -> bool array` (≥1 visible pixel), and `subset_metrics(per_row_mae, touched)` (object-touched vs untouched MAE summary for the run record, FR-008)
- [x] T015 [US2] [P] Test `data-harvester/tests/spec118/test_object_loss.py`: weight math hand-checked (`w=0` parity, `w=1` zeroes object pixels, partial w), crop convention matches the liquid arrays, missing-array returns None + count, touched-row selection on a fixture
- [x] T016 [US2] Wire `--object-mask-weight` (float, default 0.0) into `data-harvester/src/harvester/v50/direct_geometry_train.py` following the exact `--liquid-mask-weight` pattern: parse, plan echo, missing-signal warning + no-op, per-point weight multiply in the loss, and object-touched subset metrics in the run record alongside aggregate + relief-stratified metrics. Separate commit from T017 (Rule 6)
- [x] T017 [US2] Wire the identical flag into `data-harvester/src/harvester/v50/geometry_detailer_train.py` (same pattern, including the coarse-only baseline remaining unmasked so the relative gate stays honest). Separate commit
- [x] T018 [US2] [P] Test `data-harvester/tests/spec118/test_object_masked_trainers.py` (CPU, fixture store): both trainers' dry-run plans include `object_mask_weight`; a missing-array store yields the warning + weight-1.0 no-op; the loss on a 2-step CPU run with `w=1.0` is exactly 0 on a fixture whose target is only wrong at object pixels

**Checkpoint**: US2's paired comparison is one documented command away (quickstart.md §2); the runs themselves are user-run.

---

## Phase 5: User Story 3 — from-scratch object segmentation + classifier (Priority: P3)

**Goal**: A small from-scratch model predicts per-pixel visible-object class from any minimap tile; its output bridges into the existing `--feature-store` contract with zero trainer changes.

**Independent Test**: Train on US1 masks; score per-class IoU/recall on the held-out split; run on a loose hand-painted tile with no store; dry-run both geometry trainers against the bridged feature store (spec US3 acceptance 1–3).

### Implementation for User Story 3

- [x] T019 [US3] Implement `data-harvester/src/harvester/spec118/object_segment_model.py`: `derive_class_target(source_257) -> (256,256) int64` (crop + map {0,1,2} → class ids, raise on unexpected values), `ObjectSegmentNet(base=24)` — U-Net-lite (Spec 117 v2 pattern: 4-level encoder, skip decoder) RGB 256×256×3 → 3-class logits at 256×256, from scratch, record real parameter count in `architecture_identity`. No pretrained backbone (FR-010)
- [x] T020 [US3] [P] Test `data-harvester/tests/spec118/test_object_segment_model.py`: target derivation on a fixture (incl. crop alignment vs the mask), forward shape `(B,3,256,256)`, gradient flows through skips, parameter count recorded, constructable from `base` alone (bridge reconstruction, the Spec 117 bug class)
- [x] T021 [US3] Implement `data-harvester/src/harvester/spec118/object_segment_train.py` + thin CLI `data-harvester/scripts/spec118_train_objects.py`: dry-run-first; `--held-out-split` REQUIRED with no fallback (Spec 117 contract); refuses closed when the store lacks `object_geometry_visible_source_257`; class-weighted masked CE (class weights from held-out class frequencies, `none` included but capped so background doesn't swamp); per-class IoU/recall + median visible-object IoU on object-touched tiles each epoch; gate constants from research D-07 (IoU ≥ 0.40 median, recall ≥ 0.50); writes `training_plan.json`, `run_identity.json`, `checkpoint_best.pt` (with `object_config: {"base": ...}` for base-only reconstruction), `model_stage_run.json` via `object_contract.build_object_stage_run` (`promotion_verdict="pending"`, `upstream_models=[]`); reuses `direct_geometry_train.apply_held_out_split` + the `height_relative_train` curriculum/source validation imports
- [x] T022 [US3] [P] Test `data-harvester/tests/spec118/test_object_segment_train.py` (CPU fixture store): dry-run plan prints valid JSON and exits 0 without `--confirm-run`; refuses omitted split; refuses leaky split (`verified_violation_count != 0`); refuses a store missing the source array; a 2-epoch CPU run on a tiny fixture writes all four artifacts and the run record validates
- [x] T023 [US3] Implement `data-harvester/src/harvester/spec118/object_segment_infer.py` + thin CLI `data-harvester/scripts/spec118_infer_objects.py`: two mutually exclusive modes (Spec 116 pattern): `--inputs` loose PNG files/dirs (no store, no ground truth — runs unchanged on a hand-painted OOD tile, FR-009) and `--store`/`--dumps` batch; both emit per-tile class PNGs + a `v118-object-infer-v1` audit record (data-model.md); store mode adds per-class IoU/recall where ground truth exists; OOD mode records `ground_truth: "unavailable"`, never fabricates reference data
- [x] T024 [US3] [P] Test `data-harvester/tests/spec118/test_object_segment_infer.py`: loose-image mode runs with no store present and writes class PNG + audit record; store mode scores against a fixture with hand-computed IoU; modes are mutually exclusive (argparse error); checkpoint base-mismatch load refuses with a clear error
- [x] T025 [US3] Implement `data-harvester/src/harvester/spec118/object_feature_bridge.py` + thin CLI `data-harvester/scripts/spec118_objects_to_feature_map.py`: `objects_to_feature_map(store, checkpoint, output, write=False)` runs the frozen segmenter over every row's `minimap_rgb`, writes softmax channels 1..2 (doodad, building — `none` dropped as redundant) as `(N,2,256,256)` float16 `feature_map` under `schema="v115-feature-map-v1"`, `class_count=2`, `attrs.source_signal="object_geometry_visible"`, checkpoint path+sha256 bound (mirrors `spec117/lattice_bridge.py` exactly, including `object_config.base` reconstruction); source store never mutated; non-empty output refused
- [x] T026 [US3] [P] Test `data-harvester/tests/spec118/test_object_feature_bridge.py`: dry-run returns a plan only; `write=True` produces exact schema/shape/attrs + row-aligned `index.parquet`; checkpoint sha256 matches; source store bytes unchanged; channels are a valid probability simplex remainder (doodad+building ≤ 1+eps per pixel); non-empty output refused
- [x] T027 [US3] Regression-prove the handoff needs no trainer changes (Spec 117 T018 pattern): dry-run `v50_train_direct_geometry.py --feature-store <T025 fixture output>` and `v50_train_geometry_detailer.py --feature-store <same>` and confirm both accept the shape/schema with zero code edits (`input_channels: 5`, `deployment_inputs` gains the generated feature map)

**Checkpoint**: US3(i) segmenter + bridge real and tested. US3's real training, OOD eyeball, and paired geometry comparison are user-run per quickstart.md §3.

---

## Phase 5b: US3 augmentation — object prior ALONGSIDE the terrain-feature deconfounding

**Goal**: The bridge (T025) produces a `v115-feature-map-v1` store, but the geometry trainers accepted only ONE `--feature-store`, and the promoted deconfounded chain (Spec 115 `v3`) already occupies it with the terrain-feature map. So the object prior could only REPLACE that map — never augment it — even though objects occlude ground height (a different confound than roads-as-slopes). This phase makes `--feature-store` repeatable so the object prior sits alongside the terrain-feature prior.

**Independent Test**: load two feature stores (terrain-feature class_count 4 + object class_count 2), confirm `in_channels = 3 + 4 + 2 = 9`, channels concatenate in CLI order, coverage is validated per store, and all three CLIs advertise the repeatable flag.

- [x] T031 Extract shared `data-harvester/src/harvester/v50/feature_stores.py` (`FeatureBinding`, `load_feature_stores` — validates schema/class_count/feature_map/full-row-coverage per store exactly as the old single-store path did; `total_class_count`; `feature_channels_for_row` — concatenates every prior's channels in CLI order, returns `None` if any store misses the row so the eval helpers' soft skip is preserved; `plan_entries`; `road_feature_binding` — the FR-008 road diagnostic argmaxes the Spec 115 terrain-feature prior's OWN channels, identified by its `taxonomy_revision` attr; `as_bindings` backward-compat shim)
- [x] T032 Make `--feature-store` repeatable (`action="append"`) on `direct_geometry_train.py`, `geometry_detailer_train.py`, and `direct_geometry_materialize.py`; `in_channels = 3 + sum(class_counts)`; the dry-run plan records `feature_stores` (a list) + `feature_input_channels`; the materializer accepts the priors in the SAME order the checkpoint trained with. `height_relative_evaluate.py`'s preview/evaluate helpers gain a `feature_bindings` param (legacy `feature_group` kept working via `as_bindings`, so `height_relative_train.py` is untouched)
- [x] T033 [P] Test `data-harvester/tests/v50/test_feature_stores.py`: two-store concatenation preserves CLI order; single-store returns its own channels; empty/`None` is `[]`/`None`; a store missing any selected row is refused at load; wrong schema refused; `road_feature_binding` finds the taxonomy-carrying store regardless of order (and is `None` for object-only lists); `plan_entries` records each store; all three CLIs advertise `REPEATABLE` via `--help`

**Checkpoint**: The object segmenter's output is a real ADDITIONAL deconfounding input, not a replacement. The paired terrain-only vs. terrain+objects geometry comparison is user-run per quickstart.md §3b.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Reconcile docs with what was actually built, and prove no regressions.

- [x] T028 [P] Verify `specs/118-object-occlusion-masks/contracts/cli-contract.md` and `quickstart.md` against the REAL argparse of every new/changed CLI (this project's "verify docs against real argparse" convention — Spec 116 lesson: passing focused tests only proves library functions, not that documented commands parse)
- [x] T029 Run full validation from `data-harvester/`: `uv run python -m pytest tests/spec118/ tests/v50/ -q`, `uv run ruff check src/harvester/spec118 src/harvester/v50 scripts/spec118_*.py`, `uv run python -m py_compile src/harvester/spec118/*.py scripts/spec118_*.py`, full Python suite for regressions, plus `dotnet build WowViewer.slnx -c Debug` and the focused C# object-mask/serializer tests from `wow-viewer/`
- [x] T030 Update `memory-bank/activeContext.md` and `memory-bank/progress.md` with the Spec 118 implementation summary, including the "strict visible mask already streamed; only the instance array was new C#" finding

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories (US2's run-record additions and US3's stage record both need the widened schema + `object_contract.py`).
- **US1 (Phase 3)**: Depends on Foundational only. C# (T007–T010) → catalog/regen (T011–T012) → audit (T013).
- **US2 (Phase 4)**: Depends on Foundational. Code/testable against fixture stores without a real US1 rebuild; a REAL paired run requires the US1 rebuild.
- **US3 (Phase 5)**: Depends on Foundational; uses US1's array names but is code/testable against fixtures without a real rebuild. Bridge (T025) depends on the trainer checkpoint format (T021).
- **Polish (Phase 6)**: Depends on US1–US3 all being implemented.

### Parallel Opportunities

- T002, T004, T006 parallel with sibling implementation tasks once they land.
- T010 (C# tests) parallel with T012 (catalog test) — different languages/files.
- T013 (audit) parallel with T014–T018 (US2) — different files.
- T020, T022, T024, T026 (US3 tests) parallel with each other once their implementations land.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 + Phase 2 (Setup + Foundational).
2. Complete Phase 3 (US1): instance array + catalog wiring + audit — the signal exists end-to-end through the generic pipeline.
3. **STOP and VALIDATE**: run T010/T012/T013 tests; drift guard passes; audit runs on a fixture.

### Incremental Delivery

1. Setup + Foundational → schema/contract ready.
2. US1 → signal plumbed (data only, no model) → validate independently.
3. US2 → loss flag on both trainers → dry-run validated → handed to user for the paired proof (the cheap gate: if ground-truth-mask exclusion does not help, report null and stop before US3 training).
4. US3 → segmenter + infer + bridge implemented + tested → real training/OOD/geometry comparison handed to user as documented CLI invocations.
5. Polish → docs verified against real argparse, full suites green, memory updated.
