# Tasks: V50-Native Height-First Terrain Model with Dataset Corrections

**Input**: Design documents from `specs/112-v50-height-model/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Fixture-first tests are required for every trust/contract boundary (repo convention from
Specs 109/111). Heavy real-data rebuild and all training runs are **user-executed only** —
tooling prepares and prints the exact command; the assistant never launches them (FR-009/SC-006).

**Organization**: Tasks follow the three user stories. One phase must be validated before the next
starts (constitution: One Phase at a Time; each phase ≤10 tasks per Bite-Sized Plans).

## Phase 1: Setup

**Purpose**: No new project scaffolding is needed — this feature extends existing packages. One
verification task guards against starting from a broken baseline.

- [ ] T001 Verify baseline: `dotnet build wow-viewer/WowViewer.slnx -c Debug` succeeds and
  `uv run python -m pytest tests/v50/ -q` passes from `wow-viewer/data-harvester/` before any change

**Checkpoint**: Clean baseline confirmed; failures found here are pre-existing and out of scope.

---

## Phase 2: Foundational — catalog parser and reason vocabulary

**Purpose**: Shared primitives both US1 tools depend on.

- [ ] T002 [P] Implement `parse_catalog_table()` reading the frozen signal table (fixed column
  order `Signal | dtype | Shape | V50 Policy | Required | Notes`) from
  `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`, with the
  explicit `era_available` allow-list (only `mccv_rgb` today → WotLK+), in
  `wow-viewer/data-harvester/src/harvester/v50/signal_catalog.py`
- [ ] T003 [P] Add the `UnavailableSignal.reason` prefix vocabulary
  (`era_unavailable:`, `no_source_data:`, `not_yet_extracted:`) as constants plus a
  `classify_reason()` helper (additive; existing free-text reasons stay valid) in
  `wow-viewer/data-harvester/src/harvester/v50/contracts.py`
- [ ] T004 Add fixture tests for both: table parse round-trip against a snippet of the real doc
  table, era allow-list resolution, reason classification, in
  `wow-viewer/data-harvester/tests/v50/test_signal_catalog.py`

**Checkpoint**: `uv run python -m pytest tests/v50/test_signal_catalog.py -q` green.

---

## Phase 3: User Story 1 — Corrected, Honest v50.1 Corpus (Priority: P1) — MVP

**Goal**: Rebuilt Kalimdor/Azeroth stores where every declared signal is populated or explicitly
unavailable-with-reason; 1024px minimap coverage equals 256px.

**Independent Test**: Coverage audit reports (schema
`contracts/coverage-audit-report.schema.json`) show zero `zero_coverage_unexplained` signals and
`minimap_resolution_parity.parity == true` on both rebuilt stores.

- [ ] T005 [US1] Fix `AlphaTensorPackBuilder` to assign the MCNK flags it already parses onto the
  output pack's `McnkFlags16` (research.md Decision 1; shape/orientation matching the LK path's
  `ReadMcnkFlags` convention) in `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTensorPackBuilder.cs`
- [ ] T006 [P] [US1] Add a focused C# regression test proving an Alpha-format tile round-trips
  non-zero `mcnk_flags_16` through the tensor pack and `RawArraySerializer`, in
  `wow-viewer/tests/WowViewer.Core.Tests/` (new `AlphaMcnkFlagsTests.cs`)
- [ ] T007 [US1] Empirically confirm the `minimap_rgb_1024` loss mechanism (research.md Decision 2):
  reproduce under `Parallel.ForEach` vs a sequential run on a bounded tile set, then fix the
  confirmed cause (expected: synchronize/concurrent-ify `NativeMpqService`'s mutable scan-cache
  fields) in `wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs`, with a concurrent-read
  regression test in `wow-viewer/tests/WowViewer.Core.Tests/`
- [ ] T008 [US1] Implement the manifest-template generator (catalog in → template JSON out;
  refuses to emit catalog-dropped or era-unavailable signals; the only writer of the template) as
  `wow-viewer/data-harvester/src/harvester/v50/manifest_template.py` + thin
  `wow-viewer/data-harvester/scripts/v50_generate_manifest_template.py`
- [ ] T009 [P] [US1] Add template-generator tests: catalog-dropped signals absent, `mccv_rgb`
  absent for 0.5.3 with an `era_unavailable:` record, output validates against the existing
  `v50-provenance` signal shape, regeneration is deterministic, in
  `wow-viewer/data-harvester/tests/v50/test_manifest_template_matches_catalog.py`
- [ ] T010 [US1] Implement the per-signal coverage auditor (full-store scan, not sampled; emits
  `contracts/coverage-audit-report.schema.json`-conformant JSON incl. the 256/1024 parity block) as
  `wow-viewer/data-harvester/src/harvester/v50/coverage_audit.py` + thin
  `wow-viewer/data-harvester/scripts/v50_audit_signal_coverage.py`, with fixture tests in
  `wow-viewer/data-harvester/tests/v50/test_coverage_audit.py`
- [ ] T011 [US1] Regenerate `wow-viewer/data-harvester/v50_configs/v50-manifest-template-0_5_3_3368.json`
  via the T008 generator and update `v50-signals-0_5_3_3368.json` to match the catalog (drop the
  four dead signals; `mccv_rgb` era-unavailable)
- [ ] T012 [US1] **USER RUNS**: rebuild Kalimdor and Azeroth against `H:\CLIENTS` with the
  regenerated configs and re-run finalize (exact commands + duration estimates already in
  `specs/112-v50-height-model/quickstart.md` §1.3); prior stores stay intact until the rebuilt
  ones pass finalize/verify (FR-005 staging discipline)
- [ ] T013 [US1] Run the coverage audit on both rebuilt stores, record both reports under
  `wow-viewer/output/reports/v50/v50.1/`, and document SC-001/SC-002 proof (exact counts, hashes)
  in `specs/112-v50-height-model/quickstart.md` §1.4

**Checkpoint**: SC-001 and SC-002 proven on real rebuilt stores. No US2 work before this.

---

## Phase 4: User Story 2 — Full-Catalog Curriculum, Big Maps Only (Priority: P2)

**Goal**: A trainer-facing curriculum carrying every populated catalog signal, Kalimdor+Azeroth
only, deterministic within-map split.

**Independent Test**: Curriculum summary lists both maps (and only those), signal list equals the
rebuilt stores' populated set, two rebuilds produce identical splits, and a build request naming
PVPZone02/Kalidar is refused with an explicit error.

- [ ] T014 [US2] Add the map allow-list (`allowed_maps={"Kalimdor","Azeroth"}` for this lane;
  out-of-list source rows raise `CurriculumBuildError`) and replace the hardcoded 7-field
  `CURRICULUM_FIELDS` with per-build derivation from the source stores' populated manifests, in
  `wow-viewer/data-harvester/src/harvester/v50/training_curriculum.py`
- [ ] T015 [P] [US2] Extend curriculum tests: PVPZone02/Kalidar refusal, full-signal carry-through
  (a signal present in the source store appears in the curriculum), split determinism unchanged, in
  `wow-viewer/data-harvester/tests/v50/test_training_curriculum.py`
- [ ] T016 [US2] Build the real curriculum from the rebuilt stores (CPU-side, assistant-runnable)
  as `wow-viewer/output/datasets/v50/v50.1/curriculum-0_5_3_3368-corrected_v3.zarr`
  (`--val-fraction 0.15`), verify SC-003 (map restriction, full signal list, deterministic split),
  and record the proof in `specs/112-v50-height-model/quickstart.md`

**Checkpoint**: SC-003 proven. No US3 work before this.

---

## Phase 5: User Story 3 — Height-First Model with a Relative Target (Priority: P3)

**Goal**: A lean minimap-RGB → relative-height model whose target is altitude-offset-invariant,
trained (by the user) on the corrected curriculum, beating the tile-mean baseline with best epoch
after epoch 1.

**Independent Test**: Target property test proves offset invariance; the user-run training's
`training_summary.json` shows `best_epoch > 1` and `tile_mean_baseline` beaten; SC-005 visual
review passes on held-out tiles from both maps.

- [ ] T017 [US3] Implement the Relative-Height Target Contract (`contract_version="v112.1"`,
  encode/decode with `RANGE_FLOOR`, per data-model.md) and the lean from-scratch CNN
  (research.md Decision 6: small encoder/spatial decoder, single output head, no pretrained
  backbone) in `wow-viewer/data-harvester/src/harvester/v50/height_relative_model.py`
- [ ] T018 [P] [US3] Add model/target tests: encode/decode exact round-trip incl. flat-tile floor,
  the FR-007 property test (constant offset added to a tile's heights leaves the target
  byte-identical), forward pass shape/grad sanity on a tiny fixture, in
  `wow-viewer/data-harvester/tests/v50/test_height_relative_model.py`
- [ ] T019 [US3] Implement the trainer (curriculum-schema gate, `--val-key split` holdout,
  Kalimdor/Azeroth-only evaluation guard per FR-011, per-epoch metrics + in-run
  `tile_mean_baseline` + `target_contract_version` in `training_summary.json` per FR-010,
  epoch-1-best flagged as structural failure per the execution contract) in
  `wow-viewer/data-harvester/src/harvester/v50/height_relative_train.py` + thin
  `wow-viewer/data-harvester/scripts/v50_train_height_relative.py`
- [ ] T020 [P] [US3] Add trainer contract tests (CPU-safe, no CUDA): store-schema gate refusal,
  out-of-scope-map evaluation refusal, summary fields present after a mocked 2-epoch loop, baseline
  computation correctness on a fixture, in
  `wow-viewer/data-harvester/tests/v50/test_height_relative_train.py`
- [ ] T021 [US3] **USER RUNS**: training on the corrected curriculum (command + estimate printed
  in `specs/112-v50-height-model/quickstart.md` Phase 2); assistant reviews the resulting
  `training_summary.json` against SC-004 and prepares the SC-005 side-by-side reconstruction
  review (decode predictions to world units, render held-out Kalimdor/Azeroth tiles for the user's
  visual judgment)

**Checkpoint**: SC-004 numerically proven, SC-005 user-judged. Feature complete.

---

## Phase 6: Polish & Cross-Cutting

- [ ] T022 [P] Update `wow-viewer/docs/dataset-preparation-userguide.md` §8 and
  `wow-viewer/data-harvester/README.md` to route the training path through Spec 112 (corrected
  configs, corrected curriculum, `v50_train_height_relative.py`) and mark the WDL-prior §8.4
  commands as the rejected legacy lane
- [ ] T023 Run the full focused suite (`tests/v50/`, `tests/test_v50_contract.py`,
  `tests/test_v50_build_command.py` + the new C# focused filters) and record exact results in
  `specs/112-v50-height-model/quickstart.md`; update `wow-viewer/memory-bank/activeContext.md` and
  `progress.md` per Memory Bank Discipline

## Dependencies & Execution Order

- **Phase 1 → 2 → 3 → 4 → 5 → 6** strictly (One Phase at a Time; each checkpoint is a gate).
- Within Phase 3: T005/T006 and T007 are independent of T008–T010; T011 needs T008; T012 (user)
  needs T005–T011; T013 needs T010+T012.
- Within Phase 5: T017/T018 before T019/T020; T021 (user) last.
- US2 depends on US1's rebuilt stores; US3 depends on US2's curriculum. The stories are
  independently *testable* (fixtures) but the real-data chain is sequential by design.

## Parallel Opportunities

- T002 ∥ T003 (different files); T006 ∥ T007's investigation; T009 ∥ T010's fixtures;
  T015 ∥ T016 prep; T018 ∥ T019 skeleton; T020 ∥ T021 prep; T022 ∥ T023.

## Implementation Strategy

MVP is Phase 3 (US1): an honest corpus is independently valuable even if no model ever trains.
Each user-run gate (T012, T021) is a hard stop — prepared, printed, waited on.

## Task Summary

- **Total tasks**: 23 (Setup 1, Foundational 3, US1 9, US2 3, US3 5, Polish 2)
- **User-executed**: T012 (rebuild), T021 (training)
- **Suggested MVP**: T001–T013
