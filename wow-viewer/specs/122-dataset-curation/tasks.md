---
description: "Task list for Spec 122 canonical dataset curation and signal-mismatch bucketing"
---

# Tasks: Canonical Dataset Curation and Signal-Mismatch Bucketing

**Input**: Design documents from `/specs/122-dataset-curation/`

**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included — C# tests under `wow-viewer/tests/WowViewer.Core.Curation.Tests/` (mirrors the
`WowViewer.Core.PM4.Tests` convention), Python tests under `data-harvester/tests/`, matching this
project's per-slice testing convention.

**Organization**: Tasks grouped by user story in spec priority order (US1 P1 canonical
classification, US2 P1 full bucket access, US3 P2 synthetic-fidelity, US4 P3 legacy consolidation).
Every task here is assistant-executable — no GPU, no billed cloud step, no user-run training
appears anywhere in this feature (plan.md Technical Context: CPU-only classification logic).

**Implementation-time finding to carry forward**: `Parquet.Net` is already a project dependency
(`WowViewer.Core.IO.csproj`) and is already used to *read* Parquet (`V18StorePlacementsReader.cs`),
but nothing in this codebase today *writes* Parquet from C# — every existing Parquet sidecar
(`decoded_metadata.parquet`, `index.parquet`, etc.) is written Python-side from JSON the C#
harvester streams. `CurationManifestWriter.cs` (T011) is therefore this repo's first C#-side
Parquet writer, not a copy of an existing one — budget it accordingly and validate the written
files are readable by `pyarrow` on the Python side (T012), since that round-trip is the actual
proof the new contract works.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1/US2/US3/US4)
- C# paths are relative to `wow-viewer/`; Python paths relative to `wow-viewer/data-harvester/`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the new project skeletons so later phases have somewhere to add code.

- [ ] T001 Create `src/core/WowViewer.Core.Curation/WowViewer.Core.Curation.csproj` (net10, `Nullable` enabled, matching sibling `WowViewer.Core.IO.csproj` conventions), project references to `WowViewer.Core` and `WowViewer.Core.IO`, package reference to `Parquet.Net` (already used elsewhere in the solution, version pinned to match `WowViewer.Core.IO.csproj`'s existing reference)
- [ ] T002 [P] Add `WowViewer.Core.Curation` to `WowViewer.slnx`
- [ ] T003 [P] Create `tests/WowViewer.Core.Curation.Tests/WowViewer.Core.Curation.Tests.csproj` (mirrors `WowViewer.Core.PM4.Tests`'s test-framework/package references exactly), add to `WowViewer.slnx`
- [ ] T004 [P] Create empty `data-harvester/src/harvester/curation_store.py` module with a module docstring stating its purpose (thin reader over the C#-written curation manifest, per plan.md D-01) — no logic yet, that lands in US2

**Checkpoint**: `dotnet build WowViewer.slnx -c Debug` succeeds with the two new empty projects; `uv run python -c "import harvester.curation_store"` succeeds.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The shared record types, run-record schema, and Parquet writer every user story's
checks write through. MUST complete before any user story.

**⚠️ CRITICAL**: No user story work begins until this phase is complete.

- [ ] T005 Implement `src/core/WowViewer.Core.Curation/CurationRecord.cs`: `TileCurationRecord` and `MismatchFinding` types matching data-model.md's Tile Curation Record / Mismatch Finding tables exactly (field names, enum value sets as string constants — `DifficultyBucket`, `CoverageBucket`, `LightingBucket`, `SyntheticFidelityStatus`, `MismatchCategory`, `MismatchSeverity`, `Evaluability`)
- [ ] T006 [P] Test `tests/WowViewer.Core.Curation.Tests/CurationRecordTests.cs`: construction round-trips every field; enum-backed fields reject an out-of-set string value
- [ ] T007 Implement `src/core/WowViewer.Core.Curation/CurationRunRecord.cs`: `v50-curation-run-v1` JSON record type (data-model.md Curation Run Record table) with `System.Text.Json` serialization, a `CurationRunId` generator (deterministic-enough to be a real identifier, e.g. build+store+timestamp), and a `Verify(int expectedTileCount)` method that throws if `TileCount` doesn't match — this is the SC-006 hard gate, enforced in code, not left to a caller to remember
- [ ] T008 [P] Test `tests/WowViewer.Core.Curation.Tests/CurationRunRecordTests.cs`: JSON round-trip preserves every field; `Verify` throws on a tile-count mismatch and passes on a match
- [ ] T009 Implement `src/core/WowViewer.Core.Curation/CurationManifestWriter.cs`: writes `curation_manifest.parquet` (one row per `TileCurationRecord`) and `curation_findings.parquet` (one row per `MismatchFinding`) using `Parquet.Net`'s writer API, plus `curation_run.json` (serialized `CurationRunRecord`) and the `curation/latest` pointer file, under `<store>/curation/<curation_run_id>/` (data-model.md path convention). Read-only with respect to the source store (FR-014) — opens the store only to read `index.parquet`'s row count for the T007 tile-count check, never writes into the store's own directory tree.
- [ ] T010 [P] Test `tests/WowViewer.Core.Curation.Tests/CurationManifestWriterTests.cs`: writing a fixture set of records produces both Parquet files with the correct row counts and column set; the `latest` pointer resolves to the just-written `curation_run_id`; a second write with a different `curation_run_id` does not overwrite or mutate the first run's directory (FR-012 reproducibility-across-reruns)
- [ ] T011 [P] Test `data-harvester/tests/test_curation_store.py::test_manifest_written_by_csharp_is_pyarrow_readable`: given a small fixture manifest written by a C# test harness (or a checked-in fixture Parquet file produced once by T010's test), confirm `pyarrow.parquet.read_table` reads it with the exact column names/dtypes `data-model.md` specifies — this is the real cross-language contract proof (the note above), not just a C#-side unit test

**Checkpoint**: Record types, run-record schema, and the Parquet writer all work and are proven readable from Python. User story implementation can begin.

---

## Phase 3: User Story 1 - One Canonical Classification Pass (Priority: P1) 🎯 MVP

**Goal**: Every tile in a v50 store gets a durable bucket assignment and mismatch findings from one
C# pass, consolidating `v16_curation.py`, `mismatch_detector.py`, and `spec111/lighting_buckets.py`'s
logic (spec US1).

**Independent Test**: Run `curate` against an existing v50 store; every tile receives exactly one
set of bucket assignments and zero-or-more findings; a tile known from prior sessions to be blank
or height-normal-mismatched is classified accordingly (spec US1 acceptance 1-4).

### Implementation for User Story 1

- [ ] T012 [US1] Implement `src/core/WowViewer.Core.Curation/Buckets/DifficultyBucketClassifier.cs`: ports `v16_curation.py`'s `DIFFICULTY_BUCKETS`/relief-based classification logic (`height_gradient_strength`, `normal_relief`) into C#, operating on `TerrainTileTensorPack`'s already-decoded `height_257`/`normal_xyz` arrays
- [ ] T013 [P] [US1] Test `tests/WowViewer.Core.Curation.Tests/DifficultyBucketClassifierTests.cs`: a flat fixture tile classifies `easy`; a high-relief fixture classifies `hard`/`pathological`, matching the four-bucket boundary logic ported from `v16_curation.py`
- [ ] T014 [US1] Implement `src/core/WowViewer.Core.Curation/BlankTileDetector.cs` + `src/core/WowViewer.Core.Curation/Buckets/CoverageBucketClassifier.cs`: ports `is_blank_what_plate` (near-zero height variance AND near-zero alpha/mcly/liquid/object coverage) and `mcly_painted_coverage` into the `coverage_bucket` (`well_covered`/`low_coverage`/`blank`) assignment
- [ ] T015 [P] [US1] Test `tests/WowViewer.Core.Curation.Tests/BlankTileDetectorTests.cs`: a fixture tile with zero height variance and zero paint/liquid/object coverage classifies `blank`; a normal fixture does not
- [ ] T016 [US1] Implement `src/core/WowViewer.Core.Curation/Mismatch/HeightNormalMismatchDetector.cs`: ports `mismatch_detector.py`'s `compute_tile_mismatch_metrics`/`detect_mismatches` exactly, including its 4-level severity thresholds and reason strings (`height_flat_vs_normal_varied`, `insufficient_normal_coverage`, `flat_normals`, `no_normal_data`), emitting `MismatchFinding` rows with `evaluability=not_evaluable` when `normal_xyz`/`normal_mask` are absent (never guessed — spec Edge Cases)
- [ ] T017 [P] [US1] Test `tests/WowViewer.Core.Curation.Tests/HeightNormalMismatchDetectorTests.cs`: reproduces `mismatch_detector.py`'s own test fixtures/thresholds (flat height + varied normal -> `high`/`medium`/`low` severity per the ratio boundaries; insufficient normal coverage -> `not_evaluable`, not a false negative)
- [ ] T018 [US1] Implement `src/core/WowViewer.Core.Curation/Mismatch/NonFiniteSignalDetector.cs` and `src/core/WowViewer.Core.Curation/Mismatch/HasFlagTruthfulnessDetector.cs`: NaN/Inf checks across the tile's numeric arrays, and a check that each `has_*`-style presence flag is only true when its backing array actually carries non-default data (ports the `verify_v18`-style defect checks named in research.md/spec FR-005)
- [ ] T019 [P] [US1] Test `tests/WowViewer.Core.Curation.Tests/NonFiniteSignalDetectorTests.cs` and `HasFlagTruthfulnessDetectorTests.cs`: a fixture with an injected NaN is flagged `non_finite_value`; a fixture whose presence flag is true but backing array is all-default is flagged `has_flag_mismatch`
- [ ] T020 [US1] Implement `src/core/WowViewer.Core.Curation/Buckets/LightingBucketClassifier.cs`: ports `spec111/lighting_buckets.py`'s status vocabulary (`matched`/`low_confidence_ambiguous`/`low_confidence_flat_terrain`/`not_evaluated`) by reading the existing `MinimapShadingMatch.Evaluate` result already computed for a tile, plus the `MapAccumulator` reconciliation invariant (bucket_total + not_evaluated + low_confidence == total_eligible_tiles) as a self-check the classifier runs per map
- [ ] T021 [US1] Add the `curate` subcommand to `tools/harvest/WowViewer.Tool.Harvest/Program.cs` per `contracts/cli-contract.md`: argument parsing (`--clients-root`, `--build`, `--store`, `--checks`, `--map`, `--write`), dry-run-by-default plan printer (planned tile count, checks that will run vs. skip-for-missing-signal, output paths), orchestration calling every classifier/detector from T012-T020 per tile, `CurationManifestWriter`/`CurationRunRecord` write path gated behind `--write`, and the SC-006 full-coverage verification before reporting success
- [ ] T022 [US1] Real (non-fixture) smoke validation: dry-run `curate` against an existing on-disk v50 store (per quickstart.md §2), then `--write` (quickstart.md §3), confirming the printed `tile_count` matches the store's real row count exactly and every check that has backing signals in that store actually ran (not silently skipped)

**Checkpoint**: US1 delivers one canonical classification pass end-to-end, proven on real data — the MVP.

---

## Phase 4: User Story 2 - Every Bucket Stays Fully Accessible (Priority: P1)

**Goal**: Querying any bucket or finding, including "bad" ones, is exactly as easy as querying
"clean" — no special-case recovery path (spec US2).

**Independent Test**: Query the real store's curation output for a non-clean bucket (e.g.
`blank` or a `height_normal_mismatch` finding) and confirm full, equal-effort completeness against
the clean-bucket query (spec US2 acceptance 1-3).

### Implementation for User Story 2

- [ ] T023 [US2] Implement `data-harvester/src/harvester/curation_store.py`: `load_curation_manifest(store_path, curation_run_id=None)` and `load_curation_findings(store_path, curation_run_id=None)` per `contracts/cli-contract.md`'s Python-side read access section — both resolve `<store>/curation/latest` by default, both return the full table with zero filtering, matching the query contract in data-model.md ("Query Contract: every bucket equally accessible")
- [ ] T024 [P] [US2] Test `data-harvester/tests/test_curation_store.py::test_load_curation_manifest_and_findings`: against the fixture Parquet produced in T011, confirm both loaders return every row with no default filtering, and that filtering a non-clean bucket column (e.g. `coverage_bucket == "blank"`) and a clean bucket column both work identically (same call shape, same completeness) — this is US2's FR-009 guarantee expressed directly as a test
- [ ] T025 [US2] Real-store proof (quickstart.md §4): against the store `curate --write` already produced in T022, run the query commands for a non-clean bucket and a clean bucket and confirm both return non-trivial, complete results with identical effort
- [ ] T026 [US2] Document the Selection Record convention from data-model.md (a future trainer spec's own run record should carry a `curation_selection` block referencing `curation_run_id` + bucket filter + excluded counts) as a short addendum in `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`, pointing future spec authors at the new canonical manifest — this is documentation only, no new schema owned by this feature (per data-model.md's Selection Record note)

**Checkpoint**: Both US1 and US2 work independently and together — full-coverage classification, and equal-access querying of every bucket.

---

## Phase 5: User Story 3 - Synthetic-vs-Authored Minimap Fidelity Finding (Priority: P2)

**Goal**: Every tile with both a synthesized and authored minimap gets a durable, queryable
fidelity finding built on the existing `MinimapShadingMatch` correlation machinery (spec US3).

**Independent Test**: Run the fidelity check on tiles with both minimap sources; a tile a human
would judge as a poor synthetic match scores measurably worse than one judged a good match (spec
US3 acceptance 1-3, SC-004).

### Implementation for User Story 3

- [ ] T027 [US3] Implement `src/core/WowViewer.Core.Curation/Mismatch/SyntheticFidelityDetector.cs`: for a tile with both `minimap_rgb` (synthesized) and `minimap_rgb_authored`, invokes the existing `MinimapShadingMatch.Evaluate` and records its best-candidate correlation score as `synthetic_fidelity_score` plus a `synthetic_fidelity_gap` `MismatchFinding` when the score falls below a documented threshold (severity scaled by how far below); for a tile missing either minimap source, emits `synthetic_fidelity_status=not_evaluable` (spec US3 acceptance 2) rather than a false pass
- [ ] T028 [P] [US3] Test `tests/WowViewer.Core.Curation.Tests/SyntheticFidelityDetectorTests.cs`: a fixture pair with high shading correlation scores as evaluated with low/no severity; a fixture pair with deliberately mismatched shading (e.g. inverted normal-derived shading) scores lower and produces a `synthetic_fidelity_gap` finding; a tile missing the authored minimap is `not_evaluable`, not silently passing
- [ ] T029 [US3] Wire `SyntheticFidelityDetector` into the `curate` orchestration (T021) as one more per-tile check, gated the same way as every other check (skipped-and-reported, not silently absent, when a store lacks `minimap_rgb_authored` entirely)
- [ ] T030 [US3] Real-data validation (SC-004): run `curate --write` against a real store with both minimap sources, then manually inspect a handful of tiles at the extremes of the `synthetic_fidelity_score` distribution (best and worst) to confirm the score visually tracks synthetic-render quality — record the finding (pass/fail/needs-threshold-tuning) in this spec's progress notes, matching this project's established visual-verification discipline

**Checkpoint**: All three of US1/US2/US3 work together — the synthetic-vs-authored gap the user specifically flagged mid-session is now a durable, queryable signal, not an ad hoc one-off comparison.

---

## Phase 6: User Story 4 - Legacy Scripts Stop Being the Source of Truth (Priority: P3)

**Goal**: The six named scattered scripts each reach a documented disposition (thin reader or
retired), per research.md D-04's per-script table — not a blanket rule (spec US4).

**Independent Test**: Confirm each of the six scripts either reads from the canonical
classification or is clearly marked historical, with no remaining path that silently recomputes a
competing answer (spec US4 acceptance 1-2).

### Implementation for User Story 4

- [ ] T031 [US4] Implement `data-harvester/scripts/spec122_compare_legacy_mismatch.py` per `contracts/cli-contract.md`'s legacy comparison command: runs `mismatch_detector.py`'s existing `detect_mismatches` and the new C# `HeightNormalMismatchDetector`'s output (read via T023's loader) against the same real store's tiles, writes a diff report (agreements, disagreements-with-reason)
- [ ] T032 [US4] Run the SC-003 comparison for real (quickstart.md §5) against the store already classified in T022/T030; read the report and record the verdict (match, or a documented, justified improvement) — this is the gate research.md D-05 requires before any retirement below
- [ ] T033 [US4] Real-caller search (matching the Spec 109 Phase 6 / Spec 116 lesson: verify before disposing) — grep the full `wow-viewer/` tree for imports of `v16_curation`, `mismatch_detector`, and `spec111.lighting_buckets`' specific function names; record what still calls each, directly informing which of T034-T036 apply a thin-shim vs. a documented-retired disposition
- [ ] T034 [US4] Convert `data-harvester/src/harvester/v16_curation.py` and `data-harvester/src/harvester/mismatch_detector.py` per T033's findings: either thin readers delegating to `curation_store.py` (keeping any real-caller function names as compatibility wrappers), or a documented-retired header if T033 found no real remaining callers — determined by T033, not assumed
- [ ] T035 [P] [US4] Convert `data-harvester/src/harvester/spec111/lighting_buckets.py` per T033's findings and research.md D-04, same either/or disposition
- [ ] T036 [P] [US4] Add a documented-retired header to `data-harvester/scripts/build_v16_curation_manifest.py` pointing at `curate` as the v50-era replacement (research.md D-04 — this one is retired outright, not a shim, since it targets the legacy V16 store shape)
- [ ] T037 [US4] Confirm (do not modify) `data-harvester/scripts/v50_audit_signal_coverage.py` and `data-harvester/scripts/v50_audit_artifacts.py` are explicitly out of scope per research.md D-04 (different concern: artifact/store lifecycle, not per-tile quality) — add a one-line comment in each pointing at the new curation manifest as a *related but distinct* concern, so a future reader isn't left to rediscover this distinction
- [ ] T038 [US4] Add a short pointer addendum to `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md` naming the canonical curation manifest and its location convention, so this feature's existence is discoverable from the doc every prior v50 spec already reads first (AGENTS.md "Spec Docs Are Source of Truth")

**Checkpoint**: All four user stories complete. Curation has exactly one canonical, durable home; the six named scripts are honestly dispositioned, not silently left to drift.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final validation sweep and continuity bookkeeping.

- [ ] T039 [P] Run the full quickstart.md end-to-end (steps 1-6) against a real v50 store in one pass, confirming every command as documented actually runs — this project's repeatedly-learned lesson (memory note: "tests pass" is not "the documented CLI works") applies here as much as anywhere
- [ ] T040 [P] `dotnet build WowViewer.slnx -c Debug` (0 errors) and `dotnet test WowViewer.slnx -c Debug --filter WowViewer.Core.Curation.Tests` (all pass)
- [ ] T041 [P] `uv run python -m pytest tests/test_curation_store.py tests/spec122/ -q` (if a `tests/spec122/` package is warranted for the comparison-script tests) and `ruff check` / `python -m py_compile` over all touched Python files
- [ ] T042 Update `memory-bank/activeContext.md` and `memory-bank/progress.md` with a compressed summary of this feature's outcome (AGENTS.md Rule 11 / constitution "Memory Bank Discipline") — what shipped, the SC-003 comparison verdict, and the final per-script disposition table

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories (the record types and
  writer every check writes through).
- **User Story 1 (Phase 3)**: Depends on Foundational only. This is the MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational + US1's real written manifest (T022) to prove
  against — cannot be meaningfully validated without a real manifest to query, though its own code
  (T023) only depends on the schema from Phase 2.
- **User Story 3 (Phase 5)**: Depends on Foundational + the `curate` orchestration skeleton from
  US1 (T021) to wire into — otherwise independently addable.
- **User Story 4 (Phase 6)**: Depends on US1 (a real classified store to compare against) and
  logically on US2/US3 having landed their checks first, so the comparison and disposition work is
  against the feature's final check set, not a partial one.
- **Polish (Phase 7)**: Depends on all four user stories.

### Parallel Opportunities

- All `[P]`-marked tasks within a phase touch different files and can run in parallel.
- Within Phase 3 (US1), the four classifier/detector implementation-plus-test pairs (T012-T013,
  T014-T015, T016-T017, T018-T019) are independent of each other and can be built in parallel before
  T020-T022 (which depend on all of them existing to wire into `curate`).
- T034/T035/T036/T037 (Phase 6 per-script dispositions) are independent files and can run in
  parallel once T033's caller search is done.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational).
2. Complete Phase 3 (US1) — canonical classification, real-store proof.
3. **STOP and VALIDATE**: confirm SC-006 (full tile coverage) on a real store before continuing.

### Incremental Delivery

1. Setup + Foundational → Foundation ready.
2. US1 → real-store proof → MVP: one canonical classification pass exists and is trustworthy.
3. US2 → proof that every bucket, not just "clean," is equally queryable — the spec's explicit
   corrective requirement.
4. US3 → the synthetic-vs-authored fidelity finding the user specifically asked for mid-session.
5. US4 → the six scattered scripts stop being live alternate definitions of "clean" — the
   consolidation actually sticks.
6. Polish → full quickstart proof + memory-bank continuity update.

---

## Notes

- `[P]` tasks touch different files with no dependency on an incomplete task.
- `[Story]` labels trace every task back to its spec user story.
- Every task in this feature is assistant-executable; there is no GPU/billed step to hand off to
  the user anywhere in this feature (unlike most prior specs in this repo).
- Commit after each task or logical group (AGENTS.md Rule 6: one concern per committable change).
- Stop at each phase checkpoint and validate against a real store before moving to the next phase
  (AGENTS.md Rule 8: one phase at a time, done means validated).
