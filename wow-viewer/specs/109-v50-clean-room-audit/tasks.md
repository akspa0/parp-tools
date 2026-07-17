# Tasks: V50 Clean-Room Dataset and Repository Reset

**Input**: Design documents from `specs/109-v50-clean-room-audit/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Fixture-first tests are required for trust, migration, and deletion boundaries. Heavy
real-data verification/build and destructive apply commands are user-run only.

**Organization**: Tasks follow the three user stories, then converge command ownership and perform
reviewed disk reclamation. One phase must be validated before the next starts.

## Phase 1: Setup — Authority and frozen scope

**Purpose**: Resolve policy and freeze what v50 means before implementation.

- [x] T001 Reconcile the `H:\CLIENTS` fast-SSD policy and prepare the guarded clean-slate bootstrap in `AGENTS.md`, `wow-viewer/AGENTS.md`, `wow-viewer/.specify/memory/constitution.md`, and `wow-viewer/scripts/clean-legacy-outputs.ps1`
- [ ] T002 Freeze the complete v50 signal table and V18 per-signal migration policy in `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`
- [x] T002a Freeze V50 liquid provenance: make `liquid_mask`/`liquid_height` fresh-only, require `wl_liquid_surface_quads_v1` for WL sources, and refuse the legacy mixed-copy wrapper in `harvester/v50_contract.py` and `scripts/v50_build_dataset.py`
- [ ] T003 [P] Record approved generated-data roots and protected roots in `wow-viewer/specs/109-v50-clean-room-audit/research.md`
- [ ] T004 [P] Create the canonical package and fixture directories at `wow-viewer/data-harvester/src/harvester/v50/` and `wow-viewer/data-harvester/tests/v50/`
- [ ] T005 Validate Spec 109 requirements, plan, schemas, and command examples in `wow-viewer/specs/109-v50-clean-room-audit/`

**Checkpoint**: Client-root policy, protected roots, signal scope, and schemas agree.

---

## Phase 2: Foundational — Shared contracts and safety primitives

**Purpose**: Build the fail-closed primitives required by every user story.

**CRITICAL**: No user-story implementation starts before this phase passes fixture tests.

- [x] T006 [P] Add failing provenance/store-manifest contract tests in `wow-viewer/data-harvester/tests/v50/test_contracts.py` (23 tests: ArtifactRecord, DatasetStoreManifest schema round-trip, RowLineage, migrated release gates)
- [x] T007 [P] Add failing resolved-path and protected-root tests in `wow-viewer/data-harvester/tests/v50/test_path_policy.py` (7 tests; 2 symlink-escape cases self-skip via a runtime probe on hosts without symlink-creation privilege -- this host skipped both; not yet observed passing on a privileged host, see quickstart.md)
- [x] T008 Implement ArtifactRecord, DatasetStoreManifest, DatasetSignal, RowLineage, and verification enums (`ProofLevel`, `TrustState`, `Disposition`, `MigrationPolicy`, `FinalizationState`) in `wow-viewer/data-harvester/src/harvester/v50/contracts.py`; `to_dict()`/`validate()` match `contracts/v50-provenance.schema.json` exactly without a runtime jsonschema dependency
- [x] T009 Implement deterministic file, metadata-tree, Parquet, and manifest identities in `wow-viewer/data-harvester/src/harvester/v50/identity.py`; Parquet identity is invariant to physical layout/compression but sensitive to content (proven in test)
- [x] T010 Implement configurable client-library/build evidence without source hardcoding in `wow-viewer/data-harvester/src/harvester/v50/client_evidence.py`; required paths, executable candidates, and archive glob are all caller-supplied, never hardcoded
- [x] T011 Implement approved-root/protected-root resolution that rejects links and escapes in `wow-viewer/data-harvester/src/harvester/v50/path_policy.py`; protected root wins even when nested inside an approved root
- [x] T012 Migrate release identity helpers from `wow-viewer/data-harvester/src/harvester/v50_contract.py` into `wow-viewer/data-harvester/src/harvester/v50/contracts.py`; `v50_contract.py` is now a thin re-export shim (existing `tests/test_v50_contract.py` and all spec103/108 callers pass unchanged)
- [x] T013 Ran fixture-only contract/path tests from `wow-viewer/data-harvester/`: `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ -q` -> 90 passed, 2 skipped (symlink privilege), 0 failed. Recorded in `quickstart.md`

**Checkpoint**: Missing identities, unsafe paths, and incompatible releases fail before I/O-heavy work.

---

## Phase 3: User Story 1 — Fail-Closed Dataset Trust Boundary (Priority: P1) — MVP

**Goal**: Inventory all historical artifacts as unverified and audit V18 per signal without promotion.

**Independent Test**: Fixtures containing old names, forged v50 attributes, incomplete provenance,
mixed pass/fail signals, and a valid manifest yield the expected trust states; only the valid manifest
can be promoted.

### Tests for User Story 1

- [x] T014 [P] [US1] Add failing metadata-only inventory tests in `wow-viewer/data-harvester/tests/v50/test_inventory.py` (7 tests: forged v50 attributes, old pre-v50 names, kind classification, metadata-only content identity, artifact_id vs content_identity distinctness, missing root, determinism)
- [x] T015 [P] [US1] Add failing per-signal V18 verification tests, including rejected `holes_16`, in `wow-viewer/data-harvester/tests/v50/test_verify_v18.py` (9 tests)

### Implementation for User Story 1

- [x] T016 [US1] Implement deterministic metadata-only artifact discovery in `wow-viewer/data-harvester/src/harvester/v50/inventory.py`; never reads array payloads, never sets trust_state to anything but UNVERIFIED regardless of name/attributes
- [x] T017 [US1] Implement complete v50 store/index/lineage validation in `wow-viewer/data-harvester/src/harvester/v50/verify_store.py` (FR-005: schema/dtype/shape, row-count agreement, required-signal truthfulness, content-integrity hashes, partition leakage -- reuses the existing `harvester.spec103.prefab_curation.validate_source_group_split` rather than reimplementing it); 8 fixture tests
- [x] T018 [US1] Implement V18 per-signal audit planning and known-defect rejection in `wow-viewer/data-harvester/src/harvester/v50/verify_v18.py`; per-(signal,row) results, not a whole-signal verdict (FR-016)
- [x] T019 [US1] Add `inventory` and `verify-v18` subcommands in `wow-viewer/data-harvester/scripts/v50_audit_artifacts.py`; both smoke-tested end-to-end against synthetic fixtures (a real filesystem tree for inventory, a real Zarr+Parquet store for verify-v18, which correctly rejected a NaN-poisoned row and blacklisted `holes_16`). `verify-v18`'s fresh-client cross-validation (plan.md Phase 2 step 2) is an explicit, documented gap -- deferred until Spec 109 T002 freezes the signal catalog
- [x] T020 [US1] Ran the real read-only inventory command against everything currently on disk (`output/`, `models/`, `data-harvester/checkpoints`, `data-harvester/tmp`, `data-harvester/models`): 12 artifacts, ~15.6 GB, all `unverified`/`quarantine`. Report at `output/reports/v50/v50.1/inventory.json`; summarized in `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`
- [x] T021 [US1] Ran all fixture-only User Story 1 tests: `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ -q` -> 114 passed, 2 skipped (symlink privilege), 0 failed. Recorded in `quickstart.md`

**Checkpoint**: All legacy artifacts remain unverified; no name or release attribute can launder trust.

---

## Phase 4: User Story 2 — Reviewable Repository and Disk Cleanup (Priority: P2)

**Goal**: Produce an exact, dependency-aware cleanup plan and dry run without deleting anything.

**Independent Test**: A fixture inventory containing protected, depended-on, safe obsolete, linked,
and out-of-root targets includes only the safe obsolete target in the approved dry-run plan.

### Tests for User Story 2

- [x] T022 [P] [US2] Add failing dependency and cleanup-plan tests in `wow-viewer/data-harvester/tests/v50/test_cleanup.py` (7 tests; the exact mixed fixture from tasks.md's Independent Test -- protected/depended-on/out-of-root/safe-obsolete -- included only the safe-obsolete target)

### Implementation for User Story 2

- [x] T023 [US2] Implement dependency discovery across manifests, checkpoints, reports, and known output layouts in `wow-viewer/data-harvester/src/harvester/v50/dependencies.py`; scans manifest/report JSON for references by path or content hash rather than hardcoding a not-yet-frozen manifest schema
- [x] T024 [US2] Implement size inventory and deterministic cleanup-plan generation in `wow-viewer/data-harvester/src/harvester/v50/cleanup.py`; matches `v50-cleanup-plan.schema.json` exactly -- a candidate only ever appears in `targets` once it already has `dependency_check=pass` and `approved=true`, never as a marked-rejected entry
- [x] T025 [US2] Add `plan` (dry-run only, no apply) subcommand in `wow-viewer/data-harvester/scripts/v50_cleanup_artifacts.py`; disposition and replacement-proof are explicit human-reviewed JSON inputs, never inferred from a filename
- [x] T026 [US2] `docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md` already carried the per-category disposition table from the original audit; added the Phase 4 proof-of-concept run against real `tmp/` scratch artifacts as a concrete worked example
- [x] T027 [US2] Ran fixture-only cleanup planning tests (`tests/v50/ tests/test_v50_contract.py tests/spec103/` -> 121 passed, 2 skipped, 0 failed) and a real local read-only dry run: zero reviewed dispositions -> 0 targets (correct, nothing has been human-reviewed yet); with the two genuinely-disposable `data-harvester/tmp/*_smoke` scratch artifacts explicitly dispositioned + proofed -> 2 targets, 12,901,439 bytes, `dry_run_complete=true`, nothing deleted. Recorded in `quickstart.md`

**Checkpoint**: Every deletion candidate has replacement/dependency proof; apply remains unavailable.

---

## Phase 5: User Story 3 — Complete V50 Dataset Baseline (Priority: P3)

**Goal**: Build complete canonical per-build v50 stores from verified V18 signals and fresh client
extraction, with curricula represented as manifests.

**Independent Test**: A fixture V18 store with passing, failing, missing, and known-defective signals
produces a complete v50 fixture store whose copied chunks hash-identically, rejected signals are
fresh/unavailable, all rows have lineage, and curricula contain no array payloads.

### Tests for User Story 3

- [x] T028 [P] [US3] Add failing selective-migration and resumability tests in `wow-viewer/data-harvester/tests/v50/test_migrate.py` (7 tests: copy-eligible planning, blacklist rejection, ledger append-only, bit-preserving copy round-trip)
- [x] T029 [P] [US3] Add failing complete-store/finalization and curriculum-manifest tests in `wow-viewer/data-harvester/tests/v50/test_store.py` (12 tests: partial vs complete store write, `finalization_state` gating, `TestCurriculumManifest` row-reference-only payloads)

### Implementation for User Story 3

- [x] T030 [US3] Implement the complete per-build v50 Zarr writer and finalization checks in `wow-viewer/data-harvester/src/harvester/v50/store.py`; `write_v50_store()`, `read_v50_manifest()`, `finalize_store()` -- finalization recomputes observed hashes from the actual written store and refuses `complete` unless every required signal's `content_identity` matches, proven against both a stale-manifest (correctly `incomplete`, exit 1) and the real written manifest (`complete`, exit 0)
- [x] T031 [US3] Implement bit-preserving verified V18 copy, fresh-signal slots, row lineage, and resume ledger in `wow-viewer/data-harvester/src/harvester/v50/migrate.py`; `plan_signal_migration()`, `MigrationLedger`/`MigrationLedgerEntry` (append-only), `copy_signal_row()`
- [x] T032 [US3] Integrate the existing C# harvester stream for fresh signals and new builds in `wow-viewer/data-harvester/src/harvester/v50/build.py`; `build_harvest_stream_command()`, `read_harvest_stream()` (reuses `harvester.raw_reader.read_tile_blob` for the inner-blob format rather than reimplementing it), `run_fresh_extraction()` -- gated behind `--confirm-run`, prints the command and returns without launching anything otherwise
- [x] T033 [US3] Implement immutable row-selection curricula over canonical stores in `wow-viewer/data-harvester/src/harvester/v50/curriculum.py`; `build_curriculum()`/`CurriculumManifest` store only `{store_id, row_id, source_group, split}` references, never array payloads, and reuse `harvester.spec103.prefab_curation.validate_source_group_split` for partition-leakage checks
- [x] T034 [US3] Replace the current thin wrapper with migrate-v18, build, verify, finalize, and curriculum commands in `wow-viewer/data-harvester/scripts/v50_build_dataset.py`
- [x] T035 [US3] Run fixture-only migration/store tests and document exact results in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md` -- `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ tests/spec111/ -q` -> 164 passed, 2 skipped (symlink privilege), 0 failed. All 5 CLI subcommands additionally smoke-tested end-to-end against a synthetic V18 Zarr fixture (`migrate-v18 --write-store` -> `finalize` -> `verify` (both the passing case and a deliberate hash-mismatch, which correctly failed closed with `proof_level=contract`, exit 1) -> `curriculum`), all against real Zarr/Parquet I/O, no fixtures mocked
- [x] T036 [US3] Prepare the bounded sampled V18 verification command for one user-selected build in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md` -- documented with the real `verify-v18` flag names from the implemented CLI; execution against a real build under `H:\CLIENTS` remains user-run only pending build selection
- [ ] T037 [US3] After user review of sampled proof, prepare the full user-run migration and fresh-build commands with duration and output estimates in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md` -- command shape is documented, but duration/output-size estimates require a real sampled run first (T036), which has not happened yet; genuinely blocked on the user selecting and reviewing one build

**Checkpoint**: The user-run v50 release is complete and fully verified before old datasets become deletable.

---

## Phase 6: Canonical V50 Rename and Ownership Convergence

**Purpose**: Finish the rename by moving implementation authority out of historical spec modules.

- [x] T038 [P] Add compatibility/import and cross-release rejection tests in `wow-viewer/data-harvester/tests/v50/test_command_compatibility.py` (14 tests: each of the 6 canonical `v50_*.py` entries imports `main` only from its `harvester.v50` owner and never from a historical spec103-named module; each of the 6 historical shims re-exports the same owner and defines no second `main()`; a wrong-release WDL checkpoint is rejected by `load_model`; all 4 moved command-owner modules share the identical `harvester.v50.contracts` gate objects, not a locally reimplemented copy)
- [x] T039 Move WDL-prior command ownership into `wow-viewer/data-harvester/src/harvester/v50/` and keep bounded shims in historical modules only where tests require them -- `wdl_prior_train.py`, `wdl_prior_infer.py`, `wdl_prior_evaluate.py`, `wdl_prior_visualize.py` are the real implementations now; `scripts/train_spec103_wdl_prior.py`/`infer_spec103_wdl_prior.py`/`evaluate_spec103_wdl_prior.py`/`visualize_spec103_wdl_prior.py` are thin re-export shims, kept because `tests/spec103/test_wdl_prior_sanity.py` imports `filter_deployable_rows`/subprocess-invokes `infer_spec103_wdl_prior.py` by that exact name
- [x] T040 Move terrain-refiner command ownership into `wow-viewer/data-harvester/src/harvester/v50/` and keep bounded shims in historical modules only where tests require them -- `terrain_refiner_train.py` (incl. `V7TileDataset`), `terrain_refiner_infer.py` are the real implementations; `scripts/train_spec103_v7.py`/`infer_spec103_v7.py` are thin shims, kept because the same test imports `V7TileDataset` from `train_spec103_v7` by name and `runpod/spec103/*.sh` invoke both by file path
- [x] T041 Update `wow-viewer/data-harvester/scripts/v50_train_wdl_prior.py`, `v50_generate_wdl_priors.py`, `v50_review_wdl_prior.py`, and `v50_visualize_wdl_prior.py` to import only canonical v50 owners
- [x] T042 Update `wow-viewer/data-harvester/scripts/v50_train_terrain.py` and `v50_infer_terrain.py` to import only canonical v50 owners
- [x] T043 Remove obsolete wrappers or aliases only after a full caller search and record each disposition in `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md` -- repo-wide search found no wrapper safe to delete: all 6 historical `scripts/*_spec103_*.py` files are load-bearing (direct symbol imports and a subprocess invocation from `tests/spec103/test_wdl_prior_sanity.py`, plus file-path invocations from `runpod/spec103/{train,smoke,verify_bundle}.sh`), so every one stays as a shim, not a deletion candidate. `scripts/package_spec103_runpod.py`'s `_SOURCE_DIRS` was missing `src/harvester/v50` -- without it the RunPod bundle would have shipped `train_spec103_v7.py`/`infer_spec103_v7.py` shims pointing at a module the bundle never packaged, a real regression this move would otherwise have introduced; fixed
- [x] T044 Run focused v50 contract, compatibility, dataset, and command tests from `wow-viewer/data-harvester/` -- `uv run python -m pytest tests/spec103/ tests/v50/ tests/test_v50_contract.py tests/spec111/ -q` -> 178 passed, 2 skipped, 0 failed (includes `test_command_compatibility.py`'s 14 new tests). All 12 touched scripts (6 canonical + 6 historical shims) additionally smoke-tested via `--help` to catch import errors invisible to unit tests -- all 12 OK. A full `uv run python -m pytest tests/ -q` run surfaced one real regression this move introduced -- `tests/test_v50_build_command.py` still asserted Phase 1's retired placeholder refusal message -- fixed by rewriting it against the current CLI contract (subcommand required, unrecognized subcommand rejected); full suite is then 568 passed, 43 skipped, 3 failed, and those 3 failures (`tests/v24/test_export_map.py`, `tests/v25/test_h1_coarse.py` x2) are the same pre-existing, unrelated failures confirmed earlier this session, reproducing identically on unmodified `HEAD`

**Checkpoint**: V50 has one implementation owner per workflow; historical names are explicit compatibility only.

---

## Phase 7: User-Reviewed Cleanup Apply and Final Proof

**Purpose**: Reclaim disk space only after the verified v50 replacement exists.

- [x] T045 Add failing cleanup-apply identity, interrupted-run, and post-check tests in `wow-viewer/data-harvester/tests/v50/test_cleanup_apply.py` (9 tests: wrong plan_id refused, missing `--confirm` refused, matching plan_id+confirm removes the target, drifted content since planning is skipped not deleted, a re-tampered protected root is re-checked at apply time rather than trusted from the plan, an interrupted-then-resumed run is idempotent and does not double-count recovered bytes, `to_dict()` round-trips every field)
- [x] T046 Implement hash-confirmed cleanup apply and post-cleanup verification in `wow-viewer/data-harvester/src/harvester/v50/cleanup.py` -- `CleanupApplyError`, `CleanupApplyResult`, `apply_cleanup_plan()`: refuses without `confirm=True`, refuses unless the caller's `expected_plan_id` matches `plan.plan_id` exactly, re-resolves every target against `PathPolicy` at execution time (never trusting the plan's own `approved_roots` snapshot), and rehashes each target's real on-disk content immediately before deleting it -- a target whose content changed since the plan was built is skipped, not deleted on stale evidence. A target already absent (a prior interrupted apply) is treated as already-done, not an error, and its bytes are not recounted
- [x] T047 Expose cleanup apply only with explicit plan hash and protected-root checks in `wow-viewer/data-harvester/scripts/v50_cleanup_artifacts.py` -- new `apply` subcommand takes `--plan`, `--plan-id` (must match the plan file's own `plan_id` verbatim), `--approved-root`/`--protected-root` (re-supplied fresh, not read from the plan file), and `--confirm`. Smoke-tested end-to-end against a real synthetic fixture file (not mocked): a wrong `--plan-id` correctly refused (exit 1, file untouched); the matching `--plan-id --confirm` run actually deleted the fixture file and reported `1 removed, 0 skipped, 22 bytes recovered`; re-running the identical command afterward reported `1 removed, 0 skipped, 0 bytes recovered` (idempotent, no double-count)
- [x] T048 Generate the real cleanup dry-run report under `wow-viewer/output/reports/v50/v50.1/` and have the user review exact targets and expected recovered bytes -- refreshed inventory (`inventory.json`, now 13 artifacts; the 2 new entries since Phase 4 are this report directory's own prior output and `models/.gitignore`, neither a disposal candidate) and refreshed cleanup plan (`cleanup-plan.json`) with the same 2 dispositioned targets as Phase 4 (`data-harvester/tmp/v18_smoke`, `.../v22_smoke`) reproduce byte-for-byte the same 2 targets, 12,901,439 bytes expected recovered, `plan_id=sha256:fc2c657b42c33fd852a57f4873e657cd8ccbcef021487057a2eeddb826a4e346`. **User review of this exact plan_id is the pending step before T050 may proceed** -- Codex does not review-and-approve on the user's behalf
- [x] T049 Prepare the exact user-run cleanup apply command in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md` without launching it -- done against the real plan above; the command is documented, not executed
- [ ] T050 After the user runs cleanup, verify v50 store identities, protected survivors, removed targets, and recovered bytes in `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md` -- blocked on the user actually running the real apply command (T048/T049); Codex does not run destructive real-disk operations without an explicit, in-the-moment go-ahead
- [ ] T051 Compress `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` after archiving stale chronology and preserving Spec 109 truth -- deferred until after T050's real run, so the compression reflects the final state rather than needing a second pass
- [x] T052 Run the full lightweight v50 test suite and documentation consistency check from `wow-viewer/data-harvester/` -- `uv run python -m pytest tests/v50/ tests/test_v50_contract.py tests/spec103/ tests/spec111/ tests/test_v50_build_command.py -q` -> 189 passed, 2 skipped, 0 failed. Full `tests/ -q` -> 577 passed, 43 skipped, 3 failed (the same 3 pre-existing, unrelated failures as Phase 6, reproducing on unmodified `HEAD`)

**Checkpoint**: Old approved artifacts are gone, protected/client/source roots remain intact, and v50 revalidates.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: Starts immediately; policy reconciliation blocks client reads.
- **Phase 2**: Depends on Phase 1 contract lock.
- **Phase 3 / US1**: Depends on Phase 2; delivers the MVP trust boundary.
- **Phase 4 / US2**: Depends on US1 inventory; produces dry-run cleanup only.
- **Phase 5 / US3**: Depends on US1 verification primitives; produces the replacement v50 data.
- **Phase 6**: Depends on US3 fixture proof so the canonical owner is concrete.
- **Phase 7**: Depends on verified user-run US3 output and reviewed US2 cleanup plan.

### User Story Dependencies

- **US1 (P1)**: Independent after foundation; mandatory MVP.
- **US2 (P2)**: Uses US1 inventory but is independently testable with fixtures; deletion apply waits for US3.
- **US3 (P3)**: Uses US1 trust primitives; does not require US2 planning to build a verified store.

### Parallel Opportunities

- T003 and T004 can proceed independently after T001 scope is understood.
- T006 and T007 are separate fixture files.
- T014 and T015 are separate read-only audit test surfaces.
- T028 and T029 separate migration and final-store contracts.
- T039 and T040 target independent WDL and terrain command families after compatibility tests exist.

## Implementation Strategy

### MVP first

1. Complete Phases 1 and 2.
2. Complete US1 through T021.
3. Stop and validate the read-only inventory/trust boundary.
4. Do not migrate data or prepare deletion apply until the MVP is accepted.

### Incremental delivery

1. Trust boundary and inventory.
2. Cleanup classification/dry run.
3. Complete v50 fixture migration and store contract.
4. User-run sampled proof, then full migration/build.
5. Command-owner rename convergence.
6. User-reviewed cleanup apply and post-delete proof.

## Task Summary

- **Total tasks**: 52
- **Setup/foundation**: 13
- **US1**: 8
- **US2**: 6
- **US3**: 10
- **Rename convergence**: 7
- **Cleanup/final proof**: 8
- **Suggested MVP**: T001-T021

Every task uses the required checklist format and names a concrete file or directory.
