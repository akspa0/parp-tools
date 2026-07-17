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

- [ ] T022 [P] [US2] Add failing dependency and cleanup-plan tests in `wow-viewer/data-harvester/tests/v50/test_cleanup.py`

### Implementation for User Story 2

- [ ] T023 [US2] Implement dependency discovery across manifests, checkpoints, reports, and known output layouts in `wow-viewer/data-harvester/src/harvester/v50/dependencies.py`
- [ ] T024 [US2] Implement size inventory and deterministic cleanup-plan generation in `wow-viewer/data-harvester/src/harvester/v50/cleanup.py`
- [ ] T025 [US2] Add inventory-to-plan and dry-run subcommands in `wow-viewer/data-harvester/scripts/v50_cleanup_artifacts.py`
- [ ] T026 [US2] Classify old datasets, temporary client copies, model outputs, cloud downloads, caches, root Python files, and continuity debt in `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`
- [ ] T027 [US2] Run fixture-only cleanup planning tests and a local read-only dry run, then record candidate bytes without approving deletion in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md`

**Checkpoint**: Every deletion candidate has replacement/dependency proof; apply remains unavailable.

---

## Phase 5: User Story 3 — Complete V50 Dataset Baseline (Priority: P3)

**Goal**: Build complete canonical per-build v50 stores from verified V18 signals and fresh client
extraction, with curricula represented as manifests.

**Independent Test**: A fixture V18 store with passing, failing, missing, and known-defective signals
produces a complete v50 fixture store whose copied chunks hash-identically, rejected signals are
fresh/unavailable, all rows have lineage, and curricula contain no array payloads.

### Tests for User Story 3

- [ ] T028 [P] [US3] Add failing selective-migration and resumability tests in `wow-viewer/data-harvester/tests/v50/test_migrate.py`
- [ ] T029 [P] [US3] Add failing complete-store/finalization and curriculum-manifest tests in `wow-viewer/data-harvester/tests/v50/test_store.py`

### Implementation for User Story 3

- [ ] T030 [US3] Implement the complete per-build v50 Zarr writer and finalization checks in `wow-viewer/data-harvester/src/harvester/v50/store.py`
- [ ] T031 [US3] Implement bit-preserving verified V18 copy, fresh-signal slots, row lineage, and resume ledger in `wow-viewer/data-harvester/src/harvester/v50/migrate.py`
- [ ] T032 [US3] Integrate the existing C# harvester stream for fresh signals and new builds in `wow-viewer/data-harvester/src/harvester/v50/build.py`
- [ ] T033 [US3] Implement immutable row-selection curricula over canonical stores in `wow-viewer/data-harvester/src/harvester/v50/curriculum.py`
- [ ] T034 [US3] Replace the current thin wrapper with migrate-v18, build, verify, finalize, and curriculum commands in `wow-viewer/data-harvester/scripts/v50_build_dataset.py`
- [ ] T035 [US3] Run fixture-only migration/store tests and document exact results in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md`
- [ ] T036 [US3] Prepare the bounded sampled V18 verification command for one user-selected build in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md`
- [ ] T037 [US3] After user review of sampled proof, prepare the full user-run migration and fresh-build commands with duration and output estimates in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md`

**Checkpoint**: The user-run v50 release is complete and fully verified before old datasets become deletable.

---

## Phase 6: Canonical V50 Rename and Ownership Convergence

**Purpose**: Finish the rename by moving implementation authority out of historical spec modules.

- [ ] T038 [P] Add compatibility/import and cross-release rejection tests in `wow-viewer/data-harvester/tests/v50/test_command_compatibility.py`
- [ ] T039 Move WDL-prior command ownership into `wow-viewer/data-harvester/src/harvester/v50/` and keep bounded shims in historical modules only where tests require them
- [ ] T040 Move terrain-refiner command ownership into `wow-viewer/data-harvester/src/harvester/v50/` and keep bounded shims in historical modules only where tests require them
- [ ] T041 Update `wow-viewer/data-harvester/scripts/v50_train_wdl_prior.py`, `v50_generate_wdl_priors.py`, `v50_review_wdl_prior.py`, and `v50_visualize_wdl_prior.py` to import only canonical v50 owners
- [ ] T042 Update `wow-viewer/data-harvester/scripts/v50_train_terrain.py` and `v50_infer_terrain.py` to import only canonical v50 owners
- [ ] T043 Remove obsolete wrappers or aliases only after a full caller search and record each disposition in `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`
- [ ] T044 Run focused v50 contract, compatibility, dataset, and command tests from `wow-viewer/data-harvester/`

**Checkpoint**: V50 has one implementation owner per workflow; historical names are explicit compatibility only.

---

## Phase 7: User-Reviewed Cleanup Apply and Final Proof

**Purpose**: Reclaim disk space only after the verified v50 replacement exists.

- [ ] T045 Add failing cleanup-apply identity, interrupted-run, and post-check tests in `wow-viewer/data-harvester/tests/v50/test_cleanup_apply.py`
- [ ] T046 Implement hash-confirmed cleanup apply and post-cleanup verification in `wow-viewer/data-harvester/src/harvester/v50/cleanup.py`
- [ ] T047 Expose cleanup apply only with explicit plan hash and protected-root checks in `wow-viewer/data-harvester/scripts/v50_cleanup_artifacts.py`
- [ ] T048 Generate the real cleanup dry-run report under `wow-viewer/output/reports/v50/v50.1/` and have the user review exact targets and expected recovered bytes
- [ ] T049 Prepare the exact user-run cleanup apply command in `wow-viewer/specs/109-v50-clean-room-audit/quickstart.md` without launching it
- [ ] T050 After the user runs cleanup, verify v50 store identities, protected survivors, removed targets, and recovered bytes in `wow-viewer/docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md`
- [ ] T051 Compress `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` after archiving stale chronology and preserving Spec 109 truth
- [ ] T052 Run the full lightweight v50 test suite and documentation consistency check from `wow-viewer/data-harvester/`

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
