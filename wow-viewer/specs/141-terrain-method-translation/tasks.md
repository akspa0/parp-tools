# Tasks: Terrain Method Translation and Evidence Gates

**Input**: Design documents from `/specs/141-terrain-method-translation/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/method-translation.schema.md`, and `quickstart.md`.

## Phase 1: Setup

**Purpose**: Establish the v60 method-translation identity without changing the existing model or reader contracts.

- [ ] T001 Record the Spec 141 feature identity and artifact roots in `wow-viewer/.specify/feature.json` and `wow-viewer/specs/141-terrain-method-translation/`.
- [ ] T002 [P] Add method-translation schema constants and exports in `wow-viewer/data-harvester/src/harvester/v60/__init__.py`.
- [ ] T003 [P] Add the initial method source fixtures in `wow-viewer/data-harvester/tests/v60/fixtures/terrain_method_translation_methods.json`.

## Phase 2: Foundational Contracts

**Purpose**: Define the shared entities and fail-closed signal boundary before any benchmark work.

- [ ] T004 Implement `ExternalMethodRecord`, `InputContract`, and `TranslationDecision` validation in `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py`.
- [ ] T005 Implement input-modality and forbidden-read validation for `rgb_only`, `height_prior`, `point_cloud`, and `combined` in `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py`.
- [ ] T006 [P] Add accepted, diagnostic, combined, and forbidden contract fixtures in `wow-viewer/data-harvester/tests/v60/test_terrain_method_translation.py`.
- [ ] T007 [P] Add manifest contract assertions for `v60-terrain-method-translation-v1` in `wow-viewer/data-harvester/tests/v60/test_terrain_method_translation_contract.py`.

**Checkpoint**: The modality contract rejects target-derived deployment inputs and can classify a valid offline DSM/point-cloud diagnostic.

## Phase 3: User Story 1 - Method Evidence Ledger (Priority: P1)

**Goal**: Make the initial external-method research reproducible and reviewable.

**Independent Test**: The ledger dry-run validates the six initial method families and emits complete provenance, modality, domain-gap, weights, and translation fields.

- [ ] T008 [P] [US1] Encode DSM2DTM, ResDepth, SMRF, CSF, aerial object-mask, and Prithvi records in `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py` using the links and decisions in `specs/141-terrain-method-translation/research.md`.
- [ ] T009 [US1] Implement deterministic ledger validation and status reporting in `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py`.
- [ ] T010 [US1] Add the PowerShell-ready ledger audit CLI in `wow-viewer/data-harvester/scripts/v60_audit_terrain_methods.py`.
- [ ] T011 [P] [US1] Add ledger completeness and unknown-license tests in `wow-viewer/data-harvester/tests/v60/test_terrain_method_ledger.py`.

**Checkpoint**: Every initial method is explicitly reference-only, diagnostic, candidate, held, rejected, or promoted.

## Phase 4: User Story 2 - Modality and Provenance Boundary (Priority: P1)

**Goal**: Prevent a method from receiving a signal unavailable to the claimed WoW runtime.

**Independent Test**: Representative manifests classify correctly and a forbidden deployment manifest fails with the exact signal name.

- [ ] T012 [US2] Implement per-run input-read recording and forbidden-read audit output in `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py`.
- [ ] T013 [US2] Add predicted-mask versus supervision-mask provenance fields to `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py`.
- [ ] T014 [P] [US2] Add failure tests for `height_257`, `terrain_shadow_256`, `shadow_mask`, WDL, and target-side object-mask reads in `wow-viewer/data-harvester/tests/v60/test_terrain_method_forbidden_reads.py`.
- [ ] T015 [US2] Add deterministic contract audit output and fresh-output refusal to `wow-viewer/data-harvester/scripts/v60_audit_terrain_methods.py`.

**Checkpoint**: A result cannot claim RGB-only deployment compatibility when it reads DSM, point-cloud, target, or source-side supervision data.

## Phase 5: User Story 3 - RGB-Only Object-Aware Benchmark (Priority: P2)

**Goal**: Prepare a project-owned comparison of no-mask, predicted-mask, and withheld-mask RGB terrain completion.

**Independent Test**: A CPU dry-run emits deterministic splits, baselines, provenance, and independent metric conditions without serializing ground-truth object masks as inference inputs.

- [ ] T016 [US3] Implement benchmark-condition planning for no-mask, predicted-mask, and withheld-mask in `wow-viewer/data-harvester/src/harvester/v60/rgb_method_benchmark.py`.
- [ ] T017 [US3] Connect the benchmark planner to the existing object-library sieve and authored raw-RGB manifest contracts in `wow-viewer/data-harvester/src/harvester/v60/rgb_method_benchmark.py`.
- [ ] T018 [US3] Add identity, tile-mean, clean-height, contaminated-input, mask, family, and cross-tile metric declarations in `wow-viewer/data-harvester/src/harvester/v60/rgb_method_benchmark.py`.
- [ ] T019 [US3] Add the PowerShell-ready dry-run CLI in `wow-viewer/data-harvester/scripts/v60_build_rgb_method_benchmark.py`.
- [ ] T020 [P] [US3] Add benchmark planning and no-leak tests in `wow-viewer/data-harvester/tests/v60/test_rgb_method_benchmark.py`.
- [ ] T021 [US3] **USER RUNS** the dry plan, then any explicitly confirmed corpus/training/evaluation command after the contract tests pass; record the result in `specs/141-terrain-method-translation/quickstart.md`.

**Checkpoint**: No RGB-only branch is recommended for training until all three mask conditions and declared baselines are present in the dry report.

## Phase 6: User Story 4 - Research Leads and Translation Decisions (Priority: P3)

**Goal**: Preserve novel observations as falsifiable, provenance-bound research leads.

**Independent Test**: A lead remains unconfirmed without provenance and a falsification result, and a method decision links to an evidence report.

- [ ] T022 [US4] Implement `ResearchLead` validation and state transitions in `wow-viewer/data-harvester/src/harvester/v60/research_leads.py`.
- [ ] T023 [US4] Implement evidence-run-to-translation-decision binding in `wow-viewer/data-harvester/src/harvester/v60/terrain_method_translation.py`.
- [ ] T024 [P] [US4] Add research-lead provenance and promotion-gate tests in `wow-viewer/data-harvester/tests/v60/test_research_leads.py`.
- [ ] T025 [US4] Add a concise lead-recording section to `wow-viewer/specs/141-terrain-method-translation/quickstart.md`.

**Checkpoint**: New discoveries survive as explicit observations and hypotheses without being mistaken for historical or model truth.

## Phase 7: Polish and Cross-Cutting Validation

- [ ] T026 [P] Add CLI help and dry-run smoke tests in `wow-viewer/data-harvester/tests/v60/test_terrain_method_cli.py`.
- [ ] T027 [P] Update `wow-viewer/specs/141-terrain-method-translation/research.md` with any source-access or reproduction changes discovered during implementation.
- [ ] T028 Run focused `ruff`, `py_compile`, and v60 pytest checks; record exact commands and results in `wow-viewer/specs/141-terrain-method-translation/quickstart.md`.
- [ ] T029 Update `wow-viewer/memory-bank/activeContext.md`, `wow-viewer/memory-bank/progress.md`, and `wow-viewer/memory-bank/workstream-terrain-ml.md` after each completed implementation slice.
- [ ] T030 Commit each bounded implementation slice with only its intended files staged and report the commit hash.

## Dependencies and Execution Order

```text
T001-T007 foundational identity and contracts
    -> T008-T011 method ledger
    -> T012-T015 modality/provenance audit
    -> T016-T021 RGB-only benchmark preparation and user gate
    -> T022-T025 research leads and translation decisions
    -> T026-T030 polish, continuity, and commits
```

The optional DSM/point-cloud diagnostic is not a prerequisite for the RGB-only benchmark. It may be added only when an explicitly configured point-cloud or DSM source exists and must remain a separate evidence report.

## Parallel Opportunities

- T002, T003, T006, and T007 can proceed in parallel after the feature identity is recorded.
- T008 and T011 can proceed in parallel after the ledger entity contract exists.
- T014 and T015 can proceed in parallel after the forbidden-read fields are frozen.
- T020 can proceed in parallel with the benchmark planner implementation once its condition schema is fixed.
- T024 and T025 can proceed in parallel after the research-lead entity is defined.

## Implementation Strategy

### MVP First

1. Complete T001-T007.
2. Complete T008-T015 and validate the method ledger plus modality gate.
3. Stop and review the Gate 1 report before writing any RGB benchmark code.

### Incremental Delivery

1. Deliver the ledger and modality audit as the first independently useful slice.
2. Deliver the RGB-only benchmark plan and dry-run as the second slice.
3. Let the user run the explicit heavy gate only after the dry-run and forbidden-read tests pass.
4. Add research-lead persistence and translation decisions after benchmark evidence exists.

No external model weights, broad harvest, or GPU training is part of the Speckit planning slice itself.
