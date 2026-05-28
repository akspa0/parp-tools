# Tasks: V18 Dataset Canonical Contract

**Input**: Design documents from [`wow-viewer/specs/001-v18-dataset-spec/`](wow-viewer/specs/001-v18-dataset-spec/spec.md)

**Prerequisites**: [`plan.md`](wow-viewer/specs/001-v18-dataset-spec/plan.md), [`spec.md`](wow-viewer/specs/001-v18-dataset-spec/spec.md)

**Tests**: Real-data validation tasks are included because the spec and constitution require reproducible staged-client proof for dataset claims.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g. [`US1`](wow-viewer/specs/001-v18-dataset-spec/spec.md:67), [`US2`](wow-viewer/specs/001-v18-dataset-spec/spec.md:101), [`US3`](wow-viewer/specs/001-v18-dataset-spec/spec.md:132))
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the versioned V18 workflow surface without changing the current V16 path.

- [ ] T001 Create [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1) by copying forward the current [`wow-viewer/data-harvester/scripts/build_v16_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1) baseline
- [ ] T002 Update dataset root constants, usage text, command banners, and help output in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:56) so the script writes to [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728)
- [ ] T003 [P] Add initial V18 operator-path notes to [`wow-viewer/data-harvester/README.md`](wow-viewer/data-harvester/README.md:1) referencing the canonical V18 builder instead of a V16-plus-patch workflow

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Lock the canonical V18 contract surface before story work begins.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [ ] T004 Freeze the promoted V18 signal inventory and output-array mapping in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:63)
- [ ] T005 Define V18 finalized-status helpers, mandatory artifact lists, and output-root helpers in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:53)
- [ ] T006 Carry forward decoded-metadata write and validation helpers into [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1358)
- [ ] T007 Add V18 resume, validate, and merge command scaffolding in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:2629)

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Promote the Current V16 Build Flow into Canonical V18 (Priority: P1) 🎯 MVP

**Goal**: Deliver a working V18 builder that behaves like the current V16 workflow but writes a separate V18 store and enforces a publishable finalized-state contract.

**Independent Test**: Run one bounded staged-client V18 build and confirm it creates a V18 store under [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728) with no required follow-up patch command to call the build complete.

- [ ] T008 [US1] Implement the canonical `build` command flow in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1955) using the copied V16 streaming harvester path
- [ ] T009 [US1] Wire staged-client discovery, harvest-tool invocation, and V18 output-root ownership in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:57)
- [ ] T010 [US1] Implement V18 strict finalized-status gating and mandatory artifact checks in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:2027)
- [ ] T011 [US1] Implement V18 `stats` and `validate` command behavior in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:2596)
- [ ] T012 [US1] Run a bounded staged-client V18 build and inspect outputs under [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728) to prove the versioned builder works independently

**Checkpoint**: User Story 1 should now produce a bounded V18 store on its own.

---

## Phase 4: User Story 2 - Promote Metadata and Patched Signals into First-Class V18 Outputs (Priority: P1)

**Goal**: Eliminate the required V16 patch-after-build behavior by folding decoded metadata and promoted signal families into the canonical V18 build.

**Independent Test**: Run one bounded V18 build and confirm [`decoded_metadata.parquet`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1363), [`signal_validation.json`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1346), promoted signal arrays, and coverage flags are present directly in the V18 store.

- [ ] T013 [US2] Port canonical decoded-metadata writing and parity validation into the main V18 build in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1877)
- [ ] T014 [US2] Promote the current required patch-on signal families into first-class V18 arrays and coverage fields in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1315)
- [ ] T015 [US2] Fold the renderer-truth patch flow from [`wow-viewer/data-harvester/scripts/patch_v16_renderer_truth.py`](wow-viewer/data-harvester/scripts/patch_v16_renderer_truth.py:151) into V18 build finalization in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:3414)
- [ ] T016 [US2] Integrate the current object-roof mask promotion path from [`wow-viewer/data-harvester/scripts/patch_v18_object_roof_masks.py`](wow-viewer/data-harvester/scripts/patch_v18_object_roof_masks.py:169) into [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:3414)
- [ ] T017 [P] [US2] Update canonical V18 dataset loading to prefer V18-built promoted signals in [`wow-viewer/data-harvester/src/harvester/v18_dataset.py`](wow-viewer/specs/024-v18-canvas-paste-refinement-layer/spec.md:24)
- [ ] T018 [P] [US2] Update compatibility loaders to read promoted V18 signals without legacy patch assumptions in [`wow-viewer/data-harvester/src/harvester/v16_2_dataset.py`](wow-viewer/data-harvester/src/harvester/v16_2_dataset.py:6)
- [ ] T019 [US2] Run bounded V18 validation and verify canonical metadata and promoted-signal outputs under [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728)

**Checkpoint**: User Stories 1 and 2 should now work without a mandatory patch phase.

---

## Phase 5: User Story 3 - Preserve Raw Blob Expansion Path Without Breaking Current Consumers (Priority: P2)

**Goal**: Keep raw-blob preservation additive and optional while the decoded V18 contract remains the mandatory compatibility surface.

**Independent Test**: Build one bounded V18 store with raw-blob preservation disabled and one with it enabled, then verify the decoded contract remains stable in both cases.

- [ ] T020 [US3] Add optional raw-blob enablement flags and default-disabled behavior in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:1955)
- [ ] T021 [US3] Implement additive raw-blob sidecar manifest and payload layout under [`wow-viewer/output/datasets/v18/<build>.zarr/raw_blobs/`](wow-viewer/docs/architecture/v18-undecoded-blob-datastore-sketch-2026-05-27.md:44) from [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/data-harvester/scripts/build_v16_dataset.py:2436)
- [ ] T022 [US3] Enforce additive-only raw-blob validation rules in [`wow-viewer/data-harvester/scripts/build_v18_dataset.py`](wow-viewer/docs/architecture/v18-undecoded-blob-datastore-sketch-2026-05-27.md:92)
- [ ] T023 [US3] Run bounded additive-sidecar proof against [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728) and verify decoded-contract artifact counts remain stable

**Checkpoint**: All user stories should now be independently functional.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Finalize operator docs, continuity, and proof notes across the feature.

- [ ] T024 [P] Update canonical operator documentation in [`wow-viewer/data-harvester/README.md`](wow-viewer/data-harvester/README.md:98) and [`wow-viewer/README.md`](wow-viewer/README.md:13) for the V18 build path
- [ ] T025 [P] Update continuity summaries in [`wow-viewer/memory-bank/activeContext.md`](wow-viewer/memory-bank/activeContext.md:6) and [`wow-viewer/memory-bank/progress.md`](wow-viewer/memory-bank/progress.md:3) after implementation proof lands
- [ ] T026 Record final validation evidence, bounded build commands, and known launch limitations in [`wow-viewer/specs/001-v18-dataset-spec/plan.md`](wow-viewer/specs/001-v18-dataset-spec/plan.md)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup** — no dependencies, can start immediately
- **Phase 2: Foundational** — depends on Phase 1 and blocks all user stories
- **Phase 3: US1** — depends on Phase 2 and is the MVP build lane
- **Phase 4: US2** — depends on US1 because promoted signals are folded into the V18 builder
- **Phase 5: US3** — depends on US2 because additive raw-blob behavior must preserve the already-proven canonical V18 contract
- **Phase 6: Polish** — depends on whichever user stories are in scope for the current stopping point

### User Story Dependencies

- **US1**: no dependency on other stories once the foundation is complete
- **US2**: depends on US1 because it extends the canonical V18 builder rather than a separate patch lane
- **US3**: depends on US2 because raw-blob preservation must remain additive to the canonical decoded V18 contract

### Within Each User Story

- Builder ownership before consumer alignment
- Canonical metadata and promoted signals before merge/finalization polish
- Real-data proof before calling a story done

### Parallel Opportunities

- [`T003`](wow-viewer/specs/001-v18-dataset-spec/tasks.md) can run in parallel with [`T001`](wow-viewer/specs/001-v18-dataset-spec/tasks.md) after the V18 script path is chosen
- [`T017`](wow-viewer/specs/001-v18-dataset-spec/tasks.md) and [`T018`](wow-viewer/specs/001-v18-dataset-spec/tasks.md) can run in parallel after the promoted-signal contract is frozen in [`T014`](wow-viewer/specs/001-v18-dataset-spec/tasks.md)
- [`T024`](wow-viewer/specs/001-v18-dataset-spec/tasks.md) and [`T025`](wow-viewer/specs/001-v18-dataset-spec/tasks.md) can run in parallel during polish

---

## Parallel Example: User Story 2

```text
Task: "Update canonical V18 dataset loading to prefer V18-built promoted signals in wow-viewer/data-harvester/src/harvester/v18_dataset.py"
Task: "Update compatibility loaders to read promoted V18 signals without legacy patch assumptions in wow-viewer/data-harvester/src/harvester/v16_2_dataset.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: US1
4. Stop and validate one bounded V18 build under [`wow-viewer/output/datasets/v18/`](wow-viewer/README.md:728)

### Recommended Incremental Delivery

1. Land the versioned V18 builder surface
2. Fold decoded metadata and promoted signals into the canonical build
3. Re-prove resume / validate / merge behavior
4. Add optional additive raw-blob sidecars last

### Suggested MVP Scope

The minimum useful implementation is **US1 plus the decoded-metadata portion of US2** so the team gets a real V18 build root quickly without overcommitting to every optional promoted signal on the first pass.

---

## Notes

- Tasks intentionally keep all new work under [`wow-viewer/`](wow-viewer/README.md:1)
- The V18 builder is a versioned workflow fork, not a reader rewrite
- Real-data proof must always cite staged-client roots under [`output/tmp/wowarchive-clients/`](AGENTS.md:116)
- A future V20 dataset spec may replace the current intermediate raw-array /
  NPZ-shaped interchange with a direct parser → decoded → dataset pipeline, but
  that redesign is out of scope for these V18 tasks
