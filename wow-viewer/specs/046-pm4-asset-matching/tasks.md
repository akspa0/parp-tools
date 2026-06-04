---
description: "Task list for spec 046 - PM4 asset matching and replacement placement automation"
---

# Tasks: 046 - PM4 Asset Matching

**Input**: Design documents from `/specs/046-pm4-asset-matching/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/pm4-asset-match-report.schema.json`

**Tests**: Focused PM4 library tests, inspect-tool validation, and bounded real-data matching checks are required because export determinism, candidate ranking, and placement synthesis are easy to regress.

**Organization**: Tasks are grouped by user story so the first signed-off slice can replace the freeze-prone export path before moving on to automated ranking and placement synthesis.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the feature directories and bounded PM4 matching surfaces.

- [ ] T001 Create `wow-viewer/src/core/WowViewer.Core.PM4/Matching/` for shared segmentation, scoring, and placement synthesis types.
- [ ] T002 Create `wow-viewer/data-harvester/src/harvester/pm4_asset_matching/` and `wow-viewer/data-harvester/scripts/` entries for Zarr-backed signal corpus tooling.
- [ ] T003 [P] Create focused PM4 matching test files under `wow-viewer/tests/WowViewer.Core.PM4.Tests/`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Define the shared data contracts and deterministic segmentation primitives before export, matching, or placement work begins.

**⚠️ CRITICAL**: No user story work should begin until segment identity, signal records, and run-report contracts are defined and testable.

- [ ] T004 [P] Implement `Pm4ObjectSegment`, `Pm4SegmentSignalRecord`, `Pm4AssetMatchCandidate`, and `Pm4ReplacementPlacementProposal` in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/`.
- [ ] T005 [P] Implement `Pm4ObjectSegmentBuilder` in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/` with deterministic segment-id generation and ambiguity flags.
- [ ] T006 [P] Implement `Pm4SegmentSignalExtractor` in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/` for comparable PM4 segment signal derivation.
- [ ] T007 [P] Add `Pm4ObjectSegmentBuilderTests.cs` and `Pm4SegmentSignalExtractorTests.cs` in `wow-viewer/tests/WowViewer.Core.PM4.Tests/`.
- [ ] T008 [P] Implement shared inspect/report models in `wow-viewer/src/tools-shared/WowViewer.Tools.Shared/Pm4Matching/` matching `contracts/pm4-asset-match-report.schema.json`.

**Checkpoint**: Segment identity and signal contracts are stable; export and corpus tooling can now be built on top.

---

## Phase 3: User Story 1 - Export PM4 Object Segments Without Freezing (Priority: P1) 🎯 MVP

**Goal**: Replace the freeze-prone PM4 object export path with a deterministic automation workflow that emits PM4 object segments and signal corpora.

**Independent Test**: Export a bounded PM4 tile set and a directory-scale PM4 corpus through the new automation path and verify a structured output is produced without relying on the old blocking viewer flow.

### Tests for User Story 1

- [ ] T009 [P] [US1] Add inspect-command tests or focused report-shape tests in `wow-viewer/tests/WowViewer.Core.PM4.Tests/` covering PM4 segment export output.
- [ ] T010 [P] [US1] Add a bounded directory-scale regression test for cross-tile segment export stability in `wow-viewer/tests/WowViewer.Core.PM4.Tests/`.

### Implementation for User Story 1

- [ ] T011 [US1] Add `pm4 export-segments` command handling to `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`.
- [ ] T012 [US1] Implement an export job/service in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/` that streams PM4 segment records and signal payloads without viewer-shell ownership.
- [ ] T013 [US1] Add Zarr-backed PM4 segment corpus writing in `wow-viewer/data-harvester/src/harvester/pm4_asset_matching/pm4_signal_store.py`.
- [ ] T014 [US1] Add `export_pm4_segment_signals.py` in `wow-viewer/data-harvester/scripts/` for bounded corpus-building automation.
- [ ] T015 [US1] Update the viewer PM4 export surface so `Export PM4 Obj Set` becomes a thin trigger/report surface or is explicitly redirected to the new automation owner instead of running the old blocking path.

**Checkpoint**: Researchers can export PM4 object segments and signal stores without depending on the old freezing interaction.

---

## Phase 4: User Story 2 - Rank Real WMO/M2 Matches Automatically (Priority: P1)

**Goal**: Build the automated candidate-ranking lane from PM4 segment corpora to staged WMO/M2 asset reference corpora.

**Independent Test**: Run the matcher on a validation set with known placed assets and verify every eligible PM4 segment receives a ranked candidate list plus scoring rationale.

### Tests for User Story 2

- [ ] T016 [P] [US2] Add `Pm4AssetMatchScorerTests.cs` in `wow-viewer/tests/WowViewer.Core.PM4.Tests/` covering deterministic score ordering and unresolved/ambiguous states.
- [ ] T017 [P] [US2] Add Python-side smoke validation for Zarr asset-signal corpus compatibility in `wow-viewer/data-harvester/`.

### Implementation for User Story 2

- [ ] T018 [P] [US2] Implement staged-asset signal extraction in `wow-viewer/data-harvester/src/harvester/pm4_asset_matching/asset_signal_corpus.py`.
- [ ] T019 [P] [US2] Add `build_pm4_asset_signal_corpus.py` in `wow-viewer/data-harvester/scripts/`.
- [ ] T020 [US2] Implement `Pm4AssetMatchScorer` in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/` with deterministic score breakdown output.
- [ ] T021 [US2] Implement a match-report writer in `wow-viewer/src/tools-shared/WowViewer.Tools.Shared/Pm4Matching/`.
- [ ] T022 [US2] Add `pm4 match-assets` command handling to `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs` to emit ranked candidate reports from PM4 and asset signal corpora.

**Checkpoint**: Automated ranked WMO/M2 candidate lists replace the old manual matching workflow for validation-scale runs.

---

## Phase 5: User Story 3 - Generate Replacement Placement Proposals For Missing Tiles (Priority: P2)

**Goal**: Turn accepted or top-ranked candidate matches into proposal-grade replacement placements for missing development tiles.

**Independent Test**: Run placement synthesis on a bounded missing-tile target set and verify machine-readable placement proposals are emitted with PM4 and candidate provenance.

### Tests for User Story 3

- [ ] T023 [P] [US3] Add `Pm4ReplacementPlacementSynthesizerTests.cs` in `wow-viewer/tests/WowViewer.Core.PM4.Tests/` covering confidence flags, unresolved cases, and provenance retention.
- [ ] T024 [P] [US3] Add a bounded known-tile validation case that compares synthesized placement output against a tile with existing placements.

### Implementation for User Story 3

- [ ] T025 [US3] Implement `Pm4ReplacementPlacementSynthesizer` in `wow-viewer/src/core/WowViewer.Core.PM4/Matching/`.
- [ ] T026 [US3] Add `pm4 synthesize-placements` command handling to `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`.
- [ ] T027 [US3] Emit proposal-grade replacement placement manifests and summary reports via `wow-viewer/src/tools-shared/WowViewer.Tools.Shared/Pm4Matching/`.
- [ ] T028 [US3] Add bounded target-tile filtering and review-required confidence thresholds to the synthesis workflow.

**Checkpoint**: Missing-tile replacement placement proposals can be produced automatically from PM4 and candidate-match evidence.

---

## Phase 6: User Story 4 - Review Automation Results Without Returning To Broken Manual Tools (Priority: P3)

**Goal**: Provide a bounded review surface for the exported segments, ranked candidates, and replacement placements.

**Independent Test**: Inspect one emitted match report and placement proposal set without re-running the old manual PM4 matching workflow.

### Implementation for User Story 4

- [ ] T029 [P] [US4] Extend inspect-tool human-readable report printing for segment export, candidate match, and placement synthesis outputs in `wow-viewer/tools/inspect/WowViewer.Tool.Inspect/Program.cs`.
- [ ] T030 [US4] Add a bounded viewer-side review/import surface only if it consumes the emitted automation report format rather than reintroducing the old manual matching owner.
- [ ] T031 [US4] Document the automation-first review flow in `wow-viewer/docs/WoWViewer/README.md` or the relevant PM4 workflow note.

**Checkpoint**: Researchers can review automation output without falling back to the broken manual PM4 matcher.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validation, docs, and continuity sync.

- [ ] T032 [P] Update `wow-viewer/docs/research/004-pm4-format-research/spec.md` with any new bounded evidence contracts that the export/matching lane makes authoritative.
- [ ] T033 [P] Update `wow-viewer/docs/architecture/pm4-region-aware-object-grouping-2026-05-21.md` or successor notes if the automation lane changes the accepted stitch/segmentation owner.
- [ ] T034 [P] Add schema validation for emitted match reports against `specs/046-pm4-asset-matching/contracts/pm4-asset-match-report.schema.json`.
- [ ] T035 Run focused PM4 tests, inspect-tool builds, and bounded real-data validation commands from `quickstart.md`.
- [ ] T036 Update `gillijimproject_refactor/memory-bank/activeContext.md` and `progress.md` with the landed owner, proof surfaces, and remaining gaps.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on setup completion and blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on foundational segment/signal contracts.
- **User Story 2 (Phase 4)**: Depends on PM4 segment export and corpus-writing from User Story 1.
- **User Story 3 (Phase 5)**: Depends on match-report outputs from User Story 2.
- **User Story 4 (Phase 6)**: Depends on report/proposal outputs from earlier stories.
- **Polish (Phase 7)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: MVP; first independently useful slice.
- **User Story 2 (P1)**: Builds directly on exported segment corpora.
- **User Story 3 (P2)**: Builds on ranked candidate reports.
- **User Story 4 (P3)**: Review-only layer after automation outputs exist.

### Parallel Opportunities

- T004, T005, T006, and T008 can run in parallel across separate matching contract files.
- T007 can run in parallel once the contract files exist.
- T018 and T019 can run in parallel as separate asset-corpus tasks.
- T023 and T024 can run in parallel as placement-synthesis validation tasks.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational segment/signal contracts
3. Complete Phase 3: Non-freezing PM4 segment export
4. **STOP and VALIDATE**: Export a real PM4 corpus slice and confirm the old blocking workflow is no longer required

### Incremental Delivery

1. Deliver US1 to replace the freeze-prone export owner
2. Add US2 automated ranking to eliminate broken manual matching as the primary workflow
3. Add US3 placement synthesis for missing-tile reconstruction
4. Add US4 bounded review/report surfaces

### Parallel Team Strategy

With multiple developers:

1. One developer owns PM4 segment contracts/export
2. One developer owns asset reference corpus generation
3. One developer owns scorer and report output
4. One developer owns placement synthesis and review/report polish

---

## Notes

- The first slice should not quietly drift back into a viewer-owned manual matching workflow.
- Zarr is the durable signal-store owner for both PM4 and asset reference corpora in this plan.
- Placement synthesis is proposal-grade first by design; map mutation is explicitly deferred.
