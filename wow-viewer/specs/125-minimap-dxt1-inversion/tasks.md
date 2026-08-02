# Tasks: Minimap DXT1 Artifact Inversion

**Input**: Design documents from `/specs/125-minimap-dxt1-inversion/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are included where the spec mandates measurable gates (FR-014 round-trip, FR-016 baseline, FR-008..FR-012 restoration).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create `Dxt1TileCodec.cs` in `wow-viewer/src/core/WowViewer.Core.IO/Blp/` with the encode/decode cycle skeleton (FR-002)
- [ ] T002 Create `MinimapLightingBaseline.cs` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/` with the survey skeleton (FR-016)
- [ ] T003 Create `MinimapEncodingSurvey.cs` in `wow-viewer/src/core/WowViewer.Core.IO/Maps/` with the survey skeleton (FR-013)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Implement `Dxt1TileCodec.EncodeDecode` using `BCnEncoder` (`CompressionFormat.Bc1`) in `wow-viewer/src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs` (FR-002)
- [ ] T005 Implement `Dxt1TileCodec.DecodeAuthored` wrapping `BlpRgbReader` in `wow-viewer/src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs` (FR-001)
- [ ] T006 Implement `Dxt1TileCodec.RoundTripAgreement` (decode → re-encode → block-level byte agreement) in `wow-viewer/src/core/WowViewer.Core.IO/Blp/Dxt1TileCodec.cs` (FR-014)
- [ ] T007 Implement `MinimapLightingBaseline.Survey` (cross-tile vs within-tile luma variance) in `wow-viewer/src/core/WowViewer.Core.IO/Maps/MinimapLightingBaseline.cs` (FR-016)
- [ ] T008 Implement `MinimapLightingBaseline.NormalizeToBaseline` (mean/std luma rescale) in `wow-viewer/src/core/WowViewer.Core.IO/Maps/MinimapLightingBaseline.cs` (FR-016)
- [ ] T009 Implement `MinimapEncodingSurvey.Survey` (per-file encoding detection + aggregation) in `wow-viewer/src/core/WowViewer.Core.IO/Maps/MinimapEncodingSurvey.cs` (FR-001, FR-013)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Compare synthetic against authored on equal terms (Priority: P1) 🎯 MVP

**Goal**: A researcher comparing a synthesized tile against its authored counterpart gets a score that reflects terrain reconstruction, not codec damage. The synthesizer emits a DXT1 parity companion, and comparison reports parity-adjusted agreement.

**Independent Test**: Score authored tiles against synthetic renders with and without encoding parity; confirm the parity run reports a materially different (and better) agreement while the relative ranking of two deliberately different render settings is preserved.

### Tests for User Story 1

- [ ] T010 [P] [US1] Unit test `Dxt1TileCodecTests` round-trip agreement ≥95% on authored tiles in `wow-viewer/tests/WowViewer.Core.Tests/Dxt1TileCodecTests.cs` (FR-014, SC-009)
- [ ] T011 [P] [US1] Unit test parity cycle produces a measurably blockier image than pristine input in `wow-viewer/tests/WowViewer.Core.Tests/Dxt1TileCodecTests.cs` (FR-002)

### Implementation for User Story 1

- [ ] T012 [US1] Add `--dxt1-parity` flag parsing in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-015)
- [ ] T013 [US1] Emit `*_dxt1.png` parity companion per tile via `Dxt1TileCodec.EncodeDecode` in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-015)
- [ ] T014 [US1] Extend `SyntheticMinimapScorecard` to report parity-adjusted agreement alongside unadjusted, stating the encoding applied in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-003)
- [ ] T015 [US1] Exclude degenerate (single flat colour) tiles from aggregate scores and report them as excluded in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-004)
- [ ] T016 [US1] Record parity status (including "none") on every comparison row in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-005)
- [ ] T017 [US1] Wire era-gating: unrecognised build flagged, never silently defaulted, in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-006)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Generate a corpus free of codec confound (Priority: P2)

**Goal**: Someone building a training corpus that mixes authored and synthesized tiles can produce both sides with matching encoding characteristics, so a model cannot separate them by compression damage.

**Independent Test**: Build a small mixed corpus with parity enabled, train a trivial classifier to predict source; confirm near-chance performance, and well-above-chance on a non-parity corpus.

### Implementation for User Story 2

- [ ] T018 [P] [US2] Add `--encoding-survey` flag and report per-build/map encoding distribution in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-013)
- [ ] T019 [US2] Record encoding parity status on every generated corpus row (including "none") in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-005)
- [ ] T020 [US2] Ensure the parity companion is available to corpus generation so authored and synthetic rows carry matching encoding characteristics in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-015)

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Restore an authored tile toward its pre-compression appearance (Priority: P3)

**Goal**: A restorer takes an authored minimap tile and recovers an image closer to what the client's renderer produced before DXT1 quantised it — block banding reduced, seams removed, colour bias corrected.

**Independent Test**: Hold out pristine renders never seen in training, encode them, run restoration, and measure recovery against the known pre-encoding originals.

### Implementation for User Story 3

- [ ] T021 [P] [US3] Create residual restoration model definition in `wow-viewer/data-harvester/src/harvester/dxt1_restore.py` (FR-007)
- [ ] T022 [US3] Implement training script generating pristine→encoded pairs locally in `wow-viewer/data-harvester/scripts/train_v20_dxt1_restore.py` (FR-007)
- [ ] T023 [US3] Implement `restore` and `verdict` (improvement, re-encode agreement, hallucination fraction) in `wow-viewer/data-harvester/src/harvester/dxt1_restore.py` (FR-008, FR-009, FR-010)
- [ ] T024 [US3] Implement the hallucination gate (unsupported-detail fraction) and block promotion without it meeting the stated gate in `wow-viewer/data-harvester/src/harvester/dxt1_restore.py` (FR-010)
- [ ] T025 [US3] Ensure undamaged input is returned substantially unchanged (<2% colour error) in `wow-viewer/data-harvester/src/harvester/dxt1_restore.py` (FR-011, SC-006)
- [ ] T026 [US3] Separate native-resolution restoration from any resolution increase; do not share a metric in `wow-viewer/data-harvester/src/harvester/dxt1_restore.py` (FR-012)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Decode terrain shadow and reconstruct terrain directly (Priority: P3)

**Goal**: A reconstruction engineer takes any authored minimap tile and recovers the terrain that
produced it — minimap RGB → heightmap → 3D mesh with a single model that reads the terrain shadow and
converts it into ridges, mountains, and terrain detail. This is the strategic payoff: because we now
know how the minimap terrain shadow is created, the shadow in an authored tile is a readable
terrain-shape signal.

**Independent Test**: Hold out authored tiles never seen in training, decode their terrain shadow,
run the reconstruction, and measure the recovered heightmap against the known ground-truth heightmap
(MCVT) for those tiles.

### Implementation for User Story 4

- [ ] T027 [P] [US4] Implement terrain-shadow decoder using the synthesizer's lighting model in `wow-viewer/data-harvester/src/harvester/terrain_shadow_decode.py` (FR-017)
- [ ] T028 [US4] Implement the minimap-RGB → heightmap → 3D-mesh reconstruction model in `wow-viewer/data-harvester/src/harvester/terrain_reconstruct.py` (FR-018)
- [ ] T029 [US4] Implement training script for the reconstruction model in `wow-viewer/data-harvester/scripts/train_v21_terrain_reconstruct.py` (FR-018)
- [ ] T030 [US4] Implement evaluation against ground-truth MCVT heightmap (relief correlation + mesh plausibility) in `wow-viewer/data-harvester/src/harvester/terrain_reconstruct.py` (FR-019)
- [ ] T031 [US4] Implement low-confidence reporting on ambiguous (flat, no-relief) tiles in `wow-viewer/data-harvester/src/harvester/terrain_reconstruct.py` (FR-020)

**Checkpoint**: All user stories should now be independently functional

---

## Phase 7: User Story 5 - Super-resolve terrain and texturing data (Priority: P3)

**Goal**: A reconstruction engineer upscales terrain and texturing data using a super-resolution
model trained on real low-res/high-res pairs produced by the synthesizer (same terrain, matching
lighting, no objects).

**Independent Test**: Hold out high-res renders never seen in training, downscale them to low-res,
run the super-resolution model, and measure recovery against the known high-res originals.

### Implementation for User Story 5

- [ ] T032 [P] [US5] Implement the super-resolution model in `wow-viewer/data-harvester/src/harvester/super_resolve.py` (FR-021)
- [ ] T033 [US5] Implement training script for the super-resolution model in `wow-viewer/data-harvester/scripts/train_v22_super_resolve.py` (FR-021)
- [ ] T034 [US5] Implement evaluation against known high-res originals, reported separately from artifact-removal and reconstruction metrics in `wow-viewer/data-harvester/src/harvester/super_resolve.py` (FR-021, FR-012)

**Checkpoint**: All user stories should now be independently functional

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T035 [P] Add `--lighting-baseline` flag and report per-map baseline, accounting for it when scoring in `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs` (FR-016)
- [ ] T036 [P] Unit test `MinimapLightingBaselineTests` (baseline detected, independent exposures not, normalisation works) in `wow-viewer/tests/WowViewer.Core.Tests/MinimapLightingBaselineTests.cs` (FR-016)
- [ ] T037 Run quickstart.md validation (build + synthetic-minimap parity + score)
- [ ] T038 Update memory bank `activeContext.md` and `progress.md` with Spec 125 state

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Depends on US1's parity mechanism (FR-015)
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Depends on US1's encoder for training pairs (FR-007)

### Within Each User Story

- Tests (where included) MUST be written and FAIL before implementation
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: "Unit test Dxt1TileCodecTests round-trip agreement in tests/WowViewer.Core.Tests/Dxt1TileCodecTests.cs"
Task: "Unit test parity cycle blockiness in tests/WowViewer.Core.Tests/Dxt1TileCodecTests.cs"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
