# Tasks: V16 Liquid Supervision Truth Repair

**Input**: Design documents from `wow-viewer/specs/002-v16-liquid-supervision-truth-repair/`

**Prerequisites**: plan.md (required), spec.md (required for user stories)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel
- **[Story]**: Which user story this task belongs to (US1-US4)

---

## Phase 1: Fail-Loud Raw Sampling (US2)

**Goal**: Raw sample commands stop pretending success when a requested liquid source is absent.

- [ ] T001 [US2] Update `inspect_v16_harvest_samples.py` so each requested kind records `sample_count_found`, not just output files.
- [ ] T002 [US2] Add a fail-loud mode that exits non-zero when any requested kind yields zero samples.
- [ ] T003 [US2] Write a machine-readable summary JSON per build listing successful kinds and missing kinds.
- [ ] T004 [P] [US2] Add a focused smoke command in comments/docs that demonstrates the `3_0_1_8303` missing-source case.

**Checkpoint**: `3_0_1_8303` raw sample runs no longer end in silent success when `mh2o` / `mcnk_liquid` are absent.

---

## Phase 2: Deterministic Holdout Tile Trace (US4)

**Goal**: One exact bad tile can be traced from harvest-stream to derived liquid source.

- [ ] T005 [US4] Add a tile-targeting mode to `inspect_v16_harvest_samples.py` or a companion script that accepts `--map`, `--tile-x`, `--tile-y`.
- [ ] T006 [US4] Save raw arrays, decoded metadata, and derived source masks for one targeted tile into a dedicated folder.
- [ ] T007 [US4] Add a labeled comparison PNG for the targeted tile showing raw source panels and derived output.
- [ ] T008 [P] [US4] Record at least one known-wet `0_7_0_3694` tile and one known-wet `3_0_1_8303` tile in the trace workflow docs or prompt.

**Checkpoint**: A single known-bad tile from each holdout build can be traced deterministically.

---

## Phase 3: Holdout Harvest Seam Audit (US3)

**Goal**: Find the exact seam where explicit liquid truth disappears for the holdout builds.

- [ ] T009 [US3] Verify whether the holdout streamed NPZs contain usable `raw_chunks` MCNK payloads for targeted wet tiles.
- [ ] T010 [US3] Verify whether the holdout streamed NPZs contain usable `mh2o_presence_mask`, `mh2o_type_mask`, or `mclq_presence_mask` arrays.
- [ ] T011 [US3] Audit `AdtTensorPackBuilder.ReadMh2o(...)` against one holdout tile if `MH2O` is expected but missing.
- [ ] T012 [US3] Audit the pre-LK liquid extraction seam (`MCLQ` / MCNK flags / raw chunk metadata) for `0_7_0_3694`.
- [ ] T013 [US3] Patch only the proven failing seam in C# or Python, keeping already-good builds unchanged.
- [ ] T014 [P] [US3] Run `py_compile` and focused `dotnet build` proof after the seam fix lands.

**Checkpoint**: The holdout builds either produce explicit source arrays in raw samples or emit exact diagnostics proving the source is absent upstream.

---

## Phase 4: Repatch Final Stores (US3)

**Goal**: Turn the holdout seam fix into corrected final Zarr provenance.

- [ ] T015 [US3] Rerun `patch-liquids` for `0_7_0_3694`.
- [ ] T016 [US3] Rerun `patch-liquids` for `3_0_1_8303`.
- [ ] T017 [US3] Confirm `liquid_patch_report.json` changes per-source counts away from ambiguous `unified`-only when richer truth exists.
- [ ] T018 [P] [US3] Confirm `signal_validation.json` warnings shrink or become more precise after the repair.

**Checkpoint**: Final stores reflect the repaired provenance, not just the raw streamed source.

---

## Phase 5: Human Validation Refresh (US1)

**Goal**: Fresh images prove the repaired truth visually.

- [ ] T019 [US1] Regenerate finalized-store overviews for `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, and `4_0_0_11927`.
- [ ] T020 [US1] Regenerate raw harvest overviews for the repaired holdout build(s).
- [ ] T021 [US1] Compare holdout before/after images and summarize whether the tiles now show explicit `mcnk` / `mclq` / `mh2o` provenance.
- [ ] T022 [P] [US1] Update the relevant README or V16 spec doc if command behavior or validation expectations changed.

**Checkpoint**: Human validation images and JSON summaries agree on the final source truth.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1**: start immediately
- **Phase 2**: depends on Phase 1 fail-loud output
- **Phase 3**: depends on Phase 2 targeted tile tracing
- **Phase 4**: depends on Phase 3 seam fix
- **Phase 5**: depends on Phase 4 repaired stores

### Parallel Opportunities

- T004 can run while T001-T003 are in progress
- T008 can run while T005-T007 are in progress
- T014 can run after the seam fix while T015-T016 are being prepared
- T022 can run after validation results are known

## Implementation Strategy

### MVP First

1. Make raw sampling fail loud
2. Add one-tile trace mode
3. Fix one holdout build (`3_0_1_8303`) before touching the other
4. Repatch only the affected stores

### Stop Conditions

- If a holdout tile proves the raw source is absent upstream, stop patching repair logic and record that as truth.
- If the seam fix changes already-good builds, stop and back it out before continuing.
