# Tasks: V16 Inference Output Contract & Patch Pipeline

**Input**: Design documents from `wow-viewer/specs/001-v16-inference-output-contract-and-patch-pipeline/`

**Prerequisites**: plan.md (required), spec.md (required for user stories)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1-US5)

---

## Phase 1: Foundation — Zarr Read Support in Converter (US1)

**Goal**: `terrain-patch-adt` can read predictions from `.pred.zarr` directly.

- [ ] T001 [US1] Add Zarr NuGet dependency to `WowViewer.Tool.Converter.csproj` (e.g., `Zarr` or `ZarrSharp` — check what `data-harvester` uses and pick the .NET equivalent, or use Parquet-only index reading via `Parquet.Net`)
- [ ] T002 [US1] Add `--pred-zarr <path>` option to `TerrainPatchAdtCommand.cs` argument parsing (lines 17-26)
- [ ] T003 [US1] Implement `ReadPredIndexFromZarr(string predZarrPath)` that reads `index.parquet` from the Zarr store root and returns a list of tile coordinate records `(tile_id, map, tile_x, tile_y)`
- [ ] T004 [US1] Implement `ReadPredHeightFromZarr(string predZarrPath, int tileId)` that reads a single tile's `height_pred_257` array from the Zarr store
- [ ] T005 [US1] Implement `ReadPredLiquidMaskFromZarr(string predZarrPath, int tileId)` that reads `liquid_pred_mask_256` from the Zarr store (needed for US2 but build the seam now)
- [ ] T006 [US1] Add Zarr-based prediction source branch in the main patching loop: when `--pred-zarr` is set, resolve tile by `(map, tile_x, tile_y)` from the Zarr index instead of scanning `inference_summary.json` files
- [ ] T007 [US1] Add validation: when `--pred-zarr` is set and `--inference-dir` is also set, error with "use one prediction source, not both"
- [ ] T008 [US1] Add validation: when `--pred-zarr` is set, verify the store's `index.parquet` has required columns (`tile_id`, `map`, `tile_x`, `tile_y`) and required arrays (`height_pred_257`)
- [ ] T009 [P] [US1] Add unit test in `WowViewer.Core.Tests/` that creates a minimal in-memory Zarr-like index, writes it to temp, and verifies `ReadPredIndexFromZarr` parses it correctly

**Checkpoint**: `terrain-patch-adt --pred-zarr` produces identical patched ADTs to the staging-based flow for height+normals.

---

## Phase 2: Liquid Chunk Patching (US2)

**Goal**: Patched ADTs carry predicted liquid data in MH2O (LK) or MCLQ (Alpha) chunks.

- [ ] T010 [US2] Implement `ReadPredLiquidHeightFromZarr(string predZarrPath, int tileId)` in `TerrainPatchAdtCommand.cs` to read `liquid_height` arrays (or fallback to zero if absent)
- [ ] T011 [US2] Add `--replace-liquid` / `--no-replace-liquid` flags (default: replace) to `TerrainPatchAdtCommand.cs` argument parsing
- [ ] T012 [US2] Implement `PatchLkLiquidChunk(AdtObj0 obj0, float[,] liquidMask, float[,] liquidHeight)` that writes an `MH2O` chunk into the LK `_obj0.adt`. Start with MH2O type 1 (simple height-map liquid). Use existing chunk-writing infrastructure in `WowViewer.Core.IO.Maps`.
- [ ] T013 [US2] Implement `PatchAlphaLiquidChunk(...)` that writes `MCLQ` into Alpha embedded tile content. Use existing `Mcal.cs` / liquid encoding infrastructure if available; otherwise implement minimal MCLQ writer following the Alpha WDT research doc.
- [ ] T014 [US2] Wire liquid patching into the main patch loop: after height/normal patching, if `--replace-liquid` is set and liquid prediction is non-zero, call the appropriate liquid patcher based on target format.
- [ ] T015 [US2] Add edge case: when liquid mask is all zeros, skip liquid patching (preserve source or write empty, based on `--replace-liquid`).
- [ ] T016 [P] [US2] Add unit test that creates a minimal ADT object, patches liquid, and verifies the MH2O chunk is present with correct data.
- [ ] T017 [P] [US2] Add unit test for the all-zeros liquid mask edge case.

**Checkpoint**: `terrain-patch-adt --pred-zarr --replace-liquid` produces ADTs with liquid chunks for tiles that have predicted water.

---

## Phase 3: Patch Reports & Provenance (US4)

**Goal**: Every `terrain-patch-adt` run produces a `patch_report.json`.

- [ ] T018 [US4] Define `PatchReportEntry` record: `tile_name`, `map`, `tile_x`, `tile_y`, `replaced_channels` (list), `source_root_hash`, `source_obj_hash`, `pred_height_hash`, `pred_liquid_hash`, `outcome` ("patched", "skipped", "error"), `message`
- [ ] T019 [US4] Add `--report-path <path>` option to `TerrainPatchAdtCommand.cs` (default: `<output-dir>/patch_report.json`)
- [ ] T020 [US4] After each tile is patched, compute content hashes (SHA256) of source ADT root/obj and predicted height/liquid arrays, and append a `PatchReportEntry` to the report list.
- [ ] T021 [US4] At end of run, write the report list as `patch_report.json` to the output directory or `--report-path`.
- [ ] T022 [US4] Include `_inference_run.json` path in the top-level report metadata if available from the prediction store.
- [ ] T023 [P] [US4] Add unit test that patches a tile and verifies the report entry has correct fields.

**Checkpoint**: `terrain-patch-adt` always produces a patch report with per-tile provenance.

---

## Phase 4: One-Shot Pipeline (US3)

**Goal**: Single `infer-and-patch` command chains inference + patching + optional alpha conversion.

- [ ] T024 [US3] Create `InferAndPatchCommand.cs` in `WowViewer.Tool.Converter`
- [ ] T025 [US3] Add command-line arguments: `--build`, `--checkpoint`, `--client-root`, `--map`, `--output-dir`, `--alpha-output` (optional), `--device`, `--seed`, `--batch-size`, `--limit`
- [ ] T026 [US3] Implement pipeline stages: (1) invoke `infer_v16.py` via `Process.Start`, (2) invoke `terrain-patch-adt --pred-zarr` on the produced `.pred.zarr`, (3) optionally invoke `convert-lk-to-alpha` on the patched output.
- [ ] T027 [US3] Add error handling: if any stage fails, report which stage failed and exit with non-zero code. Do not silently skip failures.
- [ ] T028 [US3] Wire `InferAndPatchCommand` into `Program.cs` under the `"infer-and-patch"` case.
- [ ] T029 [P] [US3] Add integration test (smoke) that runs the full pipeline on a small tile subset and verifies output exists.

**Checkpoint**: `infer-and-patch` runs the full loop in one command.

---

## Phase 5: Inference Pair Validation (US5)

**Goal**: `validate-inference-pair` proves input/output stores are aligned.

- [ ] T030 [US5] Create `ValidateInferencePairCommand.cs` in `WowViewer.Tool.Converter`
- [ ] T031 [US5] Add command-line arguments: `--input-zarr`, `--output-zarr`
- [ ] T032 [US5] Implement validation: read both `index.parquet` files, check (a) same row count, (b) identical `tile_id` values in same order, (c) matching `tile_x`/`tile_y` values.
- [ ] T033 [US5] Report PASS/FAIL with specific mismatch details (first mismatched row, counts).
- [ ] T034 [US5] Wire `ValidateInferencePairCommand` into `Program.cs` under the `"validate-inference-pair"` case.
- [ ] T035 [P] [US5] Add unit test with a valid pair (should PASS) and a mismatched pair (should FAIL).

**Checkpoint**: `validate-inference-pair` catches all alignment issues.

---

## Phase 6: Polish & Documentation

- [ ] T036 [P] Update `wow-viewer/README.md` to document `--pred-zarr`, `--replace-liquid`, `infer-and-patch`, and `validate-inference-pair` commands.
- [ ] T037 [P] Update `wow-viewer/docs/architecture/v16-terrain-model-spec-2026-05-16.md` "Reconstruction Contract" section to reflect that direct Zarr consumption and liquid patching are now implemented.
- [ ] T038 [P] Update `wow-viewer/data-harvester/README.md` "Infer V16 and Patch Terrain" section to document the new streamlined flow.
- [ ] T039 Run full build: `dotnet build wow-viewer/WowViewer.slnx -c Debug`
- [ ] T040 Run full test suite: `dotnet test wow-viewer/WowViewer.slnx -c Debug`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Zarr Read)**: No dependencies — start immediately.
- **Phase 2 (Liquid)**: Depends on Phase 1 (reuses Zarr read infrastructure from T003-T005).
- **Phase 3 (Reports)**: Can run in parallel with Phase 2 (independent concern).
- **Phase 4 (Pipeline)**: Depends on Phase 1 (needs `--pred-zarr` working).
- **Phase 5 (Validation)**: No dependencies — can run in parallel with Phases 2-4.
- **Phase 6 (Polish)**: Depends on all prior phases.

### Parallel Opportunities

- T001-T002 can run in parallel with T009 (test infrastructure).
- T016-T017 (liquid tests) can run in parallel with T012-T015 (liquid implementation).
- T023 (report test) can run in parallel with T018-T022 (report implementation).
- Phase 3 (reports) and Phase 5 (validation) are fully independent of each other.
- T036-T038 (docs) can run in parallel.

### Within Each Phase

- Argument parsing before implementation.
- Implementation before tests.
- Tests before moving to next phase.

---

## Implementation Strategy

### MVP First (Phases 1 + 3)

1. Complete Phase 1: Direct Zarr consumption
2. Complete Phase 3: Patch reports
3. **STOP and VALIDATE**: Run `terrain-patch-adt --pred-zarr` on a real inference store and verify output + report

### Incremental Delivery

1. Phase 1 → `terrain-patch-adt --pred-zarr` works for height+normals
2. Phase 2 → liquid patching added
3. Phase 3 → patch reports always generated
4. Phase 4 → one-shot `infer-and-patch` command
5. Phase 5 → `validate-inference-pair` command
6. Phase 6 → docs updated

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Commit after each phase or logical group
- Stop at any checkpoint to validate independently
- The `infer_v16.py` script is reference-only for C# work; do not modify it as part of this feature
