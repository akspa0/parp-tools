# Tasks: Precise M2 Masks in Tensor Packs

**Input**: `specs/001-precise-m2-masks/spec.md`, `specs/001-precise-m2-masks/plan.md`

**Prerequisites**: spec.md (done), plan.md (done)

**Notes**: Implementation is already complete in `AdtTensorPackBuilder.cs` (lines 1736-2441). All FR-001 through FR-009 are satisfied. These tasks cover validation and documentation only.

---

## Phase 1: User Story 1 — Precise M2 Masks (Priority: P1)

**Goal**: Verify that M2 doodad masks show actual triangle geometry instead of rectangles/dots.

**Independent Test**: Run `extract-unified` on azeroth_32_32 from staged 3.3.5 client; inspect `object_precise_mask` for triangle fills.

- [ ] T001 [US1] Run `extract-unified --map azeroth_32_32 --build 3_3_5_12340 --staging output/tmp/wowarchive-clients/3.3.5.12340` and confirm completion without errors
- [ ] T002 [US1] Inspect `object_precise_mask` array from the NPZ/Zarr output — verify triangular fill shapes for MDDF doodad placements (not rectangles or 2px dots)
- [ ] T003 [US1] Count MDDF entries with triangle footprints vs total MDDF=764 on azeroth_32_32 — confirm >=90% (SC-001)
- [ ] T004 [US1] Verify `object_mask_257`, `mddf_mask_257`, `object_filtered_mask_257` all show triangle coverage consistent with `object_precise_mask_257`

**Checkpoint**: US1 validated — M2 masks show triangle geometry

---

## Phase 2: User Story 2 — Graceful Fallback (Priority: P2)

**Goal**: Verify pipeline doesn't crash on missing/corrupt `.skin` files.

**Independent Test**: Run on a tile with a doodad whose `.skin` file is missing — observe bounds-rectangle fallback without crash.

- [x] T005 [US2] Review `TryLoadDoodadModelMetadata` at `AdtTensorPackBuilder.cs:2340` — confirm catch block at line 2412 sets `triangleVertices = null` on skin parse failure, and `DoodadModelMetadata` is still returned with bounds only
- [x] T006 [US2] Review fallback path at lines 1848-1863 — confirm `TryProjectBoundsToTilePixels` is called when `TriangleVertices` is null/empty
- [x] T007 [US2] Review fallback path at lines 1874-1889 — confirm `PaintCircle` is called when bounds projection fails
- [ ] T008 [US2] Run `extract-unified` on a tile where an M2 has no companion `.skin` file — confirm the tile completes without exception and the masks use bounds-rectangle or centroid-circle fallback

**Checkpoint**: US2 validated — no crashes on missing `.skin` files

---

## Phase 3: User Story 3 — WMO No-Regression (Priority: P3)

**Goal**: Confirm WMO mask painting is untouched.

**Independent Test**: Run `extract-unified` on a tile with MODF placements; compare WMO mask output to a prior known-good run.

- [x] T009 [US3] Confirm `TryPaintWmoFootprint` at `AdtTensorPackBuilder.cs:2241` is unchanged — no new code path touches WMO handling
- [ ] T010 [US3] Run `extract-unified` on a tile with MODF placements previously validated for WMO masks — compare `modf_mask_257` output for pixel-identity with prior run
- [ ] T011 [US3] If no prior artifact exists, visually inspect WMO triangle coverage in `modf_mask_257` for correctness (compare triangle shapes vs known WMO geometry)

**Checkpoint**: US3 validated — WMO masks unchanged

---

## Phase 4: Documentation & Spec Sync

**Purpose**: Finalize spec status and memory bank.

- [ ] T012 Update `spec.md` status from "Draft" to "Complete"
- [ ] T013 Update `wow-viewer/memory-bank/progress.md` with spec 001 completion
- [ ] T014 Update `wow-viewer/memory-bank/activeContext.md` to note 001 done

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (US1)**: No dependencies — can start immediately
- **Phase 2 (US2)**: No implementation dependencies — code review tasks (T005-T007) can run in parallel with Phase 1
- **Phase 3 (US3)**: No implementation dependencies — can run after Phase 1
- **Phase 4 (Documentation)**: Depends on all phases being complete

### Parallel Opportunities

- T001 (US1 run), T005-T007 (US2 code review), T009 (US3 code review) can all run in parallel
- T002-T004 depend on T001 completing
- T008 depends on T005-T007
- T010-T011 depend on T009

### Execution Order (Recommended)

1. T001 (run harvest) — this is the main validation bottleneck
2. While T001 runs: T005, T006, T007, T009 (code reviews, no build needed)
3. T002, T003, T004 (inspect results)
4. T008 (fallback test)
5. T010, T011 (WMO regression)
6. T012, T013, T014 (documentation)