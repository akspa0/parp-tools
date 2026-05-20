# Tasks: V16 Dataset Signal Quality Fixes

**Plan**: `003-v16-dataset-signal-quality-fixes/plan.md`

---

## Phase 1: Archive & Clean Rebuild

- [ ] **1.1** Move existing stores: `mv output/datasets/v16/*.zarr output/datasets/v16/archive_before_003/`
- [ ] **1.2** Rebuild `3_3_5_12340` — `build_v16_dataset.py build --build 3_3_5_12340`
  - Verify `signal_validation.json` passes
  - Verify `tile_x`/`tile_y` not all 0 (5+ unique values)
  - Verify `mcnk_flags_16`, `mddf_mask`, `modf_mask`, `object_filtered_mask` arrays exist in Zarr
- [ ] **1.3** Rebuild `0_5_3_3368`
  - Same checks + verify `has_liquid_source_mcnk > 0`
- [ ] **1.4** Rebuild `0_5_5_3494`
  - Same checks
- [ ] **1.5** Rebuild `0_7_0_3694`
  - Same checks + verify `has_liquid_source_mclq > 0`
- [ ] **1.6** Rebuild `3_0_1_8303`
  - Same checks + verify tile_x not all 0
- [ ] **1.7** Rebuild `4_0_0_11927`
  - Same checks + verify `has_liquid_source_mh2o > 0`

## Phase 2: Human Validation Images

- [ ] **2.1** Generate overview + contact sheet for `3_3_5_12340`
  - `inspect_v16_dataset.py --build 3_3_5_12340 --write-overview --write-images --sample-mode random --sample-seed 42`
  - Visual check: tile coords, liquid mask, filtered vs raw object masks
- [ ] **2.2** Generate for `3_0_1_8303`
  - Spot-check ocean/coastal tiles — liquid must be present
- [ ] **2.3** Generate for `0_7_0_3694`
  - Spot-check MCLQ liquid presence
- [ ] **2.4** Generate for `0_5_3_3368`, `0_5_5_3494`, `4_0_0_11927`

## Phase 3: Training Smoke

- [ ] **3.1** Training readiness: `validate_v16_training_ready.py --build 3_3_5_12340`
  - Must return `overall_ok=true`, `issues=0`
- [ ] **3.2** Smoke train: 1 epoch, 8 train / 4 val tiles on CPU
  - Must complete without shape/dtype errors

## Phase 4: Documentation

- [ ] **4.1** Update `docs/architecture/v16-terrain-model-spec-2026-05-16.md` — add new arrays
- [ ] **4.2** Update `memory-bank/activeContext.md` — record raw format switch + new arrays
- [ ] **4.3** Update `memory-bank/progress.md` — mark spec 003 implementation complete
