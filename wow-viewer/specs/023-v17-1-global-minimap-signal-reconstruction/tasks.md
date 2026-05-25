# Tasks: V17.1 Global Minimap-Signal Reconstruction Contract

## Phase 1: Manifest-Driven Capture Targeting

- [ ] T001 Add `--curation-manifest` to `build_v16_dataset.py generate-viewer-stubs`
  - Load manifest rows (`keep=true`)
  - Filter stubs to `(build, tile_id)` keys from manifest

- [ ] T002 Emit capture ledger alongside generated stubs
  - Write per-build `manifest_capture_ledger.json`
  - Include requested tile IDs/names and pending status

## Phase 2: Manifest-Scoped Renderer-Truth Patching

- [ ] T003 Add `--curation-manifest` to `build_v16_dataset.py patch-renderer-truth`
  - Forward argument into `patch_v16_renderer_truth.py`

- [ ] T004 Add manifest filtering to `patch_v16_renderer_truth.py`
  - Restrict patch scope to manifest tile keys for the build
  - Preserve existing behavior when manifest is omitted

- [ ] T005 Add completion ledger/reporting to renderer-truth patching
  - Count captured-complete / partial / missing / skipped
  - Persist manifest-scoped completion summary JSON

## Phase 3: Normals-Training Contract Clarity

- [ ] T006 Rename confusing preview label in normals validation
  - `refined_gt` -> explicit teacher/refiner label

- [ ] T007 Keep explicit contract fields in trainer config/log
  - Verify run metadata records resolved normal contract toggles

## Phase 4: Curation/Mismatch Gate Tightening

- [ ] T008 Add V16.1 dataset curation threshold gates
  - Min terrain-validity score
  - Min minimap-target-usefulness score
  - Optional reject-what-plate flag

- [ ] T009 Expose threshold controls in `train_v16_1_common.py`
  - Pass threshold settings into `V161Dataset`
  - Log selected/rejected impact in evidence/config

## Phase 5: Validation

- [ ] T010 Run bounded `generate-viewer-stubs` with curation manifest and confirm reduced target tiles
- [ ] T011 Run bounded `patch-renderer-truth` with same manifest and inspect completion ledger
- [ ] T012 Run 1-epoch normals training sanity and verify clearer preview panel naming + contract fields
