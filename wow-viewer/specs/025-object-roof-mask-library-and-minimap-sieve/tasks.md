# Tasks: Object Roof Mask Library and Minimap Sieve

## Phase 1 — Roof Library Curation

- [x] T001 Define roof exemplar metadata schema
- [x] T002 Extend MdxViewer capture for per-asset orthographic top-down WMO+M2 rendering
  - WMO rendering, MDX/M2 rendering, doodad set variants, resume skip
- [x] T002a Add CLI flags for roof capture batch: --capture-roof, --capture-roof-asset-list, --capture-roof-resolution, --capture-roof-all-angles
- [x] T003 Write per-asset object visual outputs into a separate Zarr datastore
- [x] T004 Extract top-view object crops from MdxViewer per-asset renders (not from minimap tiles)
- [x] T005 Add family dedupe and canonical exemplar selection (in build_v18_object_roof_library.py)
- [x] T006 Emit roof atlas, catalog, and summary evidence
- [x] T007 Validate bounded roof-catalog run on a building-heavy map
- [x] T002b Build unified object catalog pipeline (build_v18_object_catalog_pipeline.py):
  - Emits per-build asset lists → MdxViewer capture → packs into unified object_visual.zarr
  - Supports --skip-capture, --skip-pack, --pack-only, --dry-run, resume

## Phase 2 — Object-Roof Mask Generation

- [x] T008 Define object-mask label contract for minimap inputs
- [x] T009 Implement metadata-driven object mask generation (patch_v18_object_roof_masks.py — marked as legacy/noise)
- [ ] T010 Add Python `uv` / transformers dependency and model-host scaffolding
- [ ] T011 Train a separate transformer-based object-identification model for roof/object signals
- [ ] T012 Add fallback object-roof inference for missing placement metadata
- [ ] T013 Emit side-by-side review artifacts for minimap, mask, provenance, and object-family outputs
- [ ] T014 Validate mask quality on an object-rich anchor tile

## Phase 3 — Training Integration

- [ ] T015 Wire object-roof signal into the normal training data pipeline
- [ ] T016 Preserve raw terrain targets while applying the object sieve
- [ ] T017 Add training evidence that records object-mask usage
- [ ] T018 Feed object-identification outputs into the main V18 model as auxiliary signals
- [ ] T019 Run a 1-epoch smoke comparison with auxiliary mask enabled

## Phase 4 — Operational Proof

- [ ] T020 Run the roof library on staged real-data anchors
- [ ] T021 Run object-mask generation on staged object-rich tiles
- [ ] T022 Record comparison results and update continuity docs
