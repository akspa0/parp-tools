# Tasks: Object Roof Mask Library and Minimap Sieve

## Phase 1 — Roof Library Curation

- [ ] T001 Define roof exemplar metadata schema
- [ ] T002 Extend MdxViewer capture for one-at-a-time object asset rendering with pose metadata
- [ ] T003 Write per-asset object visual outputs into a separate Zarr datastore
- [ ] T004 Extract top-view object crops from corpus placements and object-visual datastore
- [ ] T005 Add family dedupe and canonical exemplar selection
- [ ] T006 Emit roof atlas, catalog, and summary evidence
- [ ] T007 Validate bounded roof-catalog run on a building-heavy map

## Phase 2 — Object-Roof Mask Generation

- [ ] T008 Define object-mask label contract for minimap inputs
- [ ] T009 Implement metadata-driven object mask generation
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
