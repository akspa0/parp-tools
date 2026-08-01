# Tasks: Heightmap Pattern Miner

## Phase 1

- [x] T001 Add Spec Kit documents for the heightmap pattern miner lane.
- [x] T002 Add `scripts/mine_heightmap_patterns.py`.
- [x] T003 Load Zarr `height_257` rows through `index.parquet`.
- [x] T004 Sample configurable terrain-cell spans with chunk-aligned starts.
- [x] T005 Group locally normalized low-resolution patch signatures.
- [x] T006 Write ranked `summary.json`.
- [x] T007 Write `pattern_atlas.png`.
- [x] T008 Add artifact suppression for low-variance and saturated patches.
- [x] T009 Run bounded real-data proof and inspect top repeated motifs.
- [x] T010 Replace tiny patch matching with minimum terrain-cell spans and chunk-local example metadata.

## Later

- [ ] T011 Join V23 validation error maps against mined motif IDs.
- [ ] T012 Export a V23 curriculum manifest if motif/error clustering is useful.
