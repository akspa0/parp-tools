# Tasks: Full-Map Fractal Brush Library

**Input**: Design documents from `wow-viewer/specs/076-full-map-fractal-brush-library/`

**Prerequisites**: `spec.md`, `plan.md`

**Tests**: Include unit tests for coordinate/provenance transforms and smoke tests on bounded real V18 Zarr stores.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Full-Map Canvas Assembly

**Purpose**: Replace ADT-tile-local processing with full-map ZBrush-like signal canvases and exact provenance.

- [x] T001 [US1] Create `wow-viewer/data-harvester/src/harvester/fractal_canvas.py` with tile-local to map-canvas coordinate transforms for 256x256 alpha/minimap pixels, 257x257 height/normal vertices, and 16x16 MCLY cells.
- [x] T002 [P] [US1] Add `wow-viewer/data-harvester/tests/test_fractal_canvas.py` covering coordinate transforms, tile seam adjacency, and chunk/cell provenance on synthetic map layouts.
- [x] T003 [US1] Create `wow-viewer/data-harvester/scripts/build_full_map_fractal_canvas.py` with CLI args for `--dataset-dir`, `--build`, `--map`, `--output-dir`, `--tile-limit`, and `--layers`.
- [x] T004 [US1] Implement alpha canvas assembly in `fractal_canvas.py` from V18 `alpha_256`, writing chunked Zarr arrays per layer or a compact multi-layer canvas.
- [x] T005 [US1] Implement aligned height/normal/MCLY provenance summaries in `fractal_canvas.py` without duplicating full source arrays when references are sufficient.
- [x] T006 [US1] Add coupled terrain-art provenance fields so alpha, height, normals, MCLY context, and later source-BLP matches can be joined by canvas coordinates.
- [x] T007 [US1] Write `canvas_index.parquet` with build, map, tile_id, tile_x, tile_y, canvas extents, source array availability, and provenance hash.
- [x] T008 [US1] Add seam/debug overlay output to `build_full_map_fractal_canvas.py` showing tile boundaries and alpha/height/normal continuity.
- [x] T009 [US1] Run a bounded real-data smoke command and record output paths/counts in `wow-viewer/specs/076-full-map-fractal-brush-library/quickstart.md`.

**Checkpoint**: One build/map canvas exists and proves cross-tile continuity plus provenance.

---

## Phase 2: Full-Map Fractal Segmentation And 074 Linkage

**Purpose**: Segment real map-canvas structures and demote 074 components to evidence rather than truth.

- [x] T010 [US2] Create `wow-viewer/data-harvester/src/harvester/fractal_segments.py` with region dataclasses/schema helpers for full-map regions, members, terrain-art primitives, and curation labels.
- [x] T011 [P] [US2] Add `wow-viewer/data-harvester/tests/test_fractal_segments.py` covering cross-tile component merging, chonker classification, one-off labeling, and 074 overlap joins on synthetic canvases.
- [x] T012 [US2] Create `wow-viewer/data-harvester/scripts/segment_full_map_fractals.py` with CLI args for canvas dir, 074 catalog dir, thresholds, output dir, and max regions.
- [x] T013 [US2] Implement full-map per-layer segmentation that ignores tile seams and emits canvas-space bboxes plus tile coverage.
- [x] T014 [US2] Implement region feature extraction: alpha coverage/gradient, tile coverage count, layer profile, height/normal sculpt response, MCLY layer/texture summary.
- [x] T015 [US2] Implement overlap linkage from full-map regions to 074 `component_id` rows using build/map/tile/layer/bbox provenance.
- [x] T016 [US2] Implement curation labels and rejection reasons: `accepted_candidate`, `fractal_member`, `composite_chonker`, `one_off_detail`, `too_small_unique`, `rejected_unknown`.
- [x] T017 [US2] Render layer-separated review overlays/contact sheets by curation label, with source build/map/tile spans and 074 linkage counts.

**Checkpoint**: Review overlays show cross-tile fractal structures and rejected chonkers/one-offs separately from accepted candidates.

---

## Phase 3: Trainable Library Contract

**Purpose**: Turn validated regions into a Zarr/Parquet training dataset with provenance.

- [x] T018 [US3] Create `wow-viewer/data-harvester/src/harvester/fractal_library.py` with trainable terrain-art primitive sample schema, stable ID generation, and curation-label filters.
- [x] T019 [P] [US3] Add `wow-viewer/data-harvester/tests/test_fractal_library.py` covering accepted/rejected filtering, stable IDs, split assignment, and provenance retention.
- [x] T020 [US3] Create `wow-viewer/data-harvester/scripts/build_fractal_brush_library.py` that consumes `fractal_regions.parquet` and source canvases.
- [x] T021 [US3] Write `samples.parquet`, `rejected.parquet`, `split.parquet`, and `summary.json` with explicit rejection reasons and source lineage.
- [x] T022 [US3] Write accepted sample tensors or source-reference crop metadata into `samples.zarr` without including rejected labels in default training splits.
- [x] T023 [US3] Implement a smoke dataset loader in `fractal_library.py` returning coupled alpha, height, normals, MCLY context, minimap/object context, optional source-BLP evidence where available, and provenance.
- [x] T024 [US3] Run a smoke loader over at least 32 accepted samples and document counts in `quickstart.md`.

**Checkpoint**: The trainable library is loadable and default samples exclude chonkers/one-off details.

---

## Phase 4: Tileset, Variant, And BLP Source Evidence Join

**Purpose**: Add texture/variant/effects evidence before any minimap-based model target is selected.

- [x] T025 [P] [US4] Inventory existing tileset/texture/fingerprint outputs in `wow-viewer/data-harvester/` and document reusable inputs in `research.md`.
- [x] T026 [US4] Add texture-ID summary fields to `fractal_library.py` from `mcly_texture_ids` and `mcly_layer_mask` provenance.
- [ ] T027 [US4] Add optional texture/variant fingerprint join fields when decoded texture evidence is available.
- [ ] T028 [US4] Add a read-only BLP source-candidate inventory plan that uses existing BLP decode/tooling and prioritizes FX, environment, weather, decal, particle, `textures\BloodSplats`, and transparent alpha-bearing paths.
- [ ] T029 [US4] Add optional BLP source-candidate fingerprint join fields with asset path, fingerprint ID, similarity score, and review state.
- [ ] T030 [US4] Update `visualize_fractal_brush_library.py` to render alpha, height/normal, minimap, texture-context, and BLP-source-candidate panels side by side.
- [ ] T031 [US4] Document unresolved texture/BLP-source gaps as follow-up tasks if canonical decoded texture fingerprints are not available.

**Checkpoint**: Bounded report links accepted candidates to MCLY texture IDs and available texture/variant/BLP source-candidate evidence.

---

## Phase 4B: Full-Map Strip Processing And Near-Dedupe

**Purpose**: Remove the artificial `--tile-limit` bound and process entire maps in memory-bounded strips.

- [x] T036 [US1/US2] Add chunked full-map canvas writer (`create_chunked_canvas_group`, `write_tile_to_canvas`) in `fractal_canvas.py` so entire continents can be assembled without dense in-memory arrays.
- [x] T037 [US1/US2] Add `--tile-limit 0` support to `analyze_fractal_raw_components.py`, loading every tile for a map from the build index.
- [x] T038 [US1/US2] Implement horizontal strip segmentation in `analyze_fractal_raw_components.py` with configurable strip width and overlap, translating strip-local bboxes back to global canvas coordinates.
- [x] T039 [US1/US2] Add strip-overlap region deduplication by bounding-box IoU.
- [x] T040 [P] Add unit tests for strip processing, bbox offset, and overlap dedupe.
- [x] T041 [US2] Implement near-duplicate clustering (translation/rotation/mirror invariant) after exact dedupe, because exact matching is too brittle (only ~566 duplicates out of 12,906 raw components on full Azeroth 0.5.3).
- [x] T042 [US2] Add a contact-sheet visualizer for near-duplicate clusters with the representative crop and up to N member examples.
- [ ] T043 [US2] Tune near-dedupe thumbnail size and Hamming radius against human review of cluster quality across all maps.
- [x] T044 [US2] Add rectangle/canvas-page boundary detection to separate obvious authored paste areas from connected alpha islands.
- [x] T046 [US2] Add a top-level HTML index (`index.html`) for --maps all runs, linking per-map canvases, overlays, and cross-map dedupe/near catalogs.
- [x] T047 [US2] Add macro paste/scar grouping mode (`--macro-pastes`) that merges nearby alpha strokes into paste-scale regions instead of treating raw brush dots as the primary output.
- [x] T048 [US2] Validate macro paste grouping on bounded and small full-map real-data runs, and keep raw-stroke exact/near dedupe as diagnostic evidence only.
- [x] T049 [US2] Add macro paste visual review outputs: full-map alpha overview, macro crop contact sheets, HTML review index, and top-level run index links.
- [x] T050 [US2] Add composite hard-region overview visualization using V18-style height/normal/alpha/MCLY/object/liquid signals under the same macro boxes.
- [x] T051 [US2] Add macro visual sweep CLI for comparing close-radius/min-area settings with linked review pages.
- [x] T052 [US2] Add middle-scale `blocky_paste` segmentation so dense authored chunks inside giant parent zones are emitted separately from zone-sized macro canvases.
- [x] T053 [US2] Add blocky paste visual proof and footprint cap (`--block-max-footprint`) to suppress oversized parent remnants while retaining blocky child chunks.
- [ ] T045 [US2] Validate rectangle-page thresholds against full-map overlays and contact sheets across all maps; tune min_area, min_extent, and max_aspect_ratio.

**Checkpoint**: Every map in each selected build index can be analyzed in strips and produces global-canvas region metadata; exact dedupe is replaced/augmented by near-dedupe before model target selection.

## Phase 5: Model Target Handoff

**Purpose**: Define correct future model targets from the curated library, without training yet.

- [ ] T032 [US5] Create `wow-viewer/specs/076-full-map-fractal-brush-library/model-targets.md` listing separate one-output future targets and required input signals, based on cross-map near-duplicate clusters and rectangle-page review.
- [ ] T033 [US5] Mark `wow-viewer/specs/075-scar-mask-segmentation/tasks.md` Phase 2/3 as deprecated unless explicitly reopened for coarse diagnostics.
- [ ] T034 [US5] Update `wow-viewer/docs/architecture/full-map-fractal-brush-library-2026-06-23.md` with validation evidence and approved first model target.
- [ ] T035 [US5] Update `wow-viewer/memory-bank/activeContext.md` and `wow-viewer/memory-bank/progress.md` after validation.

**Checkpoint**: Future training target is approved and based on curated full-map library outputs, not raw 074/075 labels.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1** blocks all later phases.
- **Phase 2** depends on Phase 1 and existing 074 output only as optional evidence.
- **Phase 3** depends on Phase 2 validation overlays.
- **Phase 4** depends on Phase 3 sample schema and can proceed before model target selection.
- **Phase 5** depends on Phase 3 and at least a bounded Phase 4 evidence check.

### Parallel Opportunities

- T002 can run while T001 implementation is being refined, after transform contract is drafted.
- T010 can be built from synthetic fixtures while T011 CLI scaffolding is written.
- T018 can be built while T019 CLI scaffolding is written.
- T024 can run independently after this plan lands.

## Implementation Strategy

1. Land docs/spec deprecation first.
2. Implement Phase 1 only.
3. Validate Phase 1 against one bounded real map.
4. Do not implement segmentation, library building, or model target work until Phase 1 evidence is reviewed.
