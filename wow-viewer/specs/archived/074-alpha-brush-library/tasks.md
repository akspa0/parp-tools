# Tasks: Alpha Brush Library

> Deprecated as primary task list (2026-06-23): Remaining 074 work is evidence/diagnostic only unless explicitly reopened. Active brush-library work moves to `076-full-map-fractal-brush-library`.

**Input**: Design documents from `wow-viewer/specs/074-alpha-brush-library/`

**Prerequisites**: `plan.md`, `spec.md`

**Tests**: Include smoke tests for DINOv2 embedding extraction and synthetic-shape clustering.

**Organization**: Tasks grouped by phase. User stories are tagged in brackets.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 0: Research — DINOv2 Embedding Check on Alpha Components

**Purpose**: Validate that connected-component extraction on MCAL alpha masks produces meaningful patterns and that DINOv2 embeddings cluster them.

**Blocking**: Nothing — can start immediately.

- [x] T001 [US1] Create a one-off research notebook/script at `wow-viewer/data-harvester/scripts/_research_alpha_components.py` that reads `alpha_256` from the existing V18 Zarr store for one build (`0_5_3_3368`) and one map, runs `scipy.ndimage.label` per layer at alpha > 0.05, and prints component counts per layer.
- [x] T002 [US1] Implement component-to-patch rendering in the research script: crop each component to its bounding box, pad with 16px background, resize/pad to 224×224, save a few example PNGs per layer.
- [x] T003 [US1] Load `facebook/dinov2-small` via `transformers.Dinov2Model`, extract 384-dim embeddings for a sample of component patches, and save them as a small NumPy/Parquet file.
- [x] T004 [US1] Run PCA/UMAP on the sample embeddings, color points by layer and by visual inspection, and produce a 2D projection plot saved to `wow-viewer/output/analysis/alpha-brush-library/research/projection.png`.
- [x] T005 [US1] Test alpha thresholds 0.03, 0.05, and 0.10 on the same sample and document component counts and visual quality in `wow-viewer/specs/074-alpha-brush-library/research.md`.
- [x] T006 [US1] Compare [CLS] token vs. mean-pooled patch-token embeddings on the sample; document which clusters better in `research.md`.

**Checkpoint**: `research.md` exists and contains a convincing 2D projection plus threshold/token decision. No committed code yet — just findings.

---

## Phase 1: Foundational Library — `alpha_brush.py`

**Purpose**: Build the shared analysis library that all downstream scripts depend on.

**Blocking**: Phase 0 research must decide threshold and token strategy.

- [x] T007 [US1] Create `wow-viewer/data-harvester/src/harvester/alpha_brush.py` with TypedDicts/dataclasses: `BrushComponent`, `BrushCluster`, `BrushCatalogEntry`.
- [x] T008 [US1] Implement `extract_components(alpha_pack: np.ndarray, layer_idx: int, threshold: float, min_area: int, reject_edge: bool)` returning a list of `BrushComponent`. Use `scipy.ndimage.label` and `find_objects`.
- [x] T009 [US1] Implement `render_component_patch(component, target_size=224, padding=16, fill=0.0)` returning a `(target_size, target_size)` float32 grayscale patch with the component mask centered and padded.
- [x] T010 [US1] Implement `load_dinov2_model(model_name="facebook/dinov2-small", device="cuda")` returning the model and image processor.
- [x] T011 [US1] Implement `compute_dinov2_embeddings(patches: np.ndarray, model, processor, batch_size=64)` returning L2-normalized embeddings and handling device placement.
- [x] T012 [US1] Implement `cluster_components(components, algorithm="hdbscan", min_cluster_size=10, fallback_k=100)` returning components with assigned `cluster_id`. Fallback to `sklearn.cluster.KMeans` if HDBSCAN is not installed or produces too many noise points.
- [x] T013 [US1] Implement `build_cluster_catalog(components)` returning `clusters.jsonl` rows with centroid embedding, member count, representative IDs, and dominant layer/map.
- [x] T014 [P] [US1] Add JSONL serialization helpers: `save_components(path, components)`, `save_clusters(path, clusters)`, `save_catalog(path, entries)`.
- [x] T015 [US1] Add smoke test: `wow-viewer/data-harvester/tests/test_alpha_brush.py` with synthetic shapes (circle, square, fractal-like blob) verifying extraction and that DINOv2 embeddings cluster similar shapes together.

**Checkpoint**: `alpha_brush.py` is complete, tested, and can extract + embed + cluster synthetic alpha patches.

---

## Phase 2: Bulk Extraction Script

**Purpose**: Run the library across all available V18 builds and maps.

**Blocking**: Phase 1 library must pass smoke tests.

- [x] T016 [US1] Create `wow-viewer/data-harvester/scripts/extract_alpha_brush_catalog.py` with argparse: `--dataset-dir`, `--builds`, `--output-dir`, `--alpha-threshold`, `--min-area`, `--reject-edge`, `--cluster-algo`, `--min-cluster-size`, `--model-name`, `--batch-size`, `--seed`.
- [x] T017 [US1] In the script, enumerate tiles from each build's existing `index.parquet` in `wow-viewer/output/datasets/v18/<build>.zarr/`, filtering to tiles with `has_alpha_256 == True`.
- [x] T018 [US1] Read `alpha_256` directly from the existing V18 Zarr store per tile in batches, call `extract_components` for all 4 layers, accumulate components in a list, and log progress per build/map. Do not create a new dataset or copy pixel data.
- [x] T019 [US1] Render patches for all accumulated components and compute DINOv2 embeddings in batches.
- [x] T020 [US1] Run clustering, assign cluster IDs, and write `components.jsonl`, `clusters.jsonl`, and `catalog.jsonl` to `--output-dir`.
- [x] T021 [US1] Add deterministic seed handling and deterministic cluster ordering (sort clusters by member count descending, then by centroid hash).
- [ ] T022 [US1] Run end-to-end on `0_5_3_3368` and `3_3_5_12340` and verify output has >1000 clusters with at least 100 non-singleton clusters.

**Checkpoint**: Running `extract_alpha_brush_catalog.py` produces the full catalog JSONL files for the two primary builds.

---

## Phase 3: Visualization & Layer Analysis

**Purpose**: Make the catalog interpretable and validate layer semantics.

**Blocking**: Phase 2 catalog must exist.

- [x] T023 [US1] Create `wow-viewer/data-harvester/scripts/visualize_alpha_brush_catalog.py` with argparse: `--catalog-dir`, `--output-dir`, `--max-per-cluster`.
- [x] T024 [US1] Implement cluster montage rendering: for each cluster, draw a grid of up to 16 representative component patches and overlay cluster ID, member count, top maps, and dominant layer. Saves paginated contact sheets under `montages*/cluster_contact_sheet_*.png` plus `index.html`.
- [ ] T025 [US2] Implement per-map layer distribution rendering: for each map, show cluster-size-weighted bar charts of cluster counts per layer (L0-L3). Optionally cross-reference existing `mcly_texture_ids` and `mcly_layer_mask` zarr signals to enrich labels. Save to `montages/layer_role_grid.png`.
- [ ] T026 [US2] Compute per-map contingency tables (cluster vs. layer) and Pearson chi-squared test. Save to `reports/layer_contingency.json`.
- [ ] T027 [US2] Implement cluster-size histogram and noise-point fraction report. Save to `reports/summary.txt`.
- [ ] T028 [US2] Human-review pass: inspect `cluster_grid.png` and verify recognizable repeating patterns (ridges, riverbeds, circular fills) are present; document findings in a short `visualization_notes.md`.

**Checkpoint**: Montages and reports exist; layer contingency shows at least one map with statistically non-random layer usage (p < 0.01).

---

## Phase 4: Documentation & Handoff

**Purpose**: Preserve knowledge and mark the path to the segmentation follow-up.

**Blocking**: Phase 3 visualization must be reviewed.

- [x] T029 [P] [US1] Write `wow-viewer/specs/074-alpha-brush-library/data-model.md` with exact schemas for `BrushComponent`, `BrushCluster`, `BrushCatalogEntry`, and output files.
- [ ] T030 [P] [US1] Write `wow-viewer/specs/074-alpha-brush-library/quickstart.md` with exact commands to run extraction and visualization. Current `quickstart.md` covers setup, extraction, smoke/full runs, and output inspection; final visualization commands are pending Phase 3.
- [ ] T031 [P] [US2] Update `wow-viewer/specs/074-alpha-brush-library/research.md` with final threshold/token/clustering decisions and example outputs.
- [ ] T032 [US3] Create a stub follow-up spec note at `wow-viewer/specs/074-alpha-brush-library/073-segmentation-followup.md` outlining the next phase: train a per-patch minimap→brush-cluster segmentation model.
- [ ] T033 [P] Update `wow-viewer/memory-bank/activeContext.md` and `progress.md` to mark 072 as in-progress and 071 as in test/validation.

**Checkpoint**: A new contributor can follow `quickstart.md` and reproduce the catalog + montages.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 0**: No dependencies.
- **Phase 1**: Depends on Phase 0 decisions (threshold, token strategy).
- **Phase 2**: Depends on Phase 1 library passing smoke tests.
- **Phase 3**: Depends on Phase 2 catalog existing.
- **Phase 4**: Depends on Phase 3 visualization being reviewed.

### User Story Dependencies

- **US1 (P1)**: Spans Phases 0-4. Must complete first.
- **US2 (P2)**: Starts in Phase 0 (layer counts), fully implemented in Phase 3.
- **US3 (P3)**: Only a stub in Phase 4; full implementation deferred to feature `073-alpha-brush-segmentation`.

### Parallel Opportunities

- Phase 0 tasks T001, T002, T003 are sequential; T004/T005/T006 can run in parallel after embeddings are cached.
- Phase 1 tasks T007, T008, T009, T010 are parallel; T011 depends on T010; T012 depends on T008/T011; T013 depends on T012; T014/T015 are parallel.
- Phase 2 tasks T016/T017 are parallel; T018 depends on T017; T019 depends on T018; T020 depends on T019/T012; T021/T022 depend on T020.
- Phase 3 tasks T023/T024 are sequential; T025/T026/T027 are parallel after T024; T028 is manual review.
- Phase 4 documentation tasks T029/T030/T031/T032 are parallel; T033 depends on the others.

---

## Implementation Strategy

### MVP First

1. Complete Phase 0 → research findings
2. Complete Phase 1 → shared library + smoke tests
3. Complete Phase 2 → bulk catalog for two primary builds
4. Complete Phase 3 → montages + layer analysis
5. Complete Phase 4 → docs + handoff

### Validation Gates

- Phase 0: `research.md` shows DINOv2 clusters similar-looking alpha components.
- Phase 1: `pytest wow-viewer/data-harvester/tests/test_alpha_brush.py` passes.
- Phase 2: `extract_alpha_brush_catalog.py` finishes and produces >1000 clusters.
- Phase 3: `visualize_alpha_brush_catalog.py` produces montages with recognizable patterns.
- Phase 4: `quickstart.md` commands reproduce the full pipeline.

---

## Notes

- All new code lives under `wow-viewer/data-harvester/`.
- DINOv2 model download requires internet on first run; cache it locally via Hugging Face.
- Keep component patch rendering deterministic; use the same seed for clustering across runs.
- Do not modify existing training code or the V18 dataset builder in this feature.
