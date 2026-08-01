# Implementation Plan: Alpha Brush Library

**Branch**: `074-alpha-brush-library` | **Date**: 2026-06-22 | **Spec**: [spec.md](spec.md)

> Deprecated as primary plan (2026-06-23): This plan is retained for the existing component extractor, contact sheets, and evidence outputs. It must not be used as the active route for training labels. The replacement plan is `wow-viewer/specs/076-full-map-fractal-brush-library/plan.md`.

**Input**: Feature specification from `wow-viewer/specs/074-alpha-brush-library/spec.md`

## Summary

Build a library of unique fractal "brush" patterns discovered inside the MCAL alpha masks of every harvested terrain tile. A *brush* is a connected component of alpha > threshold inside a single alpha layer (L0-L3) on a single tile. By extracting, describing, and clustering these components across all builds and maps, we get a reusable catalog of terrain building blocks. This catalog is the prerequisite for downstream image-segmentation models that identify which brushes are present in a minimap tile, and eventually for reconstructing terrain height from identified brush pastes rather than from raw regression.

The technical approach uses existing V18 Zarr signals only — no new dataset pipeline:
1. Bulk-read `alpha_256` arrays (shape `(N, 256, 256, 4)`, dtype float32) from the existing V18 Zarr stores.
2. Extract connected components per layer with configurable alpha threshold and minimum area.
3. Cross-reference `mcly_texture_ids` and `mcly_layer_mask` for layer-stack context where useful.
4. Render each component as a binary/padded patch and run it through **DINOv2** (via Hugging Face `transformers`) to get a dense visual embedding.
5. Cluster component embeddings with **HDBSCAN** or **KMeans**.
6. Persist a JSONL catalog with cluster IDs and per-component metadata, plus PNG montages for human review.

DINOv2 replaces hand-engineered feature engineering. It is self-supervised, requires no labels, and has been shown to cluster semantically similar visual patterns. For our alpha masks, similar brush strokes (ridges, riverbeds, circular fills) should land close together in embedding space.

Because the ADT is a layered painting, each alpha layer is effectively a layer mask. A single artist brush stroke may appear as one connected component or several (if alpha dips below threshold or crosses chunk boundaries). DINOv2 embeddings help us recover the semantic identity of the stamp regardless of small breaks, and the layer index gives us the "layer mask" context for each stamp.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: NumPy, SciPy (connected components, find_objects), `transformers` (DINOv2 embeddings), scikit-learn (clustering), Zarr v3, PyArrow, Pillow, Matplotlib. `hdbscan` is optional; if unavailable, KMeans fallback is used.

**Storage**: Existing V18 Zarr stores under `wow-viewer/output/datasets/v18/`. Output catalog as JSONL + PNG montages under `wow-viewer/output/analysis/alpha-brush-library/`.

**Testing**: Script produces deterministic output when given the same data and seed. Validation is visual (montages) and quantitative (cluster sizes, per-layer contingency tables).

**Target Platform**: Windows/PowerShell development host with CUDA optional (clustering should run on CPU).

**Project Type**: Data-analysis CLI script + small shared-analysis library.

**Performance Goals**: Process all available V18 tiles in under 2 hours on the current workstation. Connected-component extraction must not loop per-tile in Python; use vectorized NumPy/SciPy ops.

**Constraints**: Do not rewrite existing MCAL readers or Zarr builders. Reuse the existing `alpha_256` arrays. Do not require proprietary client data beyond what is already staged in `output/tmp/wowarchive-clients/`.

**Scale/Scope**: ~20k tiles across multiple builds, ~4 layers per tile, up to several thousand components per tile depending on terrain complexity. Expected output: 1k–10k unique clusters.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | Pass | All code stays under `wow-viewer/`. |
| Library-First | Pass | Shared feature-extraction utilities go in `wow-viewer/data-harvester/src/harvester/`. CLI scripts are thin wrappers. |
| Real-Data Validation | Pass | Validation runs against existing V18 Zarr stores from staged clients. |
| Residual Model Chain | Pass | This spec is data extraction only. No model is trained. Future specs will add single-output segmentation models. |
| Streaming-First Dataset | N/A | No new dataset pipeline. Uses existing Zarr stores. |
| One Phase at a Time | Pass | Phases are sequential and each has a concrete validation gate. |
| Spec Docs | Pass | Plan and spec are being created before implementation. |
| Bite-Sized Plans | Pass | Max 5 phases, each with ≤ 10 tasks. |

## Project Structure

### Documentation (this feature)

```text
wow-viewer/specs/074-alpha-brush-library/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 findings
├── data-model.md        # Component/cluster/catalog schema
├── quickstart.md        # How to run the scripts
└── tasks.md             # Created by speckit-tasks
```

### Source Code

```text
wow-viewer/data-harvester/
├── src/harvester/
│   ├── alpha_brush.py       # Shared utilities: feature extraction, clustering, catalog I/O
│   └── v16_curation.py      # Existing; may be extended for AreaTable name lookups
└── scripts/
    ├── extract_alpha_brush_catalog.py   # Main extraction + clustering script
    └── visualize_alpha_brush_catalog.py # Montage + layer-analysis reports
```

### Output

```text
wow-viewer/output/analysis/alpha-brush-library/
├── components.jsonl         # One row per extracted component
├── clusters.jsonl           # One row per cluster with centroid + summary
├── catalog.jsonl            # Components joined to cluster IDs (main deliverable)
├── montages/
│   ├── cluster_grid.png     # Up to 16 examples per cluster
│   ├── layer_role_grid.png  # Cluster distribution by layer per map
│   └── cluster_size_hist.png
└── reports/
    ├── layer_contingency.json
    └── summary.txt
```

## Implementation Phases

### Phase 0 — Research & DINOv2 Embedding Check

**Goal**: Prove that connected-component extraction on `alpha_256` produces meaningful, reusable patterns, and verify that DINOv2 embeddings cluster them sensibly.

**Approach**:
1. Load one small build (e.g. `0_5_3_3368`) and a few representative maps.
2. For each layer L0-L3, run `scipy.ndimage.label` at alpha > 0.05 and inspect resulting components.
3. Tune the threshold: test 0.03, 0.05, 0.10. Document trade-offs (noise vs. broken-up patterns).
4. Render each component patch as a grayscale image, pad/crop to 224×224, and extract DINOv2-small embeddings (384-dim).
5. Run UMAP or PCA on the DINOv2 embeddings to see if clusters emerge visually.
6. Produce a small ad-hoc montage of raw components per layer and per discovered embedding cluster.

**Validation Gate**: A one-page `research.md` showing example components and a convincing 2D projection where similar-looking brushes group together when colored by DINOv2 embedding cluster.

### Phase 1 — Data Model & Shared Library

**Goal**: Define the catalog schema and build the reusable analysis library around DINOv2 embeddings.

**Approach**:
1. Define dataclasses or TypedDicts in `alpha_brush.py`:
   - `BrushComponent`: source tile info, layer, bounding box, alpha mask patch, DINOv2 embedding.
   - `BrushCluster`: cluster ID, centroid embedding, member count, representative component IDs.
   - `BrushCatalogEntry`: component joined to cluster.
2. Implement `extract_components(alpha_pack, layer_idx, threshold, min_area)` returning a list of components.
3. Implement `render_component_patch(component, target_size=224, padding=16)` to produce a normalized grayscale image patch for DINOv2.
4. Implement `compute_dinov2_embeddings(patches, model_name="facebook/dinov2-small", batch_size=64, device="cuda")` using `transformers.Dinov2Model`.
5. Implement clustering: HDBSCAN default, fallback to KMeans if clusters are too fragmented. Use cosine distance on L2-normalized DINOv2 embeddings.
6. Implement catalog serialization to JSONL and Parquet.

**Validation Gate**: Unit tests on synthetic alpha patches (circles, squares, fractal shapes) show that DINOv2 embeddings produce correct cluster assignment; similar synthetic shapes cluster together.

### Phase 2 — Bulk Extraction Script

**Goal**: Run the library across all builds and maps.

**Approach**:
1. Implement `scripts/extract_alpha_brush_catalog.py`:
   - CLI args: `--dataset-dir`, `--builds`, `--output-dir`, `--alpha-threshold`, `--min-area`, `--cluster-algo`, `--min-cluster-size`.
   - Load the Zarr index Parquet per build to enumerate tiles.
   - Filter to tiles with `has_alpha_256`.
   - For each tile, load `alpha_256` (256×256×4), extract components for all 4 layers.
   - Accumulate components, normalize features, cluster, assign cluster IDs.
   - Write `components.jsonl`, `clusters.jsonl`, `catalog.jsonl`.
2. Add deterministic seed option and deterministic cluster ordering (by size, then centroid hash).
3. Add progress logging and memory-responsible batching.

**Validation Gate**: Script runs end-to-end on `0_5_3_3368` and `3_3_5_12340` and produces >1000 clusters with >100 non-singleton clusters.

### Phase 3 — Visualization & Layer Analysis

**Goal**: Make the catalog interpretable and verify layer semantics.

**Approach**:
1. Implement `scripts/visualize_alpha_brush_catalog.py`:
   - Load `catalog.jsonl` + `clusters.jsonl`.
   - For each cluster, render a 4×4 or 5×5 grid of its member component masks.
   - Overlay cluster metadata (ID, member count, top maps, dominant layer).
2. Generate `layer_role_grid.png` showing cluster-size-weighted distribution of clusters across L0-L3 per map.
3. Compute per-map contingency tables and chi-squared test.
4. Write `reports/layer_contingency.json` and `reports/summary.txt`.

**Validation Gate**: Human review of the montage shows recognizable repeating patterns (ridges, riverbeds, circular fills). The layer-contingency report flags maps with statistically non-random layer usage.

### Phase 4 — Documentation & Integration Notes

**Goal**: Leave clear instructions and mark the path to the next spec.

**Approach**:
1. Write `data-model.md` with the exact schema.
2. Write `quickstart.md` with the exact commands.
3. Update `wow-viewer/memory-bank/activeContext.md` and `progress.md`.
4. Create a stub follow-up spec note for `073-alpha-brush-segmentation` outlining the next phase: train a per-patch segmentation model from minimap to brush cluster IDs.

**Validation Gate**: A fresh user can run the pipeline from `quickstart.md` and reproduce the catalog + montages.

## Complexity Tracking

None — the plan does not violate any constitution constraints. The only potential concern is that downstream work will train models, but model training is explicitly deferred to a future spec that will itself be small and single-output.

## Open Questions to Resolve During Implementation

1. What is the right alpha threshold for component extraction? Likely 0.05, but Phase 0 will confirm.
2. Should edge-touching components be kept or discarded? The spec says discard (avoid partial patterns), but Phase 0 may show some brushes are intentionally large and cross tile boundaries.
3. Which clustering algorithm gives clean groups on DINOv2 embeddings? HDBSCAN first; KMeans fallback.
4. Which DINOv2 model is best? `facebook/dinov2-small` is the default for speed; `dinov2-base` or `dinov2-large` may improve quality at a cost.
5. How should component patches be rendered/padded for DINOv2? Options: tight crop, fixed aspect-ratio crop, transparent background vs. black background. Phase 0 will compare.
6. Should we use the [CLS] token embedding or the mean-pooled patch tokens? DINOv2 gives both; Phase 0 will compare.
