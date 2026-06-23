# Implementation Plan: Full-Map Fractal Brush Library

**Branch**: `076-full-map-fractal-brush-library` | **Date**: 2026-06-23 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `wow-viewer/specs/076-full-map-fractal-brush-library/spec.md`

## Summary

Build the corrected terrain-art decomposition pipeline: assemble full-map signal canvases, segment alpha/fractal regions in map coordinates, classify tile-local 074 components as evidence rather than truth, preserve height/normal/MCLY/texture/source-BLP provenance, and emit a trainable brush/fractal/paste terrain-art primitive library. The terrain mesh and alpha masks are treated as one coupled sculpt-and-paint object, like stacked ZBrush-like documents/layers. Training is explicitly blocked until this library is validated visually and structurally.

## Technical Context

**Language/Version**: Python 3.11+ for data-harvester analysis, with existing C# format readers used only through existing harvested Zarr outputs.

**Primary Dependencies**: NumPy, SciPy, Zarr v3, PyArrow/Parquet, Pillow, PyTorch only for optional batched/GPU signal calculations already present in V18 paste tooling.

**Storage**: Existing `wow-viewer/output/datasets/v18/<build>.zarr/` inputs. New outputs under `wow-viewer/output/analysis/full-map-fractal-brush-library/` and optionally a curated trainable store under `wow-viewer/output/datasets/fractal-brush-library/`.

**Testing**: Unit tests for coordinate/provenance transforms and smoke CLI tests on bounded real V18 stores. Visual validation via overlays/contact sheets is required before training.

**Target Platform**: Windows/PowerShell workstation with optional CUDA for signal precomputation.

**Project Type**: Data-harvester library + CLI workflow + documentation. No model training in initial phases.

**Performance Goals**: Bounded one-map run completes in minutes; full-map assembly is implemented as tile-chunked Zarr writes and horizontal strip segmentation so entire continents do not need to fit in memory as dense arrays.

**Constraints**: No parser rewrites. Use only existing staged/Zarr dataset paths. No training until curated-library validation gates pass. Preserve one-output model rule for future model specs. Do not treat alpha-only, minimap-only, or tile-local component labels as training truth.

**Scale/Scope**: Full maps can be up to 64x64 ADT tiles; layer canvases may be up to 16384x16384 pixels per layer. Implementation supports windowed/chunked Zarr writes and horizontal strip segmentation for memory-bounded full-continent analysis.

## Constitution Check

| Principle | Status | Notes |
|-----------|--------|-------|
| Repo Independence | Pass | All new specs and future code stay under `wow-viewer/`. |
| Library-First | Pass | Shared Python utilities belong in `data-harvester/src/harvester/`; CLIs are wrappers. |
| Real-Data Validation | Pass | Validation uses existing V18 Zarr stores from staged data. |
| Residual Model Chain | Pass | Initial scope is dataset/library only; future models must be separate one-signal specs. |
| Streaming/Dataset Discipline | Pass | Outputs are Zarr/Parquet datasets, not ad hoc JSONL-only truth. |
| One Phase At A Time | Pass | Training is blocked until map-canvas and curated-library validation pass. |

## Project Structure

### Documentation

```text
wow-viewer/specs/076-full-map-fractal-brush-library/
├── spec.md
├── plan.md
├── tasks.md
├── data-model.md              # future: exact schemas after Phase 1
└── quickstart.md              # future: exact commands after first CLI lands

wow-viewer/docs/architecture/
└── full-map-fractal-brush-library-2026-06-23.md
```

### Future Source Code

```text
wow-viewer/data-harvester/src/harvester/
├── fractal_canvas.py          # map-canvas assembly and provenance transforms
├── fractal_segments.py        # full-map segmentation and curation labels
└── fractal_library.py         # trainable library schema/build helpers

wow-viewer/data-harvester/scripts/
├── build_full_map_fractal_canvas.py
├── segment_full_map_fractals.py
├── build_fractal_brush_library.py
└── visualize_fractal_brush_library.py

wow-viewer/data-harvester/tests/
├── test_fractal_canvas.py
├── test_fractal_segments.py
└── test_fractal_library.py
```

### Output Shape

```text
wow-viewer/output/analysis/full-map-fractal-brush-library/<build>/<map>/
├── canvas.zarr/
├── canvas_index.parquet
├── fractal_regions.parquet
├── fractal_region_members.parquet
├── curation_review.parquet
├── overlays/
└── summary.json

wow-viewer/output/datasets/fractal-brush-library/<run-name>/
├── samples.zarr/
├── samples.parquet
├── rejected.parquet
├── split.parquet
└── summary.json
```

## Implementation Phases

### Phase 1 — Full-Map Canvas Assembly

**Goal**: Build map-scale alpha/MCLY/height/normal canvases with exact tile/chunk/local provenance, treating each map/layer stack as a ZBrush-like sculpt-and-paint document.

**Approach**:
1. Implement coordinate transforms between tile-local pixels, chunk/cell indices, and map-canvas pixels.
2. Assemble alpha layer canvases for bounded build/map subsets from `alpha_256`.
3. Assemble aligned height/normal summary surfaces or references for spatial statistics.
4. Assemble aligned MCLY texture ID and layer-mask context at 16x16 cell resolution with map-canvas references.
5. Record coupled terrain-art provenance so alpha, height, normals, MCLY context, and later BLP-source matches can be joined by canvas coordinates.
6. Write `canvas_index.parquet` mapping tile extents and provenance metadata.
7. Render debug overlays for tile seams and cross-tile continuity.

**Validation Gate**: One bounded build/map canvas proves cross-tile continuity and provenance for valid tiles. Full-map strip processing proves the same for an entire continent without loading the full dense canvas into memory.

### Phase 2 — Full-Map Fractal Segmentation And 074 Linkage

**Goal**: Segment full-map alpha/fractal/sculpt regions and join them to existing 074 components as evidence.

**Approach**:
1. Segment per-layer alpha structures on full-map canvases with tile seams ignored.
2. Detect virtual canvas groupings inside one ADT tile or across multiple tiles.
3. Compute region features: bbox, tile coverage, layer profile, alpha gradients, height/normal sculpt response, and MCLY texture/layer context.
4. Link regions to 074 `component_id` rows by source tile/layer/bbox overlap.
5. Classify candidate state: accepted/review, fractal member, composite chonker, one-off detail, too-small unique.
6. Write `fractal_regions.parquet` and `fractal_region_members.parquet`.
7. Render review overlays by layer and curation label.

**Validation Gate**: Review overlays show at least one preserved cross-tile structure and clearly separate chonkers/one-offs from accepted candidates. Full-map strip segmentation produces global-canvas bboxes and removes strip-overlap duplicates.

### Phase 3 — Trainable Library Contract

**Goal**: Convert validated regions into a trainer-consumable terrain-art primitive library with tensors and provenance.

**Approach**:
1. Define sample schema with tensor references, curation state, stable IDs, provenance, coupled mesh/alpha/MCLY context, optional source-BLP evidence, and train/val/test split.
2. Write accepted candidate crops/windows to Zarr or reference source arrays without duplicating unnecessary data.
3. Preserve rejected rows with reason codes and visual review links.
4. Add a dataset loader smoke test returning alpha, height, normals, MCLY context, minimap/object context where available, and provenance.
5. Produce summary metrics: accepted/rejected counts, repeatability distribution, chonker ratio, one-off ratio, missing-signal counts.

**Validation Gate**: A smoke loader reads at least 32 accepted samples and no default sample has a rejected curation label.

### Phase 4 — Tileset, Variant, And BLP Source Evidence Join

**Goal**: Add texture/variant/effects evidence so minimap visual contribution can be separated from alpha placement and terrain geometry, and so possible original BLP brush/decal/effect source assets can be reviewed.

**Approach**:
1. Inventory existing decoded texture/fingerprint sources in `wow-viewer` and identify reusable outputs.
2. Join MCLY texture IDs to candidate regions and summarize dominant texture families per sample.
3. Add texture/variant fingerprint fields where decoded texture assets are available.
4. Add a read-only scan of likely BLP source assets using existing decode/tooling surfaces, prioritizing FX, environment, weather, decal, particle, `textures\BloodSplats`, and alpha-bearing transparent textures.
5. Compare likely BLP source fingerprints to accepted alpha/fractal candidates and emit asset-path provenance plus similarity scores.
6. Render review sheets comparing alpha-only, height/normal, texture-context, BLP-source-candidate, and minimap-context panels.
7. Document unresolved texture/BLP-source gaps as a bounded follow-up if needed.

**Validation Gate**: At least one bounded map report links accepted candidates to MCLY texture IDs, available texture-family evidence, and any plausible BLP source-candidate matches.

### Phase 5 — Future Model Target Specification

**Goal**: Define model targets from the curated library without starting training prematurely.

**Approach**:
1. Write a follow-up model-target note listing separate one-output targets.
2. Mark 075 scar-mask as diagnostic/coarse only.
3. Propose the first trainable target from the curated library after visual validation.
4. Document required input channels for each candidate model, including height/normal context where appropriate.

**Validation Gate**: User-approved target note exists; no new trainer is launched from raw 074/075 labels.

## Complexity Tracking

None. The plan narrows scope by blocking training until dataset truth is corrected.

## Open Questions

1. Which bounded map/build should be the first validation target for cross-tile fractals and coupled sculpt/paint primitives? Teldrassil/root-heavy regions are strong candidates if present in the V18 store.
2. Which existing tileset-variant or decoded-BLP fingerprint output is canonical, if any, or does that need a separate follow-up spec?
3. Should accepted candidates be stored as copied tensors or as source-array references plus lazy crop metadata?
