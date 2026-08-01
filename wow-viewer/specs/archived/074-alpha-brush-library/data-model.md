# Data Model: Alpha Brush Library

This document describes the current Phase 2 catalog output written by:

```powershell
uv run python scripts/extract_alpha_brush_catalog.py
```

All output files are JSON Lines. Each line is one JSON object. Paths in examples are relative to `wow-viewer/data-harvester/` unless noted.

## Output Files

Default output directory:

```text
../output/analysis/alpha-brush-library/
```

Files:

```text
components.jsonl
clusters.jsonl
catalog.jsonl
```

## `BrushComponent`

Source type: `wow-viewer/data-harvester/src/harvester/alpha_brush.py::BrushComponent`

Written to: `components.jsonl`

Fields:

| Field | Type | Meaning |
|-------|------|---------|
| `component_id` | string | Stable `brush_<sha16>` ID derived from source tile, layer, threshold, and bounding box. |
| `build` | string | Build ID, for example `0_5_3_3368`. |
| `map_name` | string | Map name from the V18 `index.parquet`, for example `Azeroth`. |
| `tile_id` | integer | Row index into the build Zarr arrays. |
| `tile_x` | integer | ADT tile X from `index.parquet`, or `-1` if absent. |
| `tile_y` | integer | ADT tile Y from `index.parquet`, or `-1` if absent. |
| `layer_idx` | integer | Alpha layer index, `0` through `3`. |
| `bbox_xywh` | array of 4 integers | Component bounding box inside the 256x256 tile: `[x, y, width, height]`. |
| `area` | integer | Component pixel count after thresholding. |
| `threshold` | number | Alpha threshold used for extraction. Default is `0.05`. |
| `touches_edge` | boolean | Whether the component touches any tile edge. |
| `embedding` | array of numbers or null | L2-normalized DINOv2 embedding. Default token strategy is mean-pooled patch tokens. |
| `cluster_id` | integer or null | Assigned cluster ID after clustering. Noise points would be `-1`; current KMeans fallback assigns non-negative IDs. |

Not serialized:

```text
alpha_patch
mask_patch
```

Those arrays are in-memory only so the catalog stays reasonably small.

Example:

```json
{"area":120,"bbox_xywh":[20,32,18,12],"build":"0_5_3_3368","cluster_id":4,"component_id":"brush_...","embedding":[0.0123],"layer_idx":2,"map_name":"Azeroth","threshold":0.05,"tile_id":42,"tile_x":30,"tile_y":44,"touches_edge":false}
```

## `BrushCluster`

Source type: `wow-viewer/data-harvester/src/harvester/alpha_brush.py::BrushCluster`

Written to: `clusters.jsonl`

Fields:

| Field | Type | Meaning |
|-------|------|---------|
| `cluster_id` | integer | Deterministically remapped cluster ID, sorted by member count then centroid hash. |
| `member_count` | integer | Number of components assigned to the cluster. |
| `centroid_embedding` | array of numbers | L2-normalized centroid embedding for the cluster. |
| `representative_component_ids` | array of strings | Up to 16 component IDs closest to the centroid. |
| `dominant_layer` | integer or null | Most common alpha layer in the cluster. |
| `dominant_map` | string or null | Most common map in the cluster. |

## `BrushCatalogEntry`

Source type: `wow-viewer/data-harvester/src/harvester/alpha_brush.py::BrushCatalogEntry`

Written to: `catalog.jsonl`

This is the primary join table for downstream tools. It repeats the source metadata and the assigned cluster without embedding vectors.

Fields:

| Field | Type | Meaning |
|-------|------|---------|
| `component_id` | string | Component ID from `components.jsonl`. |
| `cluster_id` | integer | Cluster ID from `clusters.jsonl`. |
| `build` | string | Build ID. |
| `map_name` | string | Map name. |
| `tile_id` | integer | Row index into the build Zarr arrays. |
| `tile_x` | integer | ADT tile X. |
| `tile_y` | integer | ADT tile Y. |
| `layer_idx` | integer | Alpha layer index. |
| `bbox_xywh` | array of 4 integers | Component box in tile-local pixels. |
| `area` | integer | Component area in pixels. |
| `threshold` | number | Extraction threshold. |
| `touches_edge` | boolean | Whether the component touches a tile edge. |

## Extraction Semantics

Input array:

```text
alpha_256: shape (N, 256, 256, 4), dtype float32
```

Per tile and per layer:

```text
binary_mask = alpha_256[tile_id, :, :, layer_idx] > alpha_threshold
```

Connected components use 8-connectivity. Components smaller than `--min-area` are skipped. Edge-touching components are kept by default in the CLI because Phase 0 found many large real brush strokes touch tile boundaries. Pass `--reject-edge` to drop them.

Patch rendering:

```text
component alpha crop -> 16px padding -> aspect-preserving resize -> 224x224 grayscale patch
```

DINOv2 embedding:

```text
facebook/dinov2-small
mean-pooled patch-token embedding by default
L2-normalized before clustering
```

Clustering:

```text
HDBSCAN if installed and useful
KMeans fallback otherwise
```

Current local environment note: `hdbscan` is not installed, so the tested path uses scikit-learn KMeans fallback.
