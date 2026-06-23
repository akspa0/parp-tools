# Quickstart: Full-Map Fractal Brush Library

This quickstart covers Spec 076 Phases 1-3: bounded full-map canvas assembly, segmentation, and trainable library construction.

## Phase 1 Smoke

Run from `wow-viewer/data-harvester/`:

```powershell
uv run python scripts/build_full_map_fractal_canvas.py `
  --build 0_5_3_3368 `
  --map Azeroth `
  --tile-limit 4 `
  --layers 0,1,2,3 `
  --output-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact
```

Validated local output:

```text
Full-map fractal canvas built
  build: 0_5_3_3368
  map: Azeroth
  tiles: 4
  alpha_shape: (256, 1024, 4)
  height_shape: (257, 1025)
  mcly_shape: (16, 64, 4)
  output_dir: ..\output\analysis\full-map-fractal-brush-library\smoke_0_5_3_3368_Azeroth_tile4_compact
```

Key artifacts:

```text
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/canvas.zarr
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/canvas_index.parquet
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/summary.json
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/overlays/alpha_layer_slot_0_tile_seams.png
```

## Validation

```powershell
uv run ruff check src/harvester/fractal_canvas.py tests/test_fractal_canvas.py scripts/build_full_map_fractal_canvas.py
uv run pytest tests/test_fractal_canvas.py
```

Expected test result: `4 passed`.

## Current Limits

- Phase 1 writes dense bounded canvases. Full-continent chunk streaming is a future optimization after coordinate/provenance proof is accepted.
- `--tile-limit` selects a compact same-row tile window for smoke/proof runs instead of raw index order.
- Phase 2 segmentation is implemented for bounded canvas outputs and emits region metadata plus a review overlay.
- Phase 3 writes fixed-size accepted sample tensors plus source crop/provenance metadata; Phase 4 texture/BLP evidence is not joined yet.

## Phase 2 Smoke

Run from `wow-viewer/data-harvester/` after the Phase 1 smoke:

```powershell
uv run python scripts/segment_full_map_fractals.py `
  --canvas-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact `
  --output-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/segments `
  --threshold 0.05 `
  --min-area 16 `
  --max-regions-per-layer 200
```

Validated local output:

```text
Full-map fractal segmentation complete
  regions: 38
  curation_counts: {'accepted_candidate': 34, 'composite_chonker': 1, 'fractal_member': 3}
```

Key artifacts:

```text
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/segments/fractal_regions.parquet
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/segments/fractal_regions.jsonl
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/segments/summary.json
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/segments/overlays/fractal_regions_overlay.png
```

Additional validation:

```powershell
uv run ruff check src/harvester/fractal_segments.py tests/test_fractal_segments.py scripts/segment_full_map_fractals.py
uv run pytest tests/test_fractal_segments.py
```

Expected test result: `3 passed`.

## Phase 3 Smoke

Run from `wow-viewer/data-harvester/` after the Phase 2 smoke:

```powershell
uv run python scripts/build_fractal_brush_library.py `
  --canvas-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact `
  --regions ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/segments `
  --output-dir ../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact `
  --crop-size 128 `
  --smoke-count 32
```

Validated local output:

```text
Full-map fractal brush library built
  sample_count: 37
  rejected_count: 1
  split_counts: {'test': 4, 'train': 30, 'val': 3}
  smoke: {"dataset_size": 37, "labels": {"accepted_candidate": 30, "fractal_member": 2}, "loaded": 32, "requested": 32}
```

Key artifacts:

```text
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/samples.zarr
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/samples.parquet
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/rejected.parquet
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/split.parquet
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile4_compact/summary.json
```

Additional validation:

```powershell
uv run ruff check src/harvester/fractal_library.py tests/test_fractal_library.py scripts/build_fractal_brush_library.py
uv run pytest tests/test_fractal_library.py
```

Expected test result: `3 passed`.
