# Quickstart: Full-Map Fractal Brush Library

This quickstart covers Spec 076 Phases 1-3: bounded full-map canvas assembly, segmentation, and trainable library construction.

## Phase 1 Smoke

Run from `wow-viewer/data-harvester/`:

```powershell
uv run python scripts/build_full_map_fractal_canvas.py `
  --build 0_5_3_3368 `
  --map Azeroth `
  --tile-limit 16 `
  --layers 0,1,2,3 `
  --output-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact
```

Validated local output:

```text
Full-map fractal canvas built
  build: 0_5_3_3368
  map: Azeroth
  tiles: 16
  alpha_shape: (256, 4096, 4)
  height_shape: (257, 4097)
  mcly_shape: (16, 256, 4)
  output_dir: ..\output\analysis\full-map-fractal-brush-library\smoke_0_5_3_3368_Azeroth_tile16_compact
```

Key artifacts:

```text
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/canvas.zarr
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/canvas_index.parquet
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/summary.json
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/overlays/alpha_layer_slot_0_tile_seams.png
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
- `composite_chonker` rows are preserved as composite-canvas harvest targets. They are not assumed to be wrong, but default atomic brush splits exclude them until a composite-specific target exists.
- Default atomic brush samples require at least an `8x8` alpha-pixel footprint, the smallest authoring block size for the data we care about. Smaller slivers are preserved as review evidence, not default accepted samples.
- The earlier 4-tile smoke is still useful for coordinate proof, but it does not contain enough minimum-footprint atomic candidates for the 32-sample library loader gate.

## Phase 2 Smoke

Run from `wow-viewer/data-harvester/` after the Phase 1 smoke:

```powershell
uv run python scripts/segment_full_map_fractals.py `
  --canvas-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact `
  --output-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments `
  --threshold 0.05 `
  --min-area 16 `
  --min-atomic-footprint-px 64 `
  --max-regions-per-layer 500
```

Validated local output:

```text
Full-map fractal segmentation complete
  regions: 961
  curation_counts: {'accepted_candidate': 11, 'composite_chonker': 1, 'fractal_member': 24, 'one_off_detail': 2, 'too_small_unique': 923}
```

Key artifacts:

```text
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments/fractal_regions.parquet
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments/fractal_regions.jsonl
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments/summary.json
../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments/overlays/fractal_regions_overlay.png
```

Additional validation:

```powershell
uv run ruff check src/harvester/fractal_segments.py tests/test_fractal_segments.py scripts/segment_full_map_fractals.py
uv run pytest tests/test_fractal_segments.py
```

Expected test result: `4 passed`.

## Raw Component Analysis

Use raw mode when inspecting all detected alpha components without applying atomic/composite curation labels:

```powershell
uv run python scripts/segment_full_map_fractals.py `
  --canvas-dir <canvas-output-dir> `
  --output-dir <analysis-output-dir>/segments_raw `
  --threshold 0.05 `
  --min-area 1 `
  --curation-mode raw `
  --max-regions-per-layer 2000
```

Raw mode still computes bbox, tile coverage, height/normal stats, MCLY texture summaries, and overlays; it only changes the curation label to `raw_component` for every emitted region.

## One-Shot Raw Two-Build Dedupe

Run raw component analysis for the two target builds and write one exact-shape dedupe catalog:

```powershell
uv run python scripts/analyze_fractal_raw_components.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --maps Azeroth `
  --tile-limit 64 `
  --threshold 0.05 `
  --min-area 1 `
  --max-regions-per-layer 5000 `
  --output-root ../output/analysis/full-map-fractal-brush-library/raw_two_build_Azeroth_tile64
```

To include LK Northrend in the same run while skipping maps absent from 0.5.3:

```powershell
uv run python scripts/analyze_fractal_raw_components.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --maps Azeroth Northrend `
  --tile-limit 64 `
  --threshold 0.05 `
  --min-area 1 `
  --max-regions-per-layer 5000 `
  --output-root ../output/analysis/full-map-fractal-brush-library/raw_two_build_Azeroth_Northrend_tile64
```

To analyze every map present in each build index with `--maps all`:

```powershell
uv run python scripts/analyze_fractal_raw_components.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --maps all `
  --tile-limit 64 `
  --threshold 0.05 `
  --min-area 1 `
  --max-regions-per-layer 5000 `
  --output-root ../output/analysis/full-map-fractal-brush-library/two_build_all_maps_tile64
```

Validated real run on two builds (Azeroth only):

```text
output_root: ..\output\analysis\full-map-fractal-brush-library\two_build_test1
targets: 2
raw_components: 7317
exact_patterns: 3957
duplicate_patterns: 233
```

Validated tiny smoke:

```text
uv run python scripts/analyze_fractal_raw_components.py --builds 0_5_3_3368 3_3_5_12340 --maps Azeroth --tile-limit 2 --threshold 0.05 --min-area 1 --max-regions-per-layer 100 --output-root ../output/analysis/full-map-fractal-brush-library/smoke_two_build_raw_dedupe_tile2 --no-overlay
raw_components: 239
exact_patterns: 228
duplicate_patterns: 1
```

Key dedupe artifacts:

```text
<output-root>/dedupe/raw_components.parquet
<output-root>/dedupe/raw_components.jsonl
<output-root>/dedupe/exact_patterns.parquet
<output-root>/dedupe/exact_patterns.jsonl
<output-root>/dedupe/summary.json
```

Per-build/map artifacts are written under:

```text
<output-root>/<build>_<map>_tile<N>/canvas/
<output-root>/<build>_<map>_tile<N>/segments_raw/
```

## One-Command Analysis + Visualization

Run the full two-build raw analysis and render contact sheets in one command:

```powershell
uv run python scripts/analyze_fractal_raw_components.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --maps Azeroth `
  --tile-limit 64 `
  --threshold 0.05 `
  --min-area 64 `
  --min-footprint-px 8 `
  --max-regions-per-layer 5000 `
  --output-root ../output/analysis/full-map-fractal-brush-library/two_build_Azeroth_visualized `
  --visualize `
  --repeated-only
```

This is equivalent to running `analyze_fractal_raw_components.py` followed by `visualize_fractal_raw_patterns.py`. Artifacts are written under:

```text
<output-root>/dedupe/
<output-root>/<build>_<map>_tile<N>/canvas/
<output-root>/<build>_<map>_tile<N>/segments_raw/
<output-root>/contact_sheets/
```

## Visualizing Raw Exact Patterns Separately

If you already ran analysis, render contact sheets with:

```powershell
uv run python scripts/visualize_fractal_raw_patterns.py `
  --analysis-root ../output/analysis/full-map-fractal-brush-library/two_build_test2 `
  --output-dir ../output/analysis/full-map-fractal-brush-library/two_build_test2/contact_sheets `
  --max-patterns 200 `
  --max-per-pattern 6 `
  --repeated-only
```

Validated on `two_build_test2`:

```text
patterns_rendered: 17
pages: 1
```

Use `--min-members N` to require patterns with at least N raw-component members, or omit `--repeated-only` to render unique patterns too.

## Phase 3 Smoke

Run from `wow-viewer/data-harvester/` after the Phase 2 smoke:

```powershell
uv run python scripts/build_fractal_brush_library.py `
  --canvas-dir ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact `
  --regions ../output/analysis/full-map-fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/segments `
  --output-dir ../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact `
  --crop-size 128 `
  --smoke-count 32
```

Validated local output:

```text
Full-map fractal brush library built
  sample_count: 35
  rejected_count: 926
  split_counts: {'test': 1, 'train': 26, 'val': 8}
  smoke: {"dataset_size": 35, "labels": {"accepted_candidate": 9, "fractal_member": 23}, "loaded": 32, "requested": 32}
```

Key artifacts:

```text
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/samples.zarr
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/samples.parquet
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/rejected.parquet
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/split.parquet
../output/datasets/fractal-brush-library/smoke_0_5_3_3368_Azeroth_tile16_compact/summary.json
```

Additional validation:

```powershell
uv run ruff check src/harvester/fractal_library.py tests/test_fractal_library.py scripts/build_fractal_brush_library.py
uv run pytest tests/test_fractal_library.py
```

Expected test result: `3 passed`.
