# Quickstart: Alpha Brush Library

This guide runs the current spec 074 implementation. It extracts MCAL alpha-mask brush components from existing V18 Zarr datasets, embeds them with DINOv2, clusters them, and writes JSONL catalog files.

The current useful outputs are the machine-readable brush catalog plus visual contact sheets: `components.jsonl`, `clusters.jsonl`, `catalog.jsonl`, and `montages_all/*.png`.

## Prerequisites

Run from the data-harvester project:

```powershell
cd I:\parp\parp-tools\wow-viewer\data-harvester
uv sync
```

Required V18 stores:

```text
../output/datasets/v18/0_5_3_3368.zarr/
../output/datasets/v18/3_3_5_12340.zarr/
```

Check they exist:

```powershell
Test-Path -LiteralPath "..\output\datasets\v18\0_5_3_3368.zarr"
Test-Path -LiteralPath "..\output\datasets\v18\3_3_5_12340.zarr"
```

First DINOv2 run may download `facebook/dinov2-small` into the Hugging Face cache. CUDA is optional; `--device auto` uses CUDA when available and CPU otherwise.

## Fast Smoke Run

Use this to prove the pipeline works before starting a full extraction:

```powershell
uv run python scripts/extract_alpha_brush_catalog.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --tile-limit 2 `
  --fallback-k 16 `
  --batch-size 32 `
  --output-dir ../output/analysis/alpha-brush-library/phase2-two-build-smoke
```

Expected behavior:

```text
Builds: 0_5_3_3368, 3_3_5_12340
Loading DINOv2 model facebook/dinov2-small on cuda/cpu
[build] tiles ... components=...
Embedding ... components
Clustering ... components
Wrote ... components, ... clusters, ... non-singleton clusters
```

Smoke output files:

```text
../output/analysis/alpha-brush-library/phase2-two-build-smoke/components.jsonl
../output/analysis/alpha-brush-library/phase2-two-build-smoke/clusters.jsonl
../output/analysis/alpha-brush-library/phase2-two-build-smoke/catalog.jsonl
```

The already-run local smoke produced 179 components and 16 clusters from two tiles per build.

## Full Two-Build Run

This is the required T022 validation path before Phase 3 visualization work can start:

```powershell
uv run python scripts/extract_alpha_brush_catalog.py `
  --builds 0_5_3_3368 3_3_5_12340 `
  --alpha-threshold 0.05 `
  --min-area 16 `
  --fallback-k 1000 `
  --batch-size 64 `
  --device auto `
  --seed 74 `
  --output-dir ../output/analysis/alpha-brush-library/two-build-full
```

Operational notes:

```text
0_5_3_3368 has 1,629 V18 alpha tiles locally.
3_3_5_12340 has 5,134 V18 alpha tiles locally.
```

The full run can take a long time because every extracted component is rendered to a 224x224 patch and embedded through DINOv2. If VRAM is tight, lower `--batch-size` to `16` or `8`. If CPU-only, expect a much slower run.

## Single-Build Run

Use this when debugging one build:

```powershell
uv run python scripts/extract_alpha_brush_catalog.py `
  --builds 0_5_3_3368 `
  --tile-limit 100 `
  --fallback-k 128 `
  --batch-size 32 `
  --output-dir ../output/analysis/alpha-brush-library/alpha-053-first100
```

Remove `--tile-limit` to process the whole build.

## Research Projection Run

Use this when you want visual PCA projection evidence rather than the full catalog:

```powershell
uv run python scripts/_research_alpha_components.py `
  --map Azeroth `
  --tile-limit 12 `
  --max-components 96 `
  --examples-per-layer 8 `
  --batch-size 16 `
  --device auto
```

Research outputs:

```text
../output/analysis/alpha-brush-library/research/summary.json
../output/analysis/alpha-brush-library/research/components_sample.jsonl
../output/analysis/alpha-brush-library/research/embeddings_sample.npz
../output/analysis/alpha-brush-library/research/projection.png
../output/analysis/alpha-brush-library/research/projection_cls.png
../output/analysis/alpha-brush-library/research/projection_mean.png
../output/analysis/alpha-brush-library/research/patches/
```

## Inspect The Results

Count rows:

```powershell
uv run python -c "from pathlib import Path; p=Path('../output/analysis/alpha-brush-library/two-build-full'); print('components', sum(1 for _ in (p/'components.jsonl').open())); print('clusters', sum(1 for _ in (p/'clusters.jsonl').open())); print('catalog', sum(1 for _ in (p/'catalog.jsonl').open()))"
```

Show the largest clusters:

```powershell
uv run python -c "import json; from pathlib import Path; p=Path('../output/analysis/alpha-brush-library/two-build-full/clusters.jsonl'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; [print(r['cluster_id'], r['member_count'], r['dominant_map'], r['dominant_layer']) for r in sorted(rows, key=lambda r: r['member_count'], reverse=True)[:20]]"
```

Show a few catalog entries:

```powershell
uv run python -c "import json; from pathlib import Path; p=Path('../output/analysis/alpha-brush-library/two-build-full/catalog.jsonl'); [print(json.loads(line)) for _, line in zip(range(5), p.open())]"
```

## Generate Contact Sheets

Render the top 100 clusters:

```powershell
uv run python scripts/visualize_alpha_brush_catalog.py `
  --catalog-dir ../output/analysis/alpha-brush-library/two-build-full `
  --max-clusters 100 `
  --max-per-cluster 8 `
  --clusters-per-page 50 `
  --cell-size 112 `
  --output-dir ../output/analysis/alpha-brush-library/two-build-full/montages
```

Render the full 1000-cluster library:

```powershell
uv run python scripts/visualize_alpha_brush_catalog.py `
  --catalog-dir ../output/analysis/alpha-brush-library/two-build-full `
  --max-clusters 1000 `
  --max-per-cluster 8 `
  --clusters-per-page 50 `
  --cell-size 112 `
  --output-dir ../output/analysis/alpha-brush-library/two-build-full/montages_all
```

Open:

```text
../output/analysis/alpha-brush-library/two-build-full/montages_all/index.html
```

Legend:

| Border color | Meaning |
|--------------|---------|
| Gray | L0 base/fill |
| Blue | L1 primary brush |
| Green | L2 transition/detail |
| Orange | L3 highlight/detail |

Each row is one cluster. Each cell is one representative alpha component crop from that cluster. Cell labels show full build ID, `map tileX,tileY`, `box x,y widthxheight`, layer, and area. These are atomic components, not yet multi-tile prefab/paste assemblies.

## Deduplicate Exact Scars And Rank Near Variants

Exact dedupe collapses binary-identical alpha crops into one canonical scar pattern, then ranks non-exact neighboring scars by DINOv2 embedding similarity within the existing cluster buckets.

Run exact binary dedupe over the full catalog:

```powershell
uv run python scripts/dedupe_alpha_brush_patterns.py `
  --catalog-dir ../output/analysis/alpha-brush-library/two-build-full `
  --output-dir ../output/analysis/alpha-brush-library/two-build-full/dedupe `
  --neighbors 8 `
  --examples-per-pattern 8
```

Current full-catalog result:

```text
components processed: 320,368
exact binary scar patterns: 263,188
largest exact pattern: 715 repeated components
near-neighbor rows: 2,105,504
```

Outputs:

```text
../output/analysis/alpha-brush-library/two-build-full/dedupe/exact_patterns.jsonl
../output/analysis/alpha-brush-library/two-build-full/dedupe/pattern_neighbors.jsonl
../output/analysis/alpha-brush-library/two-build-full/dedupe/dedupe_summary.json
```

Render exact scar patterns with ranked non-exact neighbors:

```powershell
uv run python scripts/visualize_alpha_brush_pattern_neighbors.py `
  --dedupe-dir ../output/analysis/alpha-brush-library/two-build-full/dedupe `
  --max-patterns 200 `
  --neighbors 7 `
  --patterns-per-page 40 `
  --cell-size 160 `
  --output-dir ../output/analysis/alpha-brush-library/two-build-full/dedupe/neighbor_montages
```

Open:

```text
../output/analysis/alpha-brush-library/two-build-full/dedupe/neighbor_montages/index.html
```

Interpretation: first cell in each row is the exact canonical scar. Following cells are non-exact neighbors ranked by cosine similarity. This is the best current view for “same brush idea, hand-fixed/blended differently”.

## Options That Matter

| Option | Default | Use |
|--------|---------|-----|
| `--builds` | all `.zarr` dirs | Limit extraction to specific builds. |
| `--tile-limit` | none | Bound a smoke/debug run. Remove it for full extraction. |
| `--alpha-threshold` | `0.05` | Component threshold from Phase 0 research. |
| `--min-area` | `16` | Drop tiny noisy components. |
| `--reject-edge` | off | Drop edge-touching components when explicitly requested. |
| `--batch-size` | `64` | DINOv2 embedding batch size. Lower this if VRAM is tight. |
| `--device` | `auto` | Use `cuda`, `cpu`, or auto-detect. |
| `--token-strategy` | `mean` | DINOv2 token strategy. `mean` is the Phase 0 default; `cls` is available for comparison. |
| `--cluster-algo` | `hdbscan` | Uses HDBSCAN if installed and useful, otherwise KMeans fallback. |
| `--fallback-k` | `100` | KMeans cluster count when HDBSCAN is unavailable or too noisy. Use a larger value for full corpus runs. |

## Current Limitations

- The contact-sheet visualizer exists. Higher-level multi-tile prefab/paste grouping is not implemented yet.
- Exact scar dedupe exists, but exact matches are only a first pass. Most patterns are still unique because the alpha masks include hand-authored fixups and blend scars.
- `hdbscan` is optional and not installed in the current local environment. The smoke-tested path uses scikit-learn KMeans fallback.
- `components.jsonl` includes embeddings and can be large on full runs.
- Full T022 validation is still open until `two-build-full` is run and the output is checked for `>1000` clusters and `>100` non-singleton clusters.

## Troubleshooting

If DINOv2 download is slow or rate-limited:

```powershell
$env:HF_TOKEN="<your token>"
```

If CUDA runs out of memory:

```powershell
--batch-size 16
```

If CPU is too slow, run a bounded extraction first:

```powershell
--tile-limit 100 --device cpu
```

If a build path is missing, rebuild or restore the V18 dataset under:

```text
wow-viewer/output/datasets/v18/<build>.zarr/
```
