# wow-viewer data-harvester — V18 Terrain System

Operator guide for the V18 terrain system: training pipeline, dataset
curation, and model inference.

**V18 is the whole terrain system.** The V18 namespace owns:

- **Dataset pipeline** — harvest → QA → V16 Zarr stores → curation → V18 refinement → manifest
- **Model family** — per-signal models (normal, height, holes, liquid, texcomp) under unified `train_v18.py`
- **Inference** — per-checkpoint assembly from independently trained V18 models

The V16.1/V17.1 files are implementation details. All public surfaces use V18 naming.

Detailed command coverage lives in:
- `docs/advanced-v16-workflows.md` — V16 dataset build/patch/validate commands
- `specs/024-v18-canvas-paste-refinement-layer/` — full V18 spec

---

## V18 Pipeline Overview

```
V16 Zarr Stores  (build_v16_dataset.py)
      │
      ▼
V16.1 Curation Manifest  (build_v16_curation_manifest.py)
  tile-level quality scoring + difficulty buckets
      │
      ▼
V18 Canvas Mining  (mine_v18_pastes_canvas.py Phase 1)
  multi-tile paste candidate extraction on stitched canvases
      │
      ▼
V18 Cross-Build Dedupe  (mine_v18_pastes_canvas.py Phase 2)
  perceptual hashing + Hamming distance → stable cluster IDs
      │
      ▼
V18 Composition Graph  (build_v18_composition_graph.py)
  adjacency/co-occurrence → composition families
      │
      ▼
V18 Paste Library Catalog  (build_v18_paste_library_catalog.py)
  canonical exemplars, deterministic names, role tags
      │
      ▼
V18 Refined Manifest  (build_v18_refined_manifest.py)
  cluster-balanced, family-aware train/val split
      │
      ▼
V18 Model Training  (train_v18.py normal/height/holes/liquid/texcomp)
  per-signal model, consumes refined manifest via --curation-manifest
```

---

## Setup

```powershell
cd wow-viewer/data-harvester
uv sync
```

---

## Standard Flow

1. Inspect raw harvest samples before any Zarr write.
2. Build or patch dataset stores with explicit write confirmation.
3. Validate dataset signals.
4. Generate visual QA artifacts.
5. Build a V16.1 curation manifest (tile-level quality scoring).
6. Run V18 canvas refinement (optional — for paste-deduped, family-balanced manifests).
7. Run trainer-readiness validation.
8. Train V18 model with curated or refined manifest.

---

## Raw Harvest QA Before Zarr

Use raw archive-backed harvest sampling first when a signal lane is under
investigation. This writes preview NPZs and PNGs under
`output/datasets/v16/harvest_signal_inspection/` and does not require an
existing `.zarr` store.

```powershell
uv run python scripts/inspect_v16_harvest_samples.py `
  --build 3_3_5_12340 `
  --maps Azeroth `
  --kinds object placement `
  --sample-count 8 `
  --output-dir ../output/datasets/v16/harvest_signal_inspection
```

Add `--compare-zarr` only when you intentionally want side-by-side comparison
against an already finalized store.

---

## Build Dataset

```powershell
cd i:/parp/parp-tools/wow-viewer
dotnet build ./WowViewer.slnx -c Debug

cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run python scripts/build_v16_dataset.py build `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --allow-zarr-write `
  --tile-workers 16 `
  --rebuild-existing
```

`build`, `patch-liquids`, `patch-objects`, and `merge-builds` now refuse to
touch `.zarr` stores unless `--allow-zarr-write` is present.

Default V16 build compression:
- codec: `lz4`
- level: `1`
- shuffle: `shuffle`

---

## Patch Existing Stores

These mutate the finalized base V16 stores.

Liquids only:

```powershell
uv run python scripts/build_v16_dataset.py patch-liquids `
  --allow-zarr-write `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

Objects only:

```powershell
uv run python scripts/build_v16_dataset.py patch-objects `
  --allow-zarr-write `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

Coordinate repair only:

```powershell
uv run python scripts/build_v16_dataset.py repair-index --build 3_3_5_12340
```

---

## Renderer-Truth Capture And Patch (V16.2)

The V16.2 dataset adds renderer-truth object masks and terrain-only minimaps
from viewer-generated captures. See `docs/advanced-v16-workflows.md` for the
full capture pipeline.

One-shot batch:

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester
uv run python scripts/build_v16_dataset.py generate-viewer-stubs --all
scripts\generate_all_renderer_truth_captures.bat
uv run python scripts/build_v16_dataset.py patch-renderer-truth --all
```

---

## Validate Dataset Signals

```powershell
uv run python scripts/build_v16_dataset.py validate-signals `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

---

## Human Visual QA

```powershell
uv run python scripts/inspect_v16_dataset.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --sample-count 24 `
  --sample-seed 1337 `
  --sample-mode liquid_focus `
  --write-overview `
  --output-dir ../output/datasets/v16/inspection
```

Review:
- `output/datasets/v16/inspection/<build>.validation_audit_overview.png`
- `output/datasets/v16/inspection/<build>.samples.json`
- `output/datasets/v16/inspection/<build>.summary.json`

---

## Trainer Readiness

```powershell
uv run python scripts/validate_v16_training_ready.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

---

## V18 Dataset Curation

V18 curation is a **four-layer** pipeline. Each layer transforms the dataset
representation before the trainer sees a single tile.

### Layer 1: V16.1 Curation Manifest (Tile-Level Quality)

This is the base quality layer. Every V18 training run starts here.

```powershell
uv run python -u scripts/build_v16_curation_manifest.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --profile normal_terrain_v16_1_1 `
  --workers -1 `
  --chunk-size 128 `
  --run-name normal_terrain_full_corpus_v16_1_1
```

What it checks per tile:
- **blank/low-signal minimaps** — reject flat featureless tiles
- **normal coverage** — does the tile have valid normals?
- **minimap-vs-normal edge agreement** — do edges in the minimap match normal edges?
- **what-plate tiles** — explicit blank genesis tiles
- **deformation richness** — how much terrain shape variation?
- **terrain-valid coverage** — how much of the tile is real terrain vs object/liquid?
- **painted alpha / MCLY presence** — does the tile have texture painting?
- **difficulty buckets** — easy / medium / hard / pathological

Outputs:
- `output/datasets/v16/curation/<run-name>/summary.json`
- `output/datasets/v16/curation/<run-name>/tiles.parquet`
- `output/datasets/v16/curation/<run-name>/kept_tiles.parquet`

### Layer 2: V18 Canvas Paste Mining + Cross-Build Dedupe

This is the key innovation over V16.1. Instead of treating tiles in isolation,
V18 stitches map canvases and detects **multi-tile paste regions** — authored
structures that span ADT boundaries and repeat across builds.

```powershell
uv run python -u scripts/mine_v18_pastes_canvas.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --canvas-size-tiles 10 `
  --canvas-stride 5 `
  --min-paste-size 32 `
  --alpha-threshold 0.15 `
  --stitch-workers 4 `
  --run-name v18_full_corpus_baseline
```

Phase 1 (canvas mining):
- Stitches tile signals (normals, SDF, alpha) into large canvases
- Detects paste regions via adaptive thresholding on combined signals
- Outputs candidates with `canvas_bbox` (pixel coords on stitched canvas)
  and `tile_coverage` (which tiles the paste spans)

Phase 2 (cross-build dedupe):
- Extracts perceptual hashes (dHash 64-bit) for each candidate crop
- Groups near-identical crops via Hamming distance (`--dedupe-hamming-threshold 16`)
- Assigns stable `cluster_id` across builds/maps
- Preserves alpha-layer signatures so RGB-similar candidates with different
  MCAL composition remain distinguishable

Outputs:
- `output/v18/pastes/<run-name>/candidates.jsonl`
- `output/v18/pastes/<run-name>/deduped_candidates.jsonl`
- `output/v18/pastes/<run-name>/dedupe_summary.json`

### Layer 3: V18 Composition Graph

Analyzes how paste candidates relate spatially — which paste families appear
adjacent, co-occurring, or stacked — to model the macro-zone composition
grammar.

```powershell
uv run python -u scripts/build_v18_composition_graph.py `
  --candidates ../output/v18/pastes/v18_full_corpus_baseline/deduped_candidates.jsonl `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --run-name v18_full_corpus_baseline
```

Outputs:
- `output/v18/pastes/<run-name>/composition_graph/`
- adjacency/co-occurrence statistics
- composition family assignments

### Layer 4: V18 Paste Library Catalog

Assigns deterministic canonical names and role tags to each paste family.

```powershell
uv run python -u scripts/build_v18_paste_library_catalog.py `
  --candidates ../output/v18/pastes/v18_full_corpus_baseline/deduped_candidates.jsonl `
  --run-name v18_full_corpus_baseline
```

Outputs:
- `output/v18/pastes/<run-name>/paste_library_catalog.json`
- stable family IDs, deterministic names
- role/shape tags: start, end, left, right, corner, connector, fill, transition
- canonical exemplar + variant linkage
- confidence metadata for auto-generated names

### Putting It Together: V18 Refined Manifest

Balances the deduped clusters into a training manifest that reduces duplicate
supervision while preserving motif diversity.

```powershell
uv run python -u scripts/build_v18_refined_manifest.py `
  --candidates ../output/v18/pastes/v18_full_corpus_baseline/deduped_candidates.jsonl `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --max-tiles-per-cluster 3 `
  --min-normal-richness 0.1 `
  --run-name v18_refined_baseline
```

Outputs:
- `output/v18/manifests/<run-name>/refined_tiles.parquet`
- cluster distribution stats and duplicate ratio metrics

The refined manifest is consumed by the trainer as a curation manifest:

```powershell
--curation-manifest ../output/v18/manifests/v18_refined_baseline
```

All V18 training commands below accept either a V16.1 curation manifest or a
V18 refined manifest as `--curation-manifest`. The V18 refined manifest adds
cluster-balancing metadata that the trainer uses for epoch sampling.

### End-to-End Pipeline (All Layers)

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

# 1. V16.1 curation manifest (tile-level quality)
uv run python -u scripts/build_v16_curation_manifest.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --profile normal_terrain_v16_1_1 --workers -1 --chunk-size 128 `
  --run-name normal_terrain_full_corpus_v16_1_1

# 2. V18 canvas mining + dedupe
uv run python -u scripts/mine_v18_pastes_canvas.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --canvas-size-tiles 10 --canvas-stride 5 --min-paste-size 32 --alpha-threshold 0.15 `
  --stitch-workers 4 --run-name v18_full_corpus_baseline

# 3. V18 refined manifest
uv run python -u scripts/build_v18_refined_manifest.py `
  --candidates ../output/v18/pastes/v18_full_corpus_baseline/deduped_candidates.jsonl `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --max-tiles-per-cluster 3 --min-normal-richness 0.1 `
  --run-name v18_refined_baseline

# 4. Train V18 normal model with refined manifest
uv run python -u scripts/train_v18.py normal `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/v18/manifests/v18_refined_baseline/refined_tiles.parquet `
  --device auto --batch-size 16 --grad-accum-steps 1 --target-vram-gb 12 --autotune-batch-size `
  --train-max-tiles 400 --train-epoch-tiles 128 --val-max-tiles 48 `
  --bucket-sampling-profile v16_1_1_normal `
  --epochs 50 --num-workers -1 --val-preview-interval 2 `
  --run-name v18_normal_refined
```

### Baseline Contract (Refined vs Non-Refined Comparison)

The V18 baseline contract script compares training with a V18 refined manifest
against training with the raw V16.1 curation manifest:

```powershell
uv run python -u scripts/run_v18_baseline_contract.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --candidates ../output/v18/pastes/v18_full_corpus_baseline/deduped_candidates.jsonl `
  --device auto --target-vram-gb 12 --autotune-batch-size `
  --train-max-tiles 400 --train-epoch-tiles 128 --val-max-tiles 48 `
  --epochs 50 --bucket-sampling-profile v16_1_1_normal `
  --run-name v18_baseline_contract
```

---

## V18 Training

### Unified Entrypoint

All V18 training uses a single entrypoint: `train_v18.py`. The first argument
is the task name; all subsequent arguments pass through to the common trainer.

```powershell
uv run python -u scripts/train_v18.py normal [...args]
uv run python -u scripts/train_v18.py height [...args]
uv run python -u scripts/train_v18.py holes [...args]
uv run python -u scripts/train_v18.py liquid [...args]
uv run python -u scripts/train_v18.py texcomp [...args]
```

Available task help:

```powershell
uv run python -u scripts/train_v18.py normal --help
```

### Recommended Launch: V18 Normal (RTX 4070 Ti SUPER, 16 GB)

```powershell
uv run python -u scripts/train_v18.py normal `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1 `
  --device auto `
  --batch-size 16 `
  --grad-accum-steps 1 `
  --target-vram-gb 12 `
  --autotune-batch-size `
  --train-max-tiles 400 `
  --train-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --val-max-tiles 48 `
  --epochs 50 `
  --num-workers -1 `
  --val-preview-interval 2 `
  --run-name v18_normal_curated_bs16_acc1_compile
```

### Small Scouting Run

```powershell
uv run python -u scripts/train_v18.py normal `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1 `
  --device auto --batch-size 16 --grad-accum-steps 1 `
  --target-vram-gb 12 --autotune-batch-size `
  --train-max-tiles 400 --train-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --val-max-tiles 48 --epochs 20 --num-workers 4 `
  --val-preview-interval 1 --no-compile `
  --run-name v18_normal_scout_pool400_ep128_bs16
```

Use `--no-compile` for tiny scouting runs where compile warmup would dominate
the whole job. Leave compile on for longer runs.

### Resume

```powershell
uv run python -u scripts/train_v18.py normal `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1 `
  --device auto --batch-size 16 --grad-accum-steps 1 `
  --target-vram-gb 12 --autotune-batch-size `
  --train-max-tiles 400 --train-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --val-max-tiles 48 --epochs 100 --num-workers -1 --val-preview-interval 2 `
  --run-name v18_normal_curated_bs16_acc1_compile `
  --resume-checkpoint ../models/v18/normal/runs/v18_normal_curated_bs16_acc1_compile/checkpoints/v18_normal_last.pt
```

### VRAM Ladder

- preferred start: `--batch-size 16 --grad-accum-steps 1`
- if headroom: `--batch-size 20 --grad-accum-steps 1`
- if more headroom: `--batch-size 24 --grad-accum-steps 1`
- first fallback: `--batch-size 12 --grad-accum-steps 1`
- second fallback: `--batch-size 8 --grad-accum-steps 1`
- if needed: `--batch-size 4 --grad-accum-steps 2`
- if needed: `--batch-size 2 --grad-accum-steps 4`
- safe floor: `--batch-size 1 --grad-accum-steps 8`

### Training Notes

- `--curation-manifest` accepts either a V16.1 curation manifest directory or a V18 refined manifest parquet file
- `--train-max-tiles` is the persistent curated train pool
- `--train-epoch-tiles` rotates a fresh per-epoch subset from that pool
- `--epochs` is total run ceiling; resume starts at `checkpoint_epoch + 1`
- `--num-workers -1` auto-resolves a CUDA-friendly worker count
- `torch.compile` is enabled by default on CUDA
- `--bucket-sampling-profile v16_1_1_normal` over-indexes `hard` tiles while preserving `medium`/`easy` stability
- `--autotune-batch-size` probes a batch-size ladder on the pooled train set
- `--normal-detail-boost` emphasizes terrain deformations over broad flats

### Model Architecture

All V18 models are **tiny, independent per-signal CNNs** with a shared
`_UNetBackbone` (encoder-decoder, ~3.5M params for normal model):

| Model | Input | Output | Params |
|-------|-------|--------|--------|
| `V18NormalModel` | minimap RGB (3ch) | normals (3ch, 257×257) | 3.5M |
| `V18NormalHeightModel` | minimap RGB + height (4ch) | normals (3ch, 257×257) | 3.5M |
| `V18HeightModel` | minimap RGB (3ch) | height (1ch, 257×257) | 3.5M |
| `V18HolesModel` | minimap RGB (3ch) | holes (1ch, 16×16) | 3.5M |
| `V18LiquidModel` | minimap RGB (3ch) | mask (1ch, 256×256) + type (5ch, 16×16) | 3.5M |
| `V18TexcompModel` | minimap RGB (3ch) | alpha (4ch) + mask (4ch) + IDs (4×16×16) | 3.5M |

Models are trained independently — each has its own checkpoint. No shared
weights between tasks. See `src/harvester/v18_models.py` for the complete
definition.

### Inference

```powershell
uv run python -u scripts/infer_v16_1.py `
  --build 3_3_5_12340 `
  --device auto --batch-size 8 `
  --height-checkpoint ../models/v18/height/runs/<run>/checkpoints/v18_height_best.pt `
  --normal-checkpoint ../models/v18/normal/runs/<run>/checkpoints/v18_normal_best.pt `
  --holes-checkpoint ../models/v18/holes/runs/<run>/checkpoints/v18_holes_best.pt `
  --liquid-checkpoint ../models/v18/liquid/runs/<run>/checkpoints/v18_liquid_best.pt `
  --texcomp-checkpoint ../models/v18/texcomp/runs/<run>/checkpoints/v18_texcomp_best.pt
```

(Namespace migration in progress — infer script will be renamed to `infer_v18.py`.)

---

## V16/V16.1 Legacy

The original V16 multitask trainer and V16.1 per-signal trainers remain
operational for reference runs and comparison baselines.

- `scripts/train_v16.py` — original V16 multitask model
- `scripts/train_v16_1_normal.py` — original V16.1 per-signal entrypoints
- `scripts/train_v16_2_normal.py` — V16.2 7-channel guidance input

These are stable but superseded. All new development uses the V18 namespace.

---

## Key Outputs

| Artifact | Path |
|----------|------|
| V16 dataset stores | `output/datasets/v16/<build>.zarr` |
| Per-build visual QA | `output/datasets/v16/inspection/` |
| V16.1 curation manifests | `output/datasets/v16/curation/<run-name>/` |
| V18 paste candidates | `output/v18/pastes/<run-name>/` |
| V18 composition graph | `output/v18/pastes/<run-name>/composition_graph/` |
| V18 paste library catalog | `output/v18/pastes/<run-name>/paste_library_catalog.json` |
| V18 refined manifests | `output/v18/manifests/<run-name>/` |
| V18 training runs | `models/v18/<task>/runs/<run-name>/` |
| V18 inference output | `output/datasets/v18_inference/<build>/<run-name>/` |
| V16 training runs (legacy) | `models/v16/runs/<run-name>/` |
| V16.1 training runs (legacy) | `models/v16_1/<task>/runs/<run-name>/` |
| V16.2 training runs (legacy) | `models/v16_2/<task>/runs/<run-name>/` |
