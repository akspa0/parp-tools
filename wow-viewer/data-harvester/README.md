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

## V18 Dataset Builder Transition

An initial copy-forward V18 dataset builder now exists at
`scripts/build_v18_dataset.py`.

Current bounded status:

- it is the versioned successor to `build_v16_dataset.py`
- it writes stores under `wow-viewer/output/datasets/v18/`
- it keeps decoded metadata plus signal validation in the canonical build flow
- it emits `finalization.json` alongside the existing validation artifacts
- it can optionally promote renderer-truth captures during `build` with
  `--capture-root`, but only behind the explicit
  `--experimental-renderer-truth-promotion` gate
- object-roof mask arrays and roof-mask provenance now flow through the shared
  harvest/tensor-pack contract instead of depending only on a downstream Python
  patch pass
- remaining follow-up work is to reduce or retire the legacy Python-only
  roof-patching path once bounded real-data proof is complete

Use this as the active implementation surface for the V18 dataset contract while
the older V16 builder remains the stable baseline.

The larger direct parser → decoded → dataset redesign is intentionally deferred
to a future V20 dataset effort; V18 remains a bounded contract-closure pass over
the existing streaming interchange shape.

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

## Focused V18 Quickstart

Spec `047` now owns a focused two-build V18 lane:

- `0_5_3_3368`
- `3_3_5_12340`

Use the focused wrappers when you want the active terrain-reconstruction path
instead of the older broad-corpus operator flow.

Build the focused V18 curation manifest:

```powershell
uv run python -u scripts/build_v18_curation_manifest.py `
  --run-name v18_focus_terrain_v1 `
  --workers -1 `
  --chunk-size 128
```

Run the focused V18 full height session:

```powershell
uv run python -u scripts/train_v18_focus.py height `
  --device cuda `
  --epochs 40 `
  --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_v1 `
  --train-bucket-rotation-fraction 0.10 `
  --val-max-tiles 32 `
  --val-interval 1 `
  --run-name v18_height_focus_full_v1
```

Run the focused V18 full normal session:

```powershell
uv run python -u scripts/train_v18_focus.py normal `
  --device cuda `
  --epochs 40 `
  --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_v1 `
  --train-bucket-rotation-fraction 0.10 `
  --val-max-tiles 32 `
  --val-interval 1 `
  --run-name v18_normal_focus_full_v1
```

Derive an explicit reduced focused manifest when you want a smaller scouting
corpus instead of the full `4096`-row kept pool:

```powershell
uv run python -u scripts/build_v18_tiny_manifest.py `
  --source-manifest ../output/datasets/v18/curation/v18_focus_terrain_v1 `
  --samples-per-bucket-per-build 0 `
  --fraction-per-bucket-per-build 0.195 `
  --run-name v18_focus_tiny_800ish
```

When you train a reduced manifest, keep the same train commands but swap:

- `--curation-manifest ../output/datasets/v18/curation/v18_focus_tiny_800ish`
- `--train-bucket-rotation-fraction 1.0`
- run names such as `v18_height_focus_tiny_800ish` and
  `v18_normal_focus_tiny_800ish`

Run minimap-only focused inference proof:

```powershell
uv run python -u scripts/infer_v18_focus.py `
  --build 3_3_5_12340 `
  --limit 8 `
  --device cuda `
  --height-checkpoint ../models/v18/height/runs/v18_height_focus_full_v1/checkpoints/v16_1_height_best.pt `
  --normal-checkpoint ../models/v18/normal/runs/v18_normal_focus_full_v1/checkpoints/v16_1_normal_best.pt `
  --run-name v18_focus_minimap_only_proof
```

Notes:

- `build_v18_curation_manifest.py` defaults to the V18 dataset root and the two
  focused builds.
- the current recommended full-session commands are the two full focused runs:
  - `v18_height_focus_full_v1`
  - `v18_normal_focus_full_v1`
- `build_v18_tiny_manifest.py` derives a reduced scouting manifest from a
  focused kept pool; use `--fraction-per-bucket-per-build` for `800`-ish cuts
  or `--samples-per-bucket-per-build` for ultra-tiny caps.
- `train_v18_focus.py` defaults to the V18 dataset root, the two focused
  builds, the latest focused `kept_tiles.parquet` when present, and startup
  batch autotune against `--target-vram-gb 8`.
- `train_v18_focus.py` also defaults to restrained rotating bucket coverage via
  `--train-bucket-rotation-fraction 0.10`, so the run can train on a bounded
  fraction of each curated bucket per epoch instead of replaying the whole pool
  every epoch.
- `train_v18_focus.py` also defaults to strict near-equal per-build sampling, so
  oversized pool/epoch requests auto-cap to the largest feasible balanced
  subset instead of silently letting one build dominate.
- `train_v18_focus.py` now also defaults to `--early-stop-patience 8`, so the
  focused lane stops after eight non-improving validation epochs instead of
  idling all the way to the epoch ceiling.
- focused full height/base-normal runs now also apply a safer auto loader
  profile when `--num-workers` stays at `-1`; explicit `--num-workers`,
  `--prefetch-factor`, and `--persistent-workers` choices are preserved.
- when bucket rotation is active, omit `--train-epoch-tiles` and let the
  trainer derive the per-epoch subset size from the bucketed manifest itself.
- for reduced-manifest experiments, pass `--train-bucket-rotation-fraction 1.0`
  so the whole smaller manifest is seen each epoch instead of being sliced
  again.
- `--epochs` is now the ceiling, not a guarantee; the focused wrapper can stop
  earlier on a long plateau while still preserving the best checkpoint.
- trainer `val_loss` and preview images are offline supervised-eval surfaces:
  they can score against hidden dataset truth, but those tensors are not part
  of the deployed forward path.
- `infer_v18_focus.py` is the focused minimap-only proof surface with V18
  dataset/output defaults.
- focused curation now rejects tiles with too little surviving trainable
  terrain, so liquid-hidden wipeout rows stop entering the active pool.
- focused `height` and `normal` losses now honor terrain-valid masks, so
  liquid-hidden and object-hidden regions do not contribute loss.
- when harvested `object_roof_mask_256` is present, that roof/top-geometry
  occlusion is also folded into the active terrain-valid mask and the height
  preview weight panel.
- The active focused lane keeps height and normal as separate model runs.

---

## Why This Multi-Step Pipeline?

Each layer solves a different problem. You can stop at any layer, but skipping
layers means your training data has a specific blind spot:

| Layer | Problem it solves | Without it |
|-------|-------------------|-----------|
| **V16.1 curation** | Blank/low-signal tiles pollute training | Model learns to predict noise |
| **V18 canvas mining** | Authored structures span ADT boundaries; tile-local training fragments them | Model never learns multi-tile patterns |
| **V18 dedupe** | Same river bend appears 12× across 6 builds | Model over-fits to repeated motifs (memorization, not generalization) |
| **V18 refined manifest** | Common motifs dominate epoch sampling | Rare/unique terrain shapes get few gradient updates |

**You can skip V18 entirely** — just use the V16.1 curation manifest directly
with `train_v18.py`. That trains on ~50K quality-filtered tiles. But those
50K tiles contain ~140 paste families repeated across builds, so the model
spends most of its time looking at slight variations of the same terrain.

**V18 refinement collapses those 50K curated tiles into ~47 unique tile
references** — one per paste cluster — plus cluster-balancing weights. The
trainer sees fewer total tiles each epoch, but each tile represents a
genuinely different terrain shape. Validation evidence shows ~13.5% val_loss
improvement from this alone.

The composition graph and paste library catalog are informational — they
describe *what* was deduped (family names, spatial relationships) for
analysis and future work. They don't directly affect training.

### Standard Flow

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

### Stable Baseline: V16 Stores

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

### In-Progress V18 Builder Slice

```powershell
cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run python scripts/build_v18_dataset.py build `
  --build 3_3_5_12340 `
  --allow-zarr-write `
  --tile-workers 16
```

Optional bounded-proof-only renderer-truth integration during the V18 build:

```powershell
uv run python scripts/build_v18_dataset.py build `
  --build 3_3_5_12340 `
  --allow-zarr-write `
  --experimental-renderer-truth-promotion `
  --capture-root ../output/tmp/mdxviewer_validation_smoke
```

Treat that capture-derived path as experimental until bounded staged-client proof
reconfirms real object loading and image capture on the current anchors.

Latest bounded proof status:

- `WowViewer.Tool.ValidationCapture capture --real-scene-dry-run` reports ready
  scene state on both staged anchors:
  - `0_5_3_3368 / Azeroth_30_48`
  - `3_3_5_12340 / Azeroth_30_48`
- a non-dry-run `--renderer` capture on
  `3_3_5_12340 / Azeroth_30_48` completed `4/4` variants and emitted output
  files, but the captured images were still flat/uniform and the derived
  `object_visibility_mask` was all black
- therefore the renderer-truth promotion lane must still be treated as
  **not yet visually proven for real object rendering**, even though the command
  path and artifact emission completed successfully

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

## Spec 025 — Object-Roof Mask Library + Sieve

This lane adds an object-roof auxiliary signal for V18 normal training while
keeping terrain targets authoritative.

Build bounded roof library (Phase 1 proof):

```powershell
uv run python -u scripts/build_v18_object_roof_library.py `
  --build 3_3_5_12340 `
  --max-tiles-per-build 512 `
  --run-name smoke_spec025_phase1_335

uv run python -u scripts/validate_v18_object_roof_library.py `
  --library-dir ../output/datasets/object_roof_library/smoke_spec025_phase1_335
```

Train learned roof-family identifier (CUDA bounded proof):

```powershell
uv run python -u scripts/train_v18_object_roof_identifier.py `
  --library-dir ../output/datasets/object_roof_library/smoke_spec025_phase1_335 `
  --run-name smoke_spec025_roof_identifier_cuda_masked `
  --device cuda `
  --epochs 1 `
  --batch-size 8 `
  --max-samples 24 `
  --apply-roof-mask
```

Infer learned masks on anchor tiles and validate:

```powershell
uv run python -u scripts/infer_v18_object_roof_masks.py `
  --dataset-root ../output/datasets/v16 `
  --build 3_3_5_12340 `
  --map Azeroth `
  --tile-x 30 `
  --tile-y 53 `
  --max-tiles 1 `
  --model-dir ../models/v18/object_roof_identifier/smoke_spec025_roof_identifier_cuda_masked `
  --library-dir ../output/datasets/object_roof_library/smoke_spec025_phase1_335 `
  --output-dir ../output/tmp/v18_object_roof_infer_smoke_335_30_53

uv run python -u scripts/validate_v18_object_roof_masks.py `
  --pred-dir ../output/tmp/v18_object_roof_infer_smoke_335_30_53 `
  --build 3_3_5_12340 `
  --map Azeroth `
  --tile-x 30 `
  --tile-y 53 `
  --min-mask-coverage 0.01 `
  --min-top-family-score 0.05
```

Patch object-roof arrays into dataset stores (explicit write gate):

```powershell
uv run python -u scripts/patch_v18_object_roof_masks.py `
  --dataset-root ../output/datasets/v16 `
  --build 3_3_5_12340 `
  --learned-mask-dir ../output/tmp/v18_object_roof_infer_smoke_335_30_53 `
  --allow-zarr-write `
  --report-root ../output/tmp/object_roof_patch_reports `
  --run-name smoke_spec025_patch_335
```

Notes:
- Side artifacts intentionally live outside `.zarr` by default under
  `output/tmp/object_roof_patch_reports/`.
- Label contract is written to `object_roof_label_contract.json` in the report
  folder.
- Training integration uses `--normal-variant v18_object_roof_aux` and consumes
  `object_roof_mask_256` / `object_roof_weight_257` as auxiliary sieve signals.

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
  --dedupe --dedupe-hamming-threshold 12 `
  --out-dir ../output/v18/pastes/v18_full_corpus_baseline
```

Phase 1 (canvas mining):
- Stitches tile signals (normals, SDF, alpha) into large canvases
- Detects paste regions via connected-component analysis on
  combined hard-region/transition signals
- Thresholds: `--component-threshold 0.35`, `--min-component-area 240`
- Outputs candidates with `canvas_bbox` (pixel coords on stitched canvas)
  and `tile_coverage` (which tiles the paste spans)

Phase 2 (cross-build dedupe):
- Extracts perceptual hashes (dHash) for each candidate crop
- Groups near-identical crops via Hamming distance (`--dedupe-hamming-threshold 12`)
- Assigns stable `cluster_id` across builds/maps
- Preserves alpha-layer signatures so RGB-similar candidates with different
  MCAL composition remain distinguishable

Outputs (under `--out-dir`):
- `candidates.jsonl`
- `candidates_deduped.jsonl` (if `--dedupe` is set)
- `dedupe_summary.json`
- cluster atlases (PNG overviews per top cluster)

### Layer 3: V18 Composition Graph

Analyzes how paste candidates relate spatially — which paste families appear
adjacent, co-occurring, or stacked — to model the macro-zone composition
grammar.

```powershell
uv run python -u scripts/build_v18_composition_graph.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_baseline/candidates_deduped.jsonl `
  --output-dir ../output/v18/pastes/v18_full_corpus_baseline/composition_graph
```

Outputs (under `--output-dir`):
- `graph.json` — adjacency/co-occurrence edges
- `summary.json` — node/edge stats, top motifs
- composition family assignments

### Layer 4: V18 Paste Library Catalog

Assigns deterministic canonical names and role tags to each paste family.

```powershell
uv run python -u scripts/build_v18_paste_library_catalog.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_baseline/candidates_deduped.jsonl `
  --output-dir ../output/v18/pastes/v18_full_corpus_baseline/library_catalog
```

Outputs (under `--output-dir`):
- `paste_library_catalog.json` — stable family IDs, deterministic names
- role/shape tags: start, end, left, right, corner, connector, fill, transition
- canonical exemplar + variant linkage
- confidence metadata for auto-generated names

### Putting It Together: V18 Refined Manifest

Balances the deduped clusters into a training manifest that reduces duplicate
supervision while preserving motif diversity.

```powershell
uv run python -u scripts/build_v18_refined_manifest.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_baseline/candidates_deduped.jsonl `
  --run-name v18_refined_baseline
```

Outputs (under `../output/tmp/<run-name>/` by default):
- `kept_tiles.parquet` — refined training rows
- `tiles.parquet`, `tiles.jsonl` — full row dump
- `selected_candidates.jsonl` — which candidates contributed
- `summary.json` — cluster distribution stats, duplicate ratio

The refined manifest is consumed by the trainer as `--curation-manifest`:

```powershell
--curation-manifest ../output/tmp/v18_refined_baseline/kept_tiles.parquet
```

All V18 training commands below accept either a V16.1 curation manifest or a
V18 refined manifest as `--curation-manifest`. The V18 refined manifest adds
cluster-balancing metadata that the trainer uses for epoch sampling.

> **Tip:** Use `--output-dir` to write the refined manifest to a permanent
> location instead of `output/tmp/`. Example:
> `--output-dir ../output/v18/manifests/v18_refined_baseline`

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
  --dedupe --dedupe-hamming-threshold 12 `
  --out-dir ../output/v18/pastes/v18_full_corpus_baseline

# 3. V18 refined manifest
uv run python -u scripts/build_v18_refined_manifest.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_baseline/candidates_deduped.jsonl `
  --run-name v18_refined_baseline

# 4. Train V18 normal model with refined manifest
uv run python -u scripts/train_v18.py normal `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/tmp/v18_refined_baseline/kept_tiles.parquet `
  --device auto --batch-size 16 --grad-accum-steps 1 --target-vram-gb 12 --autotune-batch-size `
  --train-max-tiles 400 --train-epoch-tiles 128 --val-max-tiles 48 `
  --bucket-sampling-profile v16_1_1_normal `
  --epochs 50 --num-workers -1 --val-preview-interval 2 `
  --run-name v18_normal_refined
```

### Full-Corpus V18 Paste-Curation Training (Copy/Paste)

Use this when you want the full flow from paste mining to a real V18 normal run.

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

# Shared build list for all steps
$builds = @(
  '0_5_3_3368','0_5_5_3494','0_7_0_3694',
  '3_0_1_8303','3_3_5_12340','4_0_0_11927'
)

# 1) Base curation manifest (tile-level quality)
uv run python -u scripts/build_v16_curation_manifest.py `
  --builds $builds `
  --profile normal_terrain_v16_1_1 `
  --workers -1 --chunk-size 128 `
  --run-name normal_terrain_full_corpus_v16_1_1

# 2) Mine + dedupe pastes across full corpus
uv run python -u scripts/mine_v18_pastes_canvas.py `
  --dataset-dir ../output/datasets/v16 `
  --builds $builds `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v16_1_1/kept_tiles.parquet `
  --dedupe --dedupe-hamming-threshold 12 `
  --out-dir ../output/v18/pastes/v18_full_corpus_train

# 3) Optional: composition graph and library catalog artifacts
uv run python -u scripts/build_v18_composition_graph.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_train/candidates_deduped.jsonl `
  --output-dir ../output/v18/pastes/v18_full_corpus_train/composition_graph

uv run python -u scripts/build_v18_paste_library_catalog.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_train/candidates_deduped.jsonl `
  --composition-graph ../output/v18/pastes/v18_full_corpus_train/composition_graph `
  --output-dir ../output/v18/pastes/v18_full_corpus_train/library_catalog

# 4) Build refined manifest used by trainer
uv run python -u scripts/build_v18_refined_manifest.py `
  --deduped-candidates ../output/v18/pastes/v18_full_corpus_train/candidates_deduped.jsonl `
  --composition-graph ../output/v18/pastes/v18_full_corpus_train/composition_graph `
  --run-name v18_refined_full_corpus_train `
  --output-dir ../output/v18/manifests/v18_refined_full_corpus_train

# 5) Train V18 normal from refined paste manifest (real run)
uv run python -u scripts/train_v18.py normal `
  --dataset-dir ../output/datasets/v16 `
  --builds $builds `
  --curation-manifest ../output/v18/manifests/v18_refined_full_corpus_train/kept_tiles.parquet `
  --device auto --target-vram-gb 12 --autotune-batch-size `
  --batch-size 16 --grad-accum-steps 1 `
  --train-max-tiles 4000 --train-epoch-tiles 512 `
  --val-max-tiles 256 --rotate-val-tiles --val-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --epochs 300 --num-workers -1 --val-preview-interval 2 `
  --run-name v18_normal_full_from_pastes
```

Resume the same run:

```powershell
uv run python -u scripts/train_v18.py normal `
  --dataset-dir ../output/datasets/v16 `
  --builds $builds `
  --curation-manifest ../output/v18/manifests/v18_refined_full_corpus_train/kept_tiles.parquet `
  --device auto --target-vram-gb 12 --autotune-batch-size `
  --batch-size 16 --grad-accum-steps 1 `
  --train-max-tiles 4000 --train-epoch-tiles 512 `
  --val-max-tiles 256 --rotate-val-tiles --val-epoch-tiles 128 `
  --bucket-sampling-profile v16_1_1_normal `
  --epochs 600 --num-workers -1 --val-preview-interval 2 `
  --run-name v18_normal_full_from_pastes `
  --resume-checkpoint ../models/v18/normal/runs/v18_normal_full_from_pastes/checkpoints/v18_normal_last.pt
```

### Baseline Contract (Refined vs Non-Refined Comparison)

The V18 baseline contract script compares training with a V18 refined manifest
against training with the raw V16.1 curation manifest:

```powershell
uv run python -u scripts/run_v18_baseline_contract.py `
  --refined-manifest ../output/tmp/v18_refined_baseline `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --profile small
```

The script runs refined vs non-refined comparison and writes
`comparison_report.md` to `--output-dir`.

Current script defaults in `run_v18_baseline_contract.py` are:
- `small`: `epochs=1`, `train_max_tiles=64`, `train_epoch_tiles=16`, `val_max_tiles=16`, `val_epoch_tiles=8`, `batch_size=2`
- `medium`: `epochs=2`, `train_max_tiles=256`, `train_epoch_tiles=64`, `val_max_tiles=48`, `val_epoch_tiles=24`, `batch_size=4`
- `large`: `epochs=4`, `train_max_tiles=512`, `train_epoch_tiles=128`, `val_max_tiles=96`, `val_epoch_tiles=48`, `batch_size=8`

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
- `--train-bucket-rotation-fraction` rotates a deterministic fraction of every
  build/bucket stratum each epoch and is the preferred focused-V18 scouting
  mode when you want faster epochs without throwing away the rest of the
  curated pool
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
| V18 paste candidates + dedupe | `output/v18/pastes/<run-name>/` (set via `--out-dir`) |
| V18 composition graph | `output/v18/pastes/<run-name>/composition_graph/` (set via `--output-dir`) |
| V18 paste library catalog | `output/v18/pastes/<run-name>/library_catalog/` (set via `--output-dir`) |
| V18 refined manifests (default) | `output/tmp/<run-name>/` (or set via `--output-dir`) |
| V18 training runs | `models/v18/<task>/runs/<run-name>/` |
| V18 inference output | `output/datasets/v18_inference/<build>/<run-name>/` |
| V16 training runs (legacy) | `models/v16/runs/<run-name>/` |
| V16.1 training runs (legacy) | `models/v16_1/<task>/runs/<run-name>/` |
| V16.2 training runs (legacy) | `models/v16_2/<task>/runs/<run-name>/` |
