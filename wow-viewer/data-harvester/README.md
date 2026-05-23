# wow-viewer data-harvester

Operator guide for the live V16 dataset and training workflow.

Detailed command coverage lives in:

- `data-harvester/docs/advanced-v16-workflows.md`

## Setup

```powershell
cd wow-viewer/data-harvester
uv sync
```

## Standard Flow

1. Inspect raw harvest samples before any Zarr write.
2. Build or patch dataset stores with explicit write confirmation.
3. Validate dataset signals.
4. Generate visual QA artifacts.
5. Build a target-aware curation manifest.
6. Run trainer-readiness validation.
7. Train only after the store passes raw-harvest QA, JSON QA, human-eye QA, and curation.

## Raw Harvest QA Before Zarr

Use raw archive-backed harvest sampling first when a signal lane is under investigation. This writes preview NPZs and PNGs under `output/datasets/v16/harvest_signal_inspection/` and does not require an existing `.zarr` store.

```powershell
uv run python scripts/inspect_v16_harvest_samples.py `
  --build 3_3_5_12340 `
  --maps Azeroth `
  --kinds object placement `
  --sample-count 8 `
  --output-dir ../output/datasets/v16/harvest_signal_inspection
```

Add `--compare-zarr` only when you intentionally want side-by-side comparison against an already finalized store.

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

`build`, `patch-liquids`, `patch-objects`, and `merge-builds` now refuse to touch `.zarr` stores unless `--allow-zarr-write` is present.

Default V16 build compression is light Blosc:

- codec: `lz4`
- level: `1`
- shuffle: `shuffle`

## Patch Existing Stores

These commands mutate the finalized base V16 stores. They remain correct for
the original V16 contract, but they are not the recommended first surface for
new renderer-truth guidance signals while cross-build validation is still
incomplete.

Current `V16.2` direction (direct-patch, not sidecar):

- object mask data is patched directly into existing V16 Zarr stores
- V16.2 training uses 7-channel input (3 minimap + 4 guidance channels)
- renderer-truth arrays (`object_visibility_mask`, `no_object_minimap`) are
  optional — training works without them using the guidance channels already
  in the stores
- renderer-truth patches are available for `0_5_3_3368` and `3_3_5_12340`
  via `patch_v16_renderer_truth.py`

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

## Renderer-Truth Capture And Patch (V16.2)

The V16.2 dataset adds renderer-truth object masks and terrain-only minimaps
from MdxViewer captures. This is a three-step pipeline.

**Signal inventory — what the harvest stream vs the viewer produce:**

| Signal | Source | In V16 Zarr? |
|--------|--------|-------------|
| height_257, normal_xyz, alpha_256, holes_16 | harvest stream (C#) | Yes |
| liquid_mask, liquid_height, mcnk_flags_16 | harvest stream (C#) | Yes |
| object_mask, object_precise_mask, object_filtered_mask | harvest stream (C#) | Yes |
| mddf_mask, modf_mask, object_instance_mask | harvest stream (C#) | Yes |
| minimap_rgb, shadow_mask | harvest stream (C#) | Yes |
| mcly_texture_ids, mcly_layer_mask | harvest stream (C#) | Yes |
| **object_visibility_mask** | **MdxViewer capture** | V16.2 only |
| **no_object_minimap** | **MdxViewer capture** | V16.2 only |

The harvest stream generates all terrain/texture/object signals. The viewer
produces the renderer-truth overlay that the harvest cannot: the actual
rendered minimap with objects visible vs hidden, and the diff mask.

### Step 1: Generate per-tile stubs

Reads `index.parquet` from each V16 Zarr store and writes per-tile JSON stubs
that the viewer uses for tile discovery.

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester
uv run python scripts/build_v16_dataset.py generate-viewer-stubs `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

Output: `output/tmp/mdxviewer_validation_smoke/<build>/dataset/<MapName>_<tileY>_<tileX>.json`

### Step 2: Run MdxViewer captures

Each viewer session renders all tiles for one build in multiple visibility
families (primary, noobjects, objectsonly, noliquids) and outputs PNGs.

**Required CLI parameters:**
- `--game-path` — staged client root (`output/tmp/wowarchive-clients/<build>/World of Warcraft`)
- `--build` — build version string (e.g. `0.5.3.3368`)
- `--listfile` — community listfile for name resolution
- `--world` — WDT path for the map to render (e.g. `World\Maps\Azeroth\Azeroth.wdt`)
- `--validation-dataset-root` — directory containing `dataset/` with per-tile stubs
- `--validation-output` — where to write captured PNGs
- `--validation-resolution` — capture size in pixels (256-4096)
- `--force-validation-regeneration` — re-capture even if outputs exist
- `--exit-after-validation` — close viewer when batch completes

**Run all 6 builds:**

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester
scripts\generate_all_renderer_truth_captures.bat
```

**Run one build manually:**

```powershell
$build = "3_3_5_12340"
$version = "3.3.5.12340"
$map = "Azeroth"
$root = "i:\parp\parp-tools\output\tmp\mdxviewer_validation_smoke\$build"

& "i:\parp\parp-tools\gillijimproject_refactor\src\MdxViewer\bin\Debug\net10.0-windows\ParpToolsWoWViewer.exe" `
  --game-path "i:/parp/parp-tools/output/tmp/wowarchive-clients/$build/World of Warcraft" `
  --build $version `
  --listfile "i:/parp/parp-tools/gillijimproject_refactor/test_data/community-listfile-withcapitals.csv" `
  --world "World\Maps\$map\$map.wdt" `
  --validation-dataset-root $root `
  --validation-output $root `
  --validation-resolution 512 `
  --force-validation-regeneration `
  --exit-after-validation
```

**Note:** Different builds use different map names and WDT paths. Check the
client's `World\Maps\` directory for available maps. Common maps:
- `Azeroth` (Kalimdor + Eastern Kingdoms)
- `Northrend`
- `Expansion01` (Outland)

### Step 3: Patch into Zarr stores

Reads the captured PNGs and writes `object_visibility_mask` and
`no_object_minimap` arrays into the V16 Zarr stores.

```powershell
uv run python scripts/build_v16_dataset.py patch-renderer-truth `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

### One-shot batch (all three steps after stubs)

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester
uv run python scripts/build_v16_dataset.py generate-viewer-stubs --all
scripts\generate_all_renderer_truth_captures.bat
uv run python scripts/build_v16_dataset.py patch-renderer-truth --all
```

### Future: Zarr-native viewer

The eventual goal is for the wow-viewer to read directly from V16 Zarr
stores instead of game client archives. The Zarr stores are compact and
fast compared to MPQ, and contain all the terrain/texture/object signals
already harvested. The viewer would only need the game client for
rendering (shaders, models, textures) while loading terrain geometry
directly from the store.

## Validate Dataset Signals

```powershell
uv run python scripts/build_v16_dataset.py validate-signals `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

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
- `output/datasets/v16/<build>.zarr/signal_validation.json`

## Trainer Readiness

```powershell
uv run python scripts/validate_v16_training_ready.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

## Build Curation Manifest

Dataset curation is now a separate layer between the V16 Zarr stores and the
trainers. The intent is to reject blank, nonsensical, or target-misaligned
tiles before any model sees them.

Current normal-oriented curation pass:

```powershell
uv run python -u scripts/build_v16_curation_manifest.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --profile normal_terrain_v16_1_1 `
  --workers -1 `
  --chunk-size 128 `
  --run-name normal_terrain_full_corpus_v16_1_1
```

Outputs:

- `output/datasets/v16/curation/<run-name>/summary.json`
- `output/datasets/v16/curation/<run-name>/tiles.parquet`
- `output/datasets/v16/curation/<run-name>/kept_tiles.parquet`
- `output/datasets/v16/curation/<run-name>/worst_cases.png`

For `normal_terrain_v16_1_1`, the curation layer now checks and scores:

- blank or low-signal minimaps
- normal coverage
- minimap-vs-normal edge agreement
- explicit blank genesis `what plate` tiles
- related low-signal reject cases
- deformation richness
- terrain-valid coverage
- painted alpha / MCLY presence
- per-tile difficulty buckets:
  - `easy`
  - `medium`
  - `hard`
  - `pathological`

`summary.json` now also publishes:

- `difficulty_bucket_counts`
- `difficulty_bucket_examples`
- `scouting_pool_recipe`

Curation runtime notes:

- `--workers -1` auto-resolves a CPU-friendly worker count
- `--workers 1` forces single-process behavior
- `--chunk-size` controls tile rows per worker task
- the builder now prints chunk progress per build so a long run is visibly alive

This is the intended rule for future model families too: build a target-aware
manifest first, then train from the curated tile set instead of raw Zarr rows.

## Train V16

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 4000 `
  --train-epoch-tiles 1350 `
  --val-max-tiles 150 `
  --batch-size 72 `
  --epochs 200 `
  --target-vram-gb 8 `
  --gpu-duty-cycle 100 `
  --val-interval 1 `
  --val-snapshots 8 `
  --val-snapshot-interval 1 `
  --run-name v16_full_corpus_epoch_rotation_qc1
```

Resume:

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 4000 `
  --train-epoch-tiles 1350 `
  --val-max-tiles 150 `
  --epochs 400 `
  --run-name v16_full_corpus_epoch_rotation_qc1 `
  --resume-from auto
```

When resuming, `--epochs` is the new total target, not "extra epochs from the checkpoint". If the checkpoint was written at epoch `200`, use something like `--epochs 400` to continue through epoch `400`.

### Training Notes

- `--train-max-tiles` is the persistent curated train pool.
- `--train-epoch-tiles` rotates a fresh per-epoch subset from that pool.
- `--epochs` is the total run ceiling. Resume starts at `checkpoint_epoch + 1` and stops when it reaches the requested total.
- `--curation-quality-profile basic` is the current default and filters obviously bad flat tiles before selection.
- `--num-workers -1` auto-resolves a CUDA-friendly worker count.
- `--gpu-duty-cycle 100` disables intentional step throttling.

### Validation Snapshot Behavior

- regular interval snapshots write to `models/v16/runs/<run>/validation/epoch_XXXX/`
- every new best `val_h` also writes a fresh random review set to `models/v16/runs/<run>/validation/best_epoch_XXXX/`

## Alpha/Minimap Audit

```powershell
uv run python scripts/audit_v16_alpha_minimap_alignment.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --max-tiles-per-build 24 `
  --worst-k 8
```

Outputs:

- `output/datasets/v16/validation/alpha_minimap_alignment/alpha_minimap_alignment.summary.json`
- `output/datasets/v16/validation/alpha_minimap_alignment/alpha_minimap_alignment.worst_cases.png`

Use this when validation tiles suggest the harvested supervision does not match
the baked minimap appearance.

## Renderer-Truth Validation Artifacts

The richer object-guidance lane now depends on bounded MdxViewer validation
captures in addition to the standard V16 inspection surfaces.

Current real proof roots:

- `output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48`
- `output/tmp/mdxviewer_validation_smoke/3_3_5_12340_Azeroth_30_48`
- `output/tmp/mdxviewer_validation_smoke_fix_wmo/3_3_5_12340_Azeroth_30_48`
- `output/tmp/mdxviewer_validation_smoke_heightfilter/3_3_5_12340_Azeroth_30_48`

These proofs currently establish:

- renderer-truth capture is working on the bounded `Azeroth_30_48` tile for
  `0_5_3_3368` and `3_3_5_12340`
- WMO near-camera culling needed a dedicated runtime fix before the later-build
  proofs were credible
- validation batches now wait longer before capture and can suppress very tall
  MDX clutter via a bounds-height threshold during the batch

They do not yet establish full six-build closure for the renderer-truth lane.

## Train V16.1 Normal With Curation

Use the curation manifest as an explicit trainer input.

Recommended optimized launch contract for a 16 GB card:

```powershell
uv run python -u scripts/train_v16_1_normal.py `
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
  --run-name v16_1_1_normal_curated_bs16_acc1_compile
```

Why this is the recommended starting point now:

- compile warmup is expensive, so judge throughput from epoch `2+`, not epoch `1`
- the bounded `400`-tile train pool plus `128`-tile rotating epochs avoids
  dragging the full curated manifest through every epoch
- if the current `8 x 1` launch sits under roughly `5 GB` VRAM, the repo truth
  is that the recommendation was too conservative for the 16 GB card
- prefer a larger micro-batch first so epoch wall-clock falls before reaching
  for accumulation tricks
- if `--autotune-batch-size` is enabled, the trainer now probes the pooled
  training set before the real run, writes `evidence/batch_autotune.json`, and
  can rescale `train_epoch_tiles` to preserve the original steps-per-epoch
  budget

Small scouting run for concept-mix proof before longer training:

```powershell
uv run python -u scripts/train_v16_1_normal.py `
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
  --epochs 20 `
  --num-workers 4 `
  --val-preview-interval 1 `
  --run-name v16_1_1_normal_curated_pool400_epoch128_bs16 `
  --no-compile
```

Use `--no-compile` for tiny scouting runs where compile warmup would dominate
the whole job. Leave compile on for longer runs once the pool/epoch sizing is
settled.

Fallback VRAM ladder:

- preferred start: `--batch-size 16 --grad-accum-steps 1`
- if the card still sits comfortably below saturation: `--batch-size 20 --grad-accum-steps 1`
- if the card still has real headroom after that: `--batch-size 24 --grad-accum-steps 1`
- first fallback: `--batch-size 12 --grad-accum-steps 1`
- second fallback: `--batch-size 8 --grad-accum-steps 1`
- if needed: `--batch-size 4 --grad-accum-steps 2`
- if needed: `--batch-size 2 --grad-accum-steps 4`
- safe floor: `--batch-size 1 --grad-accum-steps 8`

Use actual GPU memory and throughput, not old defaults:

- if VRAM stays under about `10 GB`, increase micro-batch again
- if throughput drops or compile becomes unstable, step back one rung
- keep `--grad-accum-steps 1` while the card can hold the larger micro-batch

V16.1 trainer runtime notes:

- `torch.compile` is enabled by default on CUDA
- `--no-compile` disables it for comparison or troubleshooting
- `--num-workers -1` auto-resolves a CUDA-friendly worker count
- `--curation-manifest` is the preferred path for normal training now
- `--bucket-sampling-profile v16_1_1_normal` over-indexes `hard` tiles while
  preserving `medium` / `easy` stability when those buckets are present
- `--target-vram-gb` now drives both per-epoch guidance logs and optional
  startup autotune
- `--autotune-batch-size` probes a batch-size ladder on the actual pooled train
  set and picks the largest safe batch under the effective VRAM target
- `--autotune-keep-epoch-steps` defaults on, so when batch-size grows the
  trainer rescales `train_epoch_tiles` to keep the original steps-per-epoch
  budget coherent
- `--normal-detail-boost` emphasizes terrain deformations over broad flats in
  the normal loss while still keeping flat tiles in the dataset
- hard-region weighting now also considers painted alpha / MCLY transitions and
  stays clipped by terrain-valid masking
- the normal trainer now also consumes raw supervision guidance channels from
  the V16 Zarr seam:
  - terrain-valid mask
  - object presence
  - painted alpha coverage
  - MCLY presence
  - blank `what plate` flag
- startup prints now show the effective batch, the curated pool sizes, and the
  curation manifest path
- best-model tracking is explicit again:
  - `v16_1_<task>_best.pt` stores `best_val` and `best_epoch`
  - validation preview PNGs now write only on new best checkpoints:
    - `validation/best_epoch_XXXX.png`
- autotune evidence now writes at:
  - `evidence/batch_autotune.json`
- new epoch evidence files:
  - `evidence/train_epoch_orders.jsonl`
  - `evidence/train_epoch_bucket_usage.jsonl`

Resume a curated normal run:

```powershell
uv run python -u scripts/train_v16_1_normal.py `
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
  --epochs 100 `
  --num-workers -1 `
  --val-preview-interval 2 `
  --run-name v16_1_1_normal_curated_bs16_acc1_compile `
  --resume-checkpoint ../models/v16_1/normal/runs/v16_1_1_normal_curated_bs16_acc1_compile/checkpoints/v16_1_normal_last.pt
```

## Train V16.2 (7-Channel Input)

V16.2 extends V16.1.1 with 7-channel input. The extra 4 channels are guidance
signals derived from the existing V16 store arrays:

| Channel | Signal | Source |
|---------|--------|--------|
| 0-2 | minimap RGB | Existing V16 store |
| 3 | object_filtered_mask | Existing V16 store (from patch-objects) |
| 4 | terrain_valid_mask_257 | Computed: normal_mask × (1 - object_presence) × (1 - liquid) |
| 5 | alpha_painted_256 | Computed: max(alpha channels 1-3) |
| 6 | mcly_any_16 (upsampled) | Computed: any MCLY layer > 0.05 |

No store mutation is needed — the guidance channels are computed on-the-fly
from arrays already in the V16 stores.

```powershell
uv run python -u scripts/train_v16_2_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --device auto `
  --batch-size 8 `
  --train-max-tiles 800 `
  --train-epoch-tiles 256 `
  --val-max-tiles 96 `
  --epochs 256 `
  --run-name v16_2_normal_all_builds_256ep
```

Resume:

```powershell
uv run python -u scripts/train_v16_2_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --device auto `
  --batch-size 8 `
  --epochs 512 `
  --resume-checkpoint ../models/v16_2/normal/runs/v16_2_normal_all_builds_256ep/checkpoints/v16_2_normal_last.pt `
  --run-name v16_2_normal_resume
```

Available V16.2 task wrappers:

- `train_v16_2_normal.py` — normal prediction (first consumer of guidance channels)
- `train_v16_2_height.py` — height prediction
- `train_v16_2_holes.py` — hole mask prediction
- `train_v16_2_liquid.py` — liquid mask + type prediction
- `train_v16_2_texcomp.py` — texture decomposition + recomposition

### Patch Renderer-Truth Masks (Optional)

If MdxViewer validation captures exist, you can patch renderer-truth arrays
into the stores. This is optional — training works without them.

```powershell
uv run python scripts/patch_v16_renderer_truth.py `
  --build 3_3_5_12340 `
  --capture-dir ../../output/tmp/mdxviewer_validation_smoke/3_3_5_12340_Azeroth_30_48 `
  --allow-zarr-write
```

This adds `object_visibility_mask` and `no_object_minimap` arrays to the store.
The V16.2 dataset loader reads them automatically when present.

V16.2 outputs go to `models/v16_2/<task>/runs/<run-name>/`.

## Key Outputs

- dataset stores: `output/datasets/v16/<build>.zarr`
- per-build visual QA: `output/datasets/v16/inspection/`
- validation reports: `output/datasets/v16/validation/`
- curation manifests: `output/datasets/v16/curation/<run-name>/`
- training runs: `models/v16/runs/<run-name>/`
- V16.1 training runs: `models/v16_1/<task>/runs/<run-name>/`
- V16.2 training runs: `models/v16_2/<task>/runs/<run-name>/`
- curation evidence: `models/v16/runs/<run-name>/evidence/`
