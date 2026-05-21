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

1. Build or patch dataset stores.
2. Validate dataset signals.
3. Generate visual QA artifacts.
4. Build a target-aware curation manifest.
5. Run trainer-readiness validation.
6. Train only after the store passes JSON QA, human-eye QA, and curation.

## Build Dataset

```powershell
cd i:/parp/parp-tools/wow-viewer
dotnet build ./WowViewer.slnx -c Debug

cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run python scripts/build_v16_dataset.py build `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --tile-workers 16 `
  --rebuild-existing
```

Default V16 build compression is light Blosc:

- codec: `lz4`
- level: `1`
- shuffle: `shuffle`

## Patch Existing Stores

Liquids only:

```powershell
uv run python scripts/build_v16_dataset.py patch-liquids `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

Objects only:

```powershell
uv run python scripts/build_v16_dataset.py patch-objects `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

Coordinate repair only:

```powershell
uv run python scripts/build_v16_dataset.py repair-index --build 3_3_5_12340
```

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
  --profile normal_terrain_v1 `
  --workers -1 `
  --chunk-size 128 `
  --run-name normal_terrain_full_corpus_v1
```

Outputs:

- `output/datasets/v16/curation/<run-name>/summary.json`
- `output/datasets/v16/curation/<run-name>/tiles.parquet`
- `output/datasets/v16/curation/<run-name>/kept_tiles.parquet`
- `output/datasets/v16/curation/<run-name>/worst_cases.png`

For `normal_terrain_v1`, the curation layer checks:

- blank or low-signal minimaps
- normal coverage
- minimap-vs-normal edge agreement
- explicit blank genesis `what plate` tiles
- related low-signal reject cases

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

## Train V16.1 Normal With Curation

Use the curation manifest as an explicit trainer input.

Recommended optimized launch contract for a 16 GB card:

```powershell
uv run python -u scripts/train_v16_1_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v1 `
  --device auto `
  --batch-size 8 `
  --grad-accum-steps 1 `
  --train-max-tiles 400 `
  --train-epoch-tiles 128 `
  --val-max-tiles 48 `
  --epochs 50 `
  --num-workers -1 `
  --val-preview-interval 2 `
  --run-name v16_1_normal_curated_bs8_acc1_compile
```

Why this is the recommended starting point now:

- compile warmup is expensive, so judge throughput from epoch `2+`, not epoch `1`
- the bounded `400`-tile train pool plus `128`-tile rotating epochs avoids
  dragging the full curated manifest through every epoch
- `8 x 1` keeps the same effective batch as `1 x 8` but uses the card much
  more directly on the 16 GB card

Small scouting run for concept-mix proof before longer training:

```powershell
uv run python -u scripts/train_v16_1_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v1 `
  --device auto `
  --batch-size 8 `
  --grad-accum-steps 1 `
  --train-max-tiles 400 `
  --train-epoch-tiles 128 `
  --val-max-tiles 48 `
  --epochs 20 `
  --num-workers 4 `
  --val-preview-interval 1 `
  --run-name v16_1_normal_curated_pool400_epoch128 `
  --no-compile
```

Use `--no-compile` for tiny scouting runs where compile warmup would dominate
the whole job. Leave compile on for longer runs once the pool/epoch sizing is
settled.

Fallback VRAM ladder:

- preferred start: `--batch-size 8 --grad-accum-steps 1`
- if needed: `--batch-size 4 --grad-accum-steps 2`
- if needed: `--batch-size 2 --grad-accum-steps 4`
- safe floor: `--batch-size 1 --grad-accum-steps 8`

Optional higher-VRAM follow-ons if the card stays comfortable:

- `--batch-size 12 --grad-accum-steps 1`
- `--batch-size 16 --grad-accum-steps 1`

V16.1 trainer runtime notes:

- `torch.compile` is enabled by default on CUDA
- `--no-compile` disables it for comparison or troubleshooting
- `--num-workers -1` auto-resolves a CUDA-friendly worker count
- `--curation-manifest` is the preferred path for normal training now
- `--normal-detail-boost` emphasizes terrain deformations over broad flats in
  the normal loss while still keeping flat tiles in the dataset
- the normal trainer now also consumes raw supervision guidance channels from
  the V16 Zarr seam:
  - terrain-valid mask
  - object presence
  - painted alpha coverage
  - MCLY presence
  - blank `what plate` flag
- startup prints now show the effective batch, the curated pool sizes, and the
  curation manifest path

Resume a curated normal run:

```powershell
uv run python -u scripts/train_v16_1_normal.py `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --curation-manifest ../output/datasets/v16/curation/normal_terrain_full_corpus_v1 `
  --device auto `
  --batch-size 8 `
  --grad-accum-steps 1 `
  --train-max-tiles 400 `
  --train-epoch-tiles 128 `
  --val-max-tiles 48 `
  --epochs 100 `
  --num-workers -1 `
  --val-preview-interval 2 `
  --run-name v16_1_normal_curated_pool400_epoch128 `
  --resume-checkpoint ../models/v16_1/normal/runs/v16_1_normal_curated_pool400_epoch128/checkpoints/v16_1_normal_last.pt
```

## Key Outputs

- dataset stores: `output/datasets/v16/<build>.zarr`
- per-build visual QA: `output/datasets/v16/inspection/`
- validation reports: `output/datasets/v16/validation/`
- curation manifests: `output/datasets/v16/curation/<run-name>/`
- training runs: `models/v16/runs/<run-name>/`
- V16.1 training runs: `models/v16_1/<task>/runs/<run-name>/`
- curation evidence: `models/v16/runs/<run-name>/evidence/`
