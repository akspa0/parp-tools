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
4. Run trainer-readiness validation.
5. Train only after the store passes both JSON and human-eye QA.

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

## Train V16

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 4000 `
  --train-epoch-tiles 1350 `
  --val-max-tiles 150 `
  --batch-size 72 `
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
  --run-name v16_full_corpus_epoch_rotation_qc1 `
  --resume-from auto
```

### Training Notes

- `--train-max-tiles` is the persistent curated train pool.
- `--train-epoch-tiles` rotates a fresh per-epoch subset from that pool.
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

## Key Outputs

- dataset stores: `output/datasets/v16/<build>.zarr`
- per-build visual QA: `output/datasets/v16/inspection/`
- validation reports: `output/datasets/v16/validation/`
- training runs: `models/v16/runs/<run-name>/`
- curation evidence: `models/v16/runs/<run-name>/evidence/`
