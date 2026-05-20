# wow-viewer data-harvester

Minimal operator guide for V16 dataset + training workflows.

Detailed workflows, options, and troubleshooting live in:
- `data-harvester/docs/advanced-v16-workflows.md`

## Setup

```powershell
cd wow-viewer/data-harvester
uv sync
```

## Standard flow

1. Build or patch dataset stores.
2. Validate dataset signals.
3. Generate visual QA artifacts for human review.
4. Train only after visual + JSON validation passes.

## Build dataset (full)

```powershell
cd i:/parp/parp-tools/wow-viewer
dotnet build ./WowViewer.slnx -c Debug

cd i:/parp/parp-tools/wow-viewer/data-harvester
uv run python scripts/build_v16_dataset.py build `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --tile-workers 16 `
  --rebuild-existing
```

Default build keeps light Zarr chunk compression on: Blosc `lz4` level `1`
with `shuffle`.

Current finalized corpus status:
- six finalized stores currently exist for `0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, and `4_0_0_11927`
- all six current `signal_validation.json` files pass
- `0_7_0_3694` has the expected allowed warning for zero `has_holes_16` coverage

## Patch liquids only (no full rebuild)

```powershell
uv run python scripts/build_v16_dataset.py patch-liquids `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

## Validate dataset signals

```powershell
uv run python scripts/build_v16_dataset.py validate-signals `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

## Human visual QA (required before training)

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

## Train V16

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 1350 `
  --val-max-tiles 150 `
  --batch-size 72 `
  --target-vram-gb 8 `
  --gpu-duty-cycle 80 `
  --val-interval 1 `
  --val-snapshots 8 `
  --val-snapshot-interval 1 `
  --run-name v16_full_corpus_1500_val10
```

Resume:

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 1350 `
  --val-max-tiles 150 `
  --run-name v16_full_corpus_1500_val10 `
  --resume-from auto
```

## Key outputs

- Dataset stores: `output/datasets/v16/<build>.zarr`
- Training runs: `models/v16/runs/<run-name>`
- Validation snapshots: `models/v16/runs/<run-name>/validation/epoch_XXXX`
- Curation evidence: `models/v16/runs/<run-name>/evidence`
