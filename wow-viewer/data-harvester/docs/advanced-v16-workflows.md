# Advanced V16 Workflows

This is the detailed companion doc for `data-harvester/README.md`.

## Dataset commands

### Build

```powershell
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --resume
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --rebuild-existing
```

Useful options:
- `--codec {none,lz4,zstd}` (default `none`)
- `--clevel <int>`
- `--shuffle {noshuffle,shuffle,bitshuffle}`
- `--signal-validation/--no-signal-validation`
- `--signal-validation-strict/--no-signal-validation-strict`

### Patch liquids in-place

```powershell
uv run python scripts/build_v16_dataset.py patch-liquids --build 0_5_3_3368
uv run python scripts/build_v16_dataset.py patch-liquids --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

Notes:
- Rewrites only `liquid_mask`, `liquid_height`, and liquid `has_*` flags in `index.parquet`.
- Creates `index.parquet.bak.liquids` unless `--no-backup`.
- Writes `liquid_patch_report.json`.

### Validate signal coverage

```powershell
uv run python scripts/build_v16_dataset.py validate-signals --build 3_3_5_12340
uv run python scripts/build_v16_dataset.py validate-signals --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927
```

### Inspect dataset

```powershell
uv run python scripts/inspect_v16_dataset.py --build 3_3_5_12340 --sample-count 24 --sample-seed 1337 --sample-mode random --write-overview
```

Sampling modes:
- `random`
- `linspace`
- `liquid_focus`

Artifacts:
- `<build>.summary.json`
- `<build>.samples.json`
- `<build>.validation_audit_overview.png`

## Training commands

### Basic

```powershell
uv run python scripts/train_v16.py --dataset-dir ../output/datasets/v16 --builds 3_3_5_12340
```

### Full-corpus curation

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 1350 `
  --val-max-tiles 150
```

### Resume

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --run-name <run-name> `
  --resume-from auto
```

### Thermal/VRAM tuning

```powershell
uv run python scripts/train_v16.py `
  --dataset-dir ../output/datasets/v16 `
  --builds 0_5_3_3368 0_5_5_3494 0_7_0_3694 3_0_1_8303 3_3_5_12340 4_0_0_11927 `
  --train-max-tiles 1350 `
  --val-max-tiles 150 `
  --batch-size 72 `
  --num-workers 6 `
  --persistent-workers `
  --prefetch-factor 4 `
  --target-vram-gb 8 `
  --gpu-duty-cycle 80
```

## Evidence and outputs

- Dataset metrics: `output/datasets/v16/<build>.zarr/harvest_metrics.json`
- Dataset signal gates: `output/datasets/v16/<build>.zarr/signal_validation.json`
- Training config/logs: `models/v16/runs/<run>/config.json`, `training_log.json`
- Curation evidence: `models/v16/runs/<run>/evidence/`
- Validation snapshots: `models/v16/runs/<run>/validation/epoch_XXXX/`
