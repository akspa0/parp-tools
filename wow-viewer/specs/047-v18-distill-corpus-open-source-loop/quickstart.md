# Quickstart: V18 Focused Two-Build Terrain Reconstruction System

## Preconditions

- focused V18 stores exist:
  - `wow-viewer/output/datasets/v18/0_5_3_3368.zarr`
  - `wow-viewer/output/datasets/v18/3_3_5_12340.zarr`
- Python environment is ready:

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester
uv sync
```

## 1. Build the focused V18 curation manifest

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/build_v18_curation_manifest.py `
  --run-name v18_focus_terrain_v1 `
  --workers -1 `
  --chunk-size 128
```

Expected output root:

- `wow-viewer/output/datasets/v18/curation/v18_focus_terrain_v1/`

Key artifacts:

- `summary.json`
- `tiles.parquet`
- `kept_tiles.parquet`

## 2. Run focused V18 full height training

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/train_v18_focus.py height `
  --device cuda `
  --epochs 40 `
  --curation-manifest ..\output\datasets\v18\curation\v18_focus_terrain_v1 `
  --train-bucket-rotation-fraction 0.10 `
  --val-max-tiles 32 `
  --val-interval 1 `
  --run-name v18_height_focus_full_v1
```

## 3. Run focused V18 full normal training

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/train_v18_focus.py normal `
  --device cuda `
  --epochs 40 `
  --curation-manifest ..\output\datasets\v18\curation\v18_focus_terrain_v1 `
  --train-bucket-rotation-fraction 0.10 `
  --val-max-tiles 32 `
  --val-interval 1 `
  --run-name v18_normal_focus_full_v1
```

## 4. Derive an optional smaller focused manifest

Use this when the main experiment is "keep the focused lane, but cut the
corpus far below the full 4096-row kept pool."

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/build_v18_tiny_manifest.py `
  --source-manifest ..\output\datasets\v18\curation\v18_focus_terrain_v1 `
  --samples-per-bucket-per-build 0 `
  --fraction-per-bucket-per-build 0.195 `
  --run-name v18_focus_tiny_800ish
```

Expected output root:

- `wow-viewer/output/datasets/v18/curation/v18_focus_tiny_800ish/`

Train that smaller scouting manifest with the same commands above, but swap:

- `--curation-manifest ..\output\datasets\v18\curation\v18_focus_tiny_800ish`
- `--train-bucket-rotation-fraction 1.0`
- run names such as `v18_height_focus_tiny_800ish` and
  `v18_normal_focus_tiny_800ish`

## 5. Run minimap-only focused inference proof

Run minimap-only focused inference proof:

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/infer_v18_focus.py `
  --build 3_3_5_12340 `
  --limit 8 `
  --device cuda `
  --height-checkpoint ..\models\v18\height\runs\v18_height_focus_full_v1\checkpoints\v16_1_height_best.pt `
  --normal-checkpoint ..\models\v18\normal\runs\v18_normal_focus_full_v1\checkpoints\v16_1_normal_best.pt `
  --run-name v18_focus_minimap_only_proof
```

This proof path consumes minimap RGB only during the run. It may still compare
outputs against offline dataset truth later, but those hidden tensors are not
part of the deployed forward pass.

## 6. Validate Python surfaces after changes

```powershell
cd i:\parp\parp-tools

uv run python -m py_compile `
  wow-viewer/data-harvester/scripts/build_v18_curation_manifest.py `
  wow-viewer/data-harvester/scripts/build_v18_tiny_manifest.py `
  wow-viewer/data-harvester/scripts/infer_v18_focus.py `
  wow-viewer/data-harvester/scripts/train_v18_focus.py `
  wow-viewer/data-harvester/scripts/train_v18.py `
  wow-viewer/data-harvester/scripts/train_v16_1_common.py `
  wow-viewer/data-harvester/src/harvester/v16_1_dataset.py `
  wow-viewer/data-harvester/src/harvester/test_v18_focus_masks.py `
  wow-viewer/data-harvester/src/harvester/test_v18_tiny_manifest.py
```

## Notes

- `train_v18_focus.py` defaults to:
  - dataset root: `wow-viewer/output/datasets/v18`
  - builds: `0_5_3_3368`, `3_3_5_12340`
  - latest focused `kept_tiles.parquet` if `--curation-manifest` is omitted
  - startup batch autotune against `--target-vram-gb 8`
  - restrained rotating bucket coverage via `--train-bucket-rotation-fraction 0.10`
  - strict near-equal per-build sampling by default; oversized pool/epoch requests
    auto-cap to the largest feasible balanced subset
  - early-stop patience `8` by default; `--epochs` is the ceiling, not a
    promise that every focused run will actually consume all epochs
- the current recommended full-session commands are the two `v18_focus_terrain_v1`
  runs above:
  - height: `v18_height_focus_full_v1`
  - normal: `v18_normal_focus_full_v1`
- `build_v18_tiny_manifest.py` derives a smaller balanced manifest from an
  existing focused `kept_tiles.parquet`; use `--fraction-per-bucket-per-build`
  for `800`-ish scouting subsets or `--samples-per-bucket-per-build` for
  ultra-tiny caps
- when bucket rotation is active, the per-epoch subset size is derived from the
  curated bucket counts; omit `--train-epoch-tiles` unless you intentionally
  want the older fixed-count epoch sampler
- when you intentionally train on a reduced manifest, pass
  `--train-bucket-rotation-fraction 1.0`; the tiny manifest itself is already
  the dataset throttle
- focused full height/base-normal runs now auto-apply a safer loader-pressure
  profile when `--num-workers` stays at `-1`; explicit `--num-workers`,
  `--prefetch-factor`, and `--persistent-workers` choices are preserved
- the focused curation manifest now rejects tiles with too little surviving
  trainable terrain, so liquid-hidden wipeouts stop entering the active pool
- the focused height and normal losses now honor terrain-valid masks, so
  liquid-hidden and object-hidden regions do not contribute loss
- trainer `val_loss` and preview images are offline supervised evaluation
  surfaces; they can use hidden truth/mask tensors for scoring, but they are
  not the deployment proof path
- `infer_v18_focus.py` is the focused minimap-only inference proof entrypoint
- the focused height and normal runs are independent by design
- quilt-level terrain stitching and later ADT writeback are downstream work;
  the training lane prepares the terrain predictions they consume
