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

## 2. Run focused V18 height training

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/train_v18_focus.py height `
  --device cuda `
  --epochs 40 `
  --train-max-tiles 4096 `
  --train-epoch-tiles 2048 `
  --val-max-tiles 256 `
  --val-interval 1 `
  --run-name v18_height_focus_basic
```

## 3. Run focused V18 normal training

```powershell
cd i:\parp\parp-tools\wow-viewer\data-harvester

uv run python scripts/train_v18_focus.py normal `
  --device cuda `
  --epochs 40 `
  --train-max-tiles 4096 `
  --train-epoch-tiles 2048 `
  --val-max-tiles 256 `
  --val-interval 1 `
  --run-name v18_normal_focus_basic
```

## 4. Validate Python surfaces after changes

```powershell
cd i:\parp\parp-tools

uv run python -m py_compile `
  wow-viewer/data-harvester/scripts/build_v18_curation_manifest.py `
  wow-viewer/data-harvester/scripts/train_v18_focus.py `
  wow-viewer/data-harvester/scripts/train_v18.py `
  wow-viewer/data-harvester/scripts/train_v16_1_common.py `
  wow-viewer/data-harvester/src/harvester/v16_1_dataset.py
```

## Notes

- `train_v18_focus.py` defaults to:
  - dataset root: `wow-viewer/output/datasets/v18`
  - builds: `0_5_3_3368`, `3_3_5_12340`
  - latest focused `kept_tiles.parquet` if `--curation-manifest` is omitted
  - startup batch autotune against `--target-vram-gb 8`
  - strict near-equal per-build sampling by default; oversized pool/epoch requests
    auto-cap to the largest feasible balanced subset
- the focused curation manifest now rejects tiles with too little surviving
  trainable terrain, so liquid-hidden wipeouts stop entering the active pool
- the focused height and normal losses now honor terrain-valid masks, so
  liquid-hidden and object-hidden regions do not contribute loss
- the focused height and normal runs are independent by design
- quilt-level terrain stitching and later ADT writeback are downstream work;
  the training lane prepares the terrain predictions they consume
