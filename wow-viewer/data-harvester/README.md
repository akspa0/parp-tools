# wow-viewer-data-harvester

Dataset generation, model training, and inference for WowViewer terrain AI.

## Quick Start

```bash
cd wow-viewer/data-harvester
uv sync
```

Use `uv run` as the normal entrypoint:

```powershell
uv run python -c "import sys; print(sys.executable)"
```

If a sandboxed agent session cannot reach the uv-managed AppData paths, the
repo-local launcher is still available as a fallback:

```powershell
.\scripts\run-data-harvester-python.ps1 -c "import sys; print(sys.executable)"
```

## V16 Dataset (Zarr)

The V16 dataset is a consolidated Zarr store per client build — no individual
NPZ shards. Data streams directly from the C# harvester into Zarr with no
intermediate files on disk.

### Build a V16 dataset

```bash
# Prerequisites: build the C# harvester first
dotnet build ../WowViewer.slnx -c Debug

# Single build (auto-discovered terrain maps):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340

# Specific maps:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --maps Azeroth Northrend

# Limit tiles (testing):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --limit 100

# Check stats:
uv run python scripts/build_v16_dataset.py stats --build 3_3_5_12340
```

Output: `wow-viewer/output/datasets/v16/<build>.zarr/`

Build behavior:
- Progress is printed during streaming, including tile counts, placement counts, raw streamed NPZ volume, and current staged store size.
- Harvester stderr is forwarded live with a `[harvest:<map>]` prefix.
- When `--maps` is not supplied, the builder asks `WowViewer.Tool.Harvest discover-maps` for a V16-driven map list and skips WMO-only, no-tile, and no-V16-usable-tile maps automatically.
- Builds stage into `wow-viewer/output/datasets/v16/<build>.zarr.partial/` and only replace the final `.zarr` store after successful finalization.
- If a discovered map still produces zero usable V16 tiles during streaming, the builder now warns and skips that map instead of aborting the whole build.
- Tiles dropped for missing required dataset keys are also written to `wow-viewer/output/datasets/v16/<build>.rejected_tiles.jsonl` so rejected coordinates and missing keys survive the console log.

### Train V16

```bash
.\scripts\run-data-harvester-python.ps1 scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340
```

### V16 Zarr Store Contents

Each `<build>.zarr/` contains:

| Array | Shape per tile | dtype | Description |
|-------|---------------|-------|-------------|
| `height_257` | 257×257 | float32 | Per-vertex height (world Z) |
| `normal_xyz` | 257×257×3 | float32 | Unit normals (zero-padded vertices masked) |
| `normal_mask` | 257×257 | bool | True = valid normal, False = pad/gap |
| `alpha_256` | 256×256×4 | float32 | MCAL blend weights [0,1] |
| `holes_16` | 16×16 | bool | Per-chunk hole flags |
| `liquid_mask` | 256×256 | float32 | Water presence [0,1], zero-filled if absent |
| `liquid_height` | 256×256 | float32 | Water surface Z, zero-filled if absent |
| `object_mask` | 257×257 | bool | True = object footprint |
| `minimap_rgb` | 256×256×3 | uint8 | Minimap image [0,255] |
| `shadow_mask` | 256×256 | float32 | MCSH shadow [0,1], zero-filled if absent |
| `mcly_texture_ids` | 16×16×4 | int32 | Texture IDs per layer, -1 if absent |
| `mcly_layer_mask` | 16×16×4 | float32 | Layer visibility, zero-filled if absent |

Plus `index.parquet` with columns: `tile_id`, `build`, `map`, `tile_x`,
`tile_y`, `height_mean`, `height_std`, and `has_*` flags for each signal.

Compression: blosc-zstd-5 with bitshuffle. Typical build (~5000 tiles): ~500 MB.

## V15 Dataset (NPZ shards, legacy)

Individual NPZ shards from `WowViewer.Tool.Harvest`. See
`docs/dataset-preparation-userguide.md` for the full NPZ signal reference.

## Model Architecture (V16)

ConvNeXt V2 Nano encoder (15.6M pretrained) + U-Net decoder with skip fusion.
~27.4M parameters total.

| Head | Output | Loss |
|------|---------|------|
| height | (B,1,257,257) | L1, object-masked |
| normals | (B,3,257,257) | 1 - cosine, normal-masked |
| alpha | (B,4,256,256) | L1, object-masked |
| holes | (B,1,16,16) | L1, object-masked |
| liquid | (B,1,256,256) | L1, object-masked |

## Training

```bash
# V16 (Zarr-based):
.\scripts\run-data-harvester-python.ps1 scripts/train_v16.py --builds 3_3_5_12340 --epochs 200

# V15 (NPZ-based, legacy):
.\scripts\run-data-harvester-python.ps1 scripts/train_v15.py --epochs 200
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/build_v16_dataset.py` | V16 build pipeline (harvester → Zarr, no temp files) |
| `scripts/run-data-harvester-python.ps1` | Repo-local launcher for `.venv` packages when the venv stub is broken |
| `scripts/train_v15.py` | V15 training script |
| `src/harvester/v16_dataset.py` | V16 PyTorch Dataset (Zarr) |
| `src/harvester/v15_dataset.py` | V15 PyTorch Dataset (NPZ) |
| `src/harvester/v15_model.py` | V15Model + V16Model (same architecture) |
| `src/harvester/v16_model.py` | Duplicate — use v15_model.py |
