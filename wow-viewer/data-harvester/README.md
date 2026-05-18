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
intermediate NPZ files on disk, and archive-backed ADT families now stay in
memory instead of being staged through `%TEMP%`.

### Build a V16 dataset

```bash
# Prerequisites: build the C# harvester first
dotnet build ../WowViewer.slnx -c Debug

# Single build (auto-discovered terrain maps):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340

# Resume an interrupted staged build:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --resume

# Force a rebuild even if the final store already looks complete:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --rebuild-existing

# Backfill _resume_state.json into older completed final stores:
uv run python scripts/backfill_v16_resume_state.py --builds 0_5_3_3368 0_5_5_3494 3_3_5_12340

# Generate human-friendly summaries and sample sheets from existing stores:
uv run python scripts/inspect_v16_dataset.py --build 3_3_5_12340 --backfill-summary --write-images

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
- Completed final stores (`<build>.zarr/`) are now skipped by default so restart commands do not silently rebuild already-finished builds. Use `--rebuild-existing` to force a rebuild.
- Builds stage into `wow-viewer/output/datasets/v16/<build>.zarr.partial/` and only replace the final `.zarr` store after successful finalization.
- Interrupted builds can resume from `wow-viewer/output/datasets/v16/<build>.zarr.partial/` with `--resume`; completed maps are skipped from the saved `_resume_state.json`.
- If `--resume` is passed before a build has actually written resume state yet, the builder now falls back to a fresh staged build instead of aborting on a missing `_resume_state.json`.
- Successful final stores now keep `_resume_state.json` as completion metadata, so future restart commands can recognize them as finished without rebuilding.
- If a discovered map still produces zero usable V16 tiles during streaming, the builder now warns and skips that map instead of aborting the whole build.
- Tiles dropped for missing required dataset keys are also written to `wow-viewer/output/datasets/v16/<build>.rejected_tiles.jsonl` so rejected coordinates and missing keys survive the console log.
- Future builds now default to a faster Blosc profile: `lz4`, compression level `1`, `shuffle`. Older finished stores using `zstd` remain valid.
- `scripts/backfill_v16_resume_state.py` can add `_resume_state.json` to older completed final stores so their completion metadata matches the new format.
- On Windows, transient `WinError 5` / `WinError 32` chunk-write races in Zarr `LocalStore` are now retried with bounded backoff instead of aborting the whole build immediately.
- Tile writes are now buffered in memory and flushed to Zarr in small slice batches instead of one row at a time, which should reduce chunk rewrite churn and improve throughput on larger maps.
- Incoming fixed-shape signals are now coerced to their canonical Zarr shapes before batching, so variable layer-count payloads do not break `np.stack(...)` during resume/build runs.
- Placement-heavy tiles still cost more than empty terrain because object masks are painted per placement, but the builder no longer reparses the same placement catalog twice per tile for masks and placement arrays.
- `stats` now reports logical raw array size versus on-disk Zarr size, including per-array ratios and whole-store savings, so compression wins are visible instead of inferred.

### Train V16

```bash
uv run python scripts/train_v16.py \
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

Compression for new builds defaults to blosc-lz4-1 with shuffle. Existing
older stores may still use blosc-zstd-5 with bitshuffle.

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
uv run python scripts/train_v16.py --builds 3_3_5_12340 --epochs 200

# V15 (NPZ-based, legacy):
uv run python scripts/train_v15.py --epochs 200
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/build_v16_dataset.py` | V16 build pipeline (harvester → Zarr, resume, no archive temp staging) |
| `scripts/backfill_v16_resume_state.py` | Backfill `_resume_state.json` into older completed final stores |
| `scripts/inspect_v16_dataset.py` | Backfill `_dataset_summary.json` and sample visualizations from existing V16 stores |
| `scripts/run-data-harvester-python.ps1` | Repo-local launcher for `.venv` packages when the venv stub is broken |
| `scripts/train_v15.py` | V15 training script |
| `src/harvester/v16_dataset.py` | V16 PyTorch Dataset (Zarr) |
| `src/harvester/v15_dataset.py` | V15 PyTorch Dataset (NPZ) |
| `src/harvester/v15_model.py` | V15Model + V16Model (same architecture) |
| `src/harvester/v16_model.py` | Duplicate — use v15_model.py |
