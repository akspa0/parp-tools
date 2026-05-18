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

# Validate that V16Dataset/DataLoader/V15Model can consume the built store:
uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340

# Repair tile_x/tile_y in index.parquet from a metadata-only re-stream.
# This rewrites the index only; it does not touch the Zarr arrays:
uv run python scripts/build_v16_dataset.py repair-index --build 3_3_5_12340

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
- `scripts/validate_v16_training_ready.py` now provides a dedicated training-readiness proof surface: it opens the finalized stores, reads real samples through `V16Dataset`, checks a real `DataLoader` batch, and can run one `V15Model` forward pass on CPU so dataset validity is separated from trainer validity.
- `repair-index` can now rewrite `index.parquet` tile coordinates in place from a metadata-only re-stream of the staged client, so bad coordinate bookkeeping no longer forces a full dataset rebuild.
- Future harvest output now carries explicit `tile_x` / `tile_y` in `metadata.json`, and the builder trusts explicit metadata first instead of brittle `source_adt_path` parsing.

### Train V16

```bash
# First validate that the current trainer can read the built dataset:
uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340

# Then train:
uv run python scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 \
    --train-max-tiles 2000 \
    --val-max-tiles 256

# Resume the same run from latest checkpoint:
uv run python scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 \
    --train-max-tiles 2000 \
    --val-max-tiles 256 \
    --run-name <existing-run-name> \
    --resume-from auto

# Resume from best checkpoint instead:
uv run python scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 \
    --train-max-tiles 2000 \
    --val-max-tiles 256 \
    --run-name <existing-run-name> \
    --resume-from best
```

By default, curation excludes placeholder map labels (`map=memory`, empty,
unknown) so those rows are not selected into train/val subsets. Use
`--include-placeholder-map-tiles` only if you intentionally want them included.

Subset curation + evidence artifacts are written per run:
- `models/v16/runs/<run>/evidence/curation_manifest.json`
- `models/v16/runs/<run>/evidence/train_selection.jsonl`
- `models/v16/runs/<run>/evidence/val_selection.jsonl`
- `models/v16/runs/<run>/evidence/train_epoch_orders.jsonl`

Validation snapshot exports now also include one labeled overview image:
- `models/v16/runs/<run>/validation/epoch_XXXX/validation_overview.png`

Checkpoint files per run:
- `models/v16/runs/<run>/checkpoints/v16_last.pt` (written every epoch)
- `models/v16/runs/<run>/checkpoints/v16_best.pt` (best `val_h`)
- `models/v16/runs/<run>/checkpoints/v16_final.pt` (end-of-run snapshot)

Training-readiness validation writes:
- `wow-viewer/output/datasets/v16/validation/<build>.training_readiness.json`

It currently validates the signals that `train_v16.py` actually uses:
- `input`, `height`, `normals`, `normal_mask`, `alpha`, `holes`, `liquid`, `liquid_height`, `mcly_ids`, `mcly_mask`, `weight`

It also checks that `instance_mask` can still be read cleanly, but the current trainer does not yet use:
- `instance_mask`

### Infer V16 and Patch Terrain (Current Fast Path)

```bash
# 1) Run deterministic inference into a paired prediction store + patch-ready summaries.
uv run python scripts/infer_v16.py \
    --build 3_3_5_12340 \
    --checkpoint ../models/v16/runs/<run>/checkpoints/v16_best.pt

# 2) Patch LK ADTs from inference summaries.
dotnet run --project ../tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- \
    terrain-patch-adt \
    --input-adt-dir <staged-client-map-root> \
    --inference-dir ../output/datasets/v16_inference/<run-name>/patch_ready \
    --output-dir <patched-lk-output-root>

# 3) Optional: produce alphaWDT from the patched LK output.
dotnet run --project ../tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -- \
    convert-lk-to-alpha --input <patched-lk-output-root> --output <patched-output.wdt>
```

Outputs:
- prediction store: `wow-viewer/output/datasets/v16_inference/<run-name>/<build>.pred.zarr`
- patch-ready summaries: `wow-viewer/output/datasets/v16_inference/<run-name>/patch_ready/`
- patched LK terrain: `<patched-lk-output-root>`
- optional alpha output: `<patched-output.wdt>`

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
| normals | (B,3,257,257) | 1 - cosine, normal-masked + per-sample `has_normals` gate |
| alpha | (B,4,256,256) | L1, object-masked + per-sample `has_alpha` gate |
| holes | (B,1,16,16) | L1, object-masked + per-sample `has_holes` gate |
| liquid_mask | (B,1,256,256) | L1, object-masked + per-sample `has_liquid` gate |
| liquid_height | (B,1,256,256) | masked L1 on liquid-present pixels + per-sample `has_liquid` gate |
| mcly | (B,4,16,16,16 logits) | Cross-entropy, masked by `mcly_layer_mask` + per-sample `has_mcly` gate |

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
| `scripts/validate_v16_training_ready.py` | Validate that the current V16Dataset/DataLoader/model stack can consume finalized V16 stores |
| `scripts/run-data-harvester-python.ps1` | Repo-local launcher for `.venv` packages when the venv stub is broken |
| `scripts/train_v15.py` | V15 training script |
| `src/harvester/v16_dataset.py` | V16 PyTorch Dataset (Zarr) |
| `src/harvester/v15_dataset.py` | V15 PyTorch Dataset (NPZ) |
| `src/harvester/v15_model.py` | V15Model + V16Model (same architecture) |
