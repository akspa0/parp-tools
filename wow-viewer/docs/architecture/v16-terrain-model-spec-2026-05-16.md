# V16 Terrain Model — Dataset & Pipeline Specification

## What Changed From V15

V15 used 23K+ individual NPZ files (one per tile), each containing a loose
bag of arrays with no standardized feature set. This created several problems:

- **No liquid data** in the NPZ shards (MH2O reader was not wired up for all builds).
- **No standardization**: different builds produce different feature subsets.
- **Extreme file count**: 23K tiny files cause I/O overhead, poor cache behavior,
  and are impossible to patch incrementally.
- **Redundant metadata**: each NPZ carries its own metadata.json, texture pixels,
  raw chunks — most of which the training loop never reads.

V16 replaces all of this with a single consolidated Zarr store per build,
using standardized feature arrays and a light default Blosc profile for new
builds. Data flows directly from the C# harvester through a lean raw binary
pipe into the Zarr writer — **no intermediate NPZ files on disk**.

## Dataset Layout

```
wow-viewer/output/datasets/v16/
  <build_key>.zarr/              # One Zarr v3 LocalStore per client build
    zarr.json                    # Group metadata
    index.parquet                # Tile index: (tile_id, map, tx, ty, has_* flags, height stats)
    height_257/                  # (N, 257, 257) float32
    normal_xyz/                  # (N, 257, 257, 3) float32
    normal_mask/                 # (N, 257, 257) bool
    alpha_256/                   # (N, 256, 256, 4) float32
    holes_16/                    # (N, 16, 16) bool
    liquid_mask/                 # (N, 256, 256) float32
    liquid_height/               # (N, 256, 256) float32
    object_mask/                 # (N, 257, 257) bool
    object_precise_mask/         # (N, 257, 257) float32
    object_instance_mask/        # (N, 257, 257) int32
    mcnk_flags_16/               # (N, 16, 16) int32
    mddf_mask/                   # (N, 257, 257) float32
    modf_mask/                   # (N, 257, 257) float32
    object_filtered_mask/        # (N, 257, 257) float32
    minimap_rgb/                 # (N, 256, 256, 3) uint8
    shadow_mask/                 # (N, 256, 256) float32
    mcly_texture_ids/            # (N, 16, 16, 4) int32
    mcly_layer_mask/             # (N, 16, 16, 4) float32
    placements.parquet           # Per-placement rows with asset-path linkage
    signal_validation.json       # Post-build signal/provenance validation report
```

### Key Design Decisions

1. **Flat arrays indexed by tile number** — no per-tile files. The Parquet index
   maps row position → (map, tile_x, tile_y).
2. **Standardized feature set** — every tile has every array. Missing signals
   are stored as zero-filled arrays with a `has_<signal>` boolean column in
   the index. This eliminates per-sample feature-gating complexity.
3. **Configurable Blosc compression** — future builds now default to
   `lz4` level `1` with `shuffle` for better write throughput, while older
   completed stores may still use `zstd` level `5` with `bitshuffle`.
4. **One Zarr store per build** — builds can be loaded independently or merged
   for cross-client training via `--builds`.
5. **Liquid data is mandatory when present** — `liquid_mask` and `liquid_height`
   are zero-filled for tiles without water, and `has_liquid_mask` in the index
   marks which tiles have real liquid data.
6. **No intermediate files** — the C# harvester streams lean `ARRY`
   length-prefixed tile blobs over a pipe directly to the Python builder. The
   Zarr store is the only canonical on-disk training artifact.
7. **Keep every fixed-shape training/loss signal** — the final Zarr store keeps
   the comprehensive terrain, liquid, masking, and MCLY supervision surfaces,
   while placement rows and liquid-source provenance live beside it in Parquet
   metadata rather than being discarded.

## Array Specifications

| Array | Shape per tile | dtype | Source NPZ key | Units / Range |
|-------|---------------|-------|-----------------|---------------|
| `height_257` | 257×257 | float32 | `height_257` | World Z, meters |
| `normal_xyz` | 257×257×3 | float32 | `mcnr_normal_xyz` | Unit vectors [-1,1] |
| `normal_mask` | 257×257 | bool | derived | True = valid vertex |
| `alpha_256` | 256×256×4 | float32 | `mcal_alpha_pack_256` | [0,1] blend weights |
| `holes_16` | 16×16 | bool | `hole_mask_16` | True = hole |
| `liquid_mask` | 256×256 | float32 | `unified_liquid_mask` | [0,1] water presence |
| `liquid_height` | 256×256 | float32 | `unified_liquid_height` | World Z, meters |
| `object_mask` | 257×257 | bool | `object_mask_257` | True = object placed |
| `object_precise_mask` | 257×257 | float32 | `object_precise_mask_257` | [0,1] placement coverage |
| `object_instance_mask` | 257×257 | int32 | `object_instance_mask_257` | 0 = terrain, 1+ = instance id |
| `mcnk_flags_16` | 16×16 | int32 | `mcnk_flags_16` | MCNK liquid/classification flags |
| `mddf_mask` | 257×257 | float32 | `mddf_mask_257` | Raw doodad mask |
| `modf_mask` | 257×257 | float32 | `modf_mask_257` | Raw WMO mask |
| `object_filtered_mask` | 257×257 | float32 | `object_filtered_mask_257` | Loss-gating mask after MDDF filtering |
| `minimap_rgb` | 256×256×3 | uint8 | `minimap_rgb_256` | [0,255] RGB |
| `shadow_mask` | 256×256 | float32 | `mcsh_shadow_mask_256` | [0,1] |
| `mcly_texture_ids` | 16×16×4 | int32 | `mcly_texture_ids` | Texture layer IDs |
| `mcly_layer_mask` | 16×16×4 | float32 | `mcly_layer_mask` | Layer visibility |

### Derived Fields

- **`normal_mask`**: computed from `mcnr_normal_xyz` where
  `|nx| + |ny| + |nz| > 1e-6`. Zero-vectors (pad/gap vertices) are replaced
  with (0,0,1) before training and masked out of the normals loss.
- **Liquid arrays** are derived by Python from the richer harvested liquid
  sources using the priority chain `MCNK flags > MCLQ > MH2O > WL* > none`.
  The raw harvester may emit `mh2o_*`, `mclq_*`, explicit presence masks, and
  `wl_liquid_*`; the finalized V16 store persists the unified training targets
  plus liquid-source provenance flags in `index.parquet`.
- **Object loss weighting** uses `object_filtered_mask`, not the raw merged
  object mask. The final terrain-loss weight seen by the trainer is
  `1.0 - object_filtered_mask`.

## Index (Parquet)

| Column | dtype | Description |
|--------|-------|-------------|
| `tile_id` | int64 | Row index into Zarr arrays |
| `build` | string | Build key (e.g. "3_3_5_12340") |
| `map` | string | Map name (e.g. "Azeroth", "Northrend") |
| `tile_x` | int32 | Tile X coordinate |
| `tile_y` | int32 | Tile Y coordinate |
| `height_mean` | float32 | Per-tile height mean (for z-score denormalization) |
| `height_std` | float32 | Per-tile height std (for z-score denormalization) |
| `n_mddf` | int32 | Number of MDDF placements exported for the tile |
| `n_modf` | int32 | Number of MODF placements exported for the tile |

In addition to the core columns above, the index carries:

- `has_<array>` boolean columns for every fixed-shape V16 array in the store:
  `height_257`, `normal_xyz`, `normal_mask`, `alpha_256`, `holes_16`,
  `liquid_mask`, `liquid_height`, `object_mask`, `object_precise_mask`,
  `object_instance_mask`, `mcnk_flags_16`, `mddf_mask`, `modf_mask`,
  `object_filtered_mask`, `minimap_rgb`, `shadow_mask`,
  `mcly_texture_ids`, `mcly_layer_mask`.
- Liquid provenance booleans:
  `has_liquid_source_mcnk`, `has_liquid_source_mh2o`,
  `has_liquid_source_mclq`, `has_liquid_source_unified`,
  `has_liquid_source_wl`.

## Build Pipeline

The V16 build pipeline has zero intermediate files. Data flows from the C#
harvester through a length-prefixed raw-binary protocol over stdout directly
into the Python Zarr writer.

### Prerequisites

1. Build the C# harvester:
   ```bash
   dotnet build wow-viewer/WowViewer.slnx -c Debug
   ```

2. Stage client data in `output/tmp/wowarchive-clients/<build>/World of Warcraft/`.

3. Install Python dependencies:
   ```bash
   cd wow-viewer/data-harvester
   uv sync
   ```

4. Run Python commands through `uv run`:
   ```powershell
   uv run python -c "import sys; print(sys.executable)"
   ```

5. If a sandboxed agent session cannot reach the uv-managed AppData paths, the
   repo-local launcher is still available as a fallback:
   ```powershell
   .\scripts\run-data-harvester-python.ps1 -c "import sys; print(sys.executable)"
   ```

### Build Commands

```bash
cd wow-viewer/data-harvester

# Single build (auto-discovered terrain maps):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340

# Resume an interrupted staged build:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --resume

# Force a rebuild even if the final store already looks complete:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --rebuild-existing

# Backfill _resume_state.json into older completed final stores:
uv run python scripts/backfill_v16_resume_state.py --builds 0_5_3_3368 0_5_5_3494 3_3_5_12340

# Generate summaries and sample sheets from existing stores:
uv run python scripts/inspect_v16_dataset.py --build 3_3_5_12340 --backfill-summary --write-images

# Validate that the current training stack can consume the built dataset:
uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340

# Repair tile_x/tile_y in index.parquet without rebuilding the tensor arrays:
uv run python scripts/build_v16_dataset.py repair-index --build 3_3_5_12340

# Multiple builds:
uv run python scripts/build_v16_dataset.py build --builds 3_3_5_12340 4_0_0_11927

# Specific maps only:
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --maps Azeroth Northrend

# Limit tiles (for testing):
uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --limit 100

# Check stats:
uv run python scripts/build_v16_dataset.py stats --build 3_3_5_12340
```

Output goes to `wow-viewer/output/datasets/v16/<build_key>.zarr/`.

### How It Works

1. The Python builder spawns the C# `WowViewer.Tool.Harvest harvest-stream` process.
2. The harvester opens the MPQ archives, keeps archive-backed ADT families
   (`root`, `_tex0`, `_obj0`) in memory, iterates tiles for each map, and writes
   length-prefixed raw blobs to stdout (default V16 format: 4-byte magic
   `ARRY` + 4-byte little-endian length + serialized tile payload). All
   diagnostics go to stderr.
3. The Python builder reads the binary stream, decodes each tile blob in memory,
   normalizes the arrays, and writes them into the Zarr store.
4. When the harvester finishes all tiles, it writes an `ENDS` sentinel.
5. The Python builder finalizes the Zarr store, writes the Parquet index, and
   trims arrays to the actual tile count.

**No temporary NPZ files are written to disk.** The only canonical on-disk artifact is the
final Zarr store.

During builds, the Python side forwards harvester stderr live and prints
periodic progress lines with streamed tile counts, placement counts, raw NPZ
volume, and staged store size. The dataset is written to
`<build>.zarr.partial/` first and is only promoted to `<build>.zarr/` after
successful finalization, so interrupted runs do not silently poison the final
dataset path. Interrupted builds can resume from the staged partial store with
`--resume`; completed maps are skipped from the saved `_resume_state.json`.
If `--resume` is passed before a build has actually written resume state yet,
the builder now falls back to a fresh staged build instead of aborting on a
missing `_resume_state.json`.
Successful final stores now retain `_resume_state.json` as completion metadata,
so future restart commands can recognize them as already finished.
Completed final stores are skipped by default on future build commands so
restarting one incomplete build does not silently rebuild the already-finished
ones; `--rebuild-existing` opts back into a full rebuild. Older completed final
stores can be backfilled with `_resume_state.json` using
`scripts/backfill_v16_resume_state.py`.
When `--maps` is omitted, the builder now calls `WowViewer.Tool.Harvest
discover-maps` and keeps only maps whose WDT/archive probe path can produce at
least one V16-usable tile (`height_257` + `minimap_rgb_256`). Pure WMO-only,
zero-tile, and "terrain but no V16-usable tile" maps are skipped.
If a discovered map still yields zero usable tiles during the full stream, the
builder warns and skips that map instead of aborting the whole dataset build.
Tiles dropped for missing required dataset keys are persisted to
`wow-viewer/output/datasets/v16/<build>.rejected_tiles.jsonl` so rejected
coordinates and missing keys survive the console log.
On Windows, transient Zarr `LocalStore` atomic-replace failures (`WinError 5`
or `WinError 32`) during chunk writes are now retried with bounded backoff
instead of aborting the entire dataset build on the first hit.
The Python writer now also buffers tiles in memory and flushes them to Zarr in
small first-dimension batches instead of one tile-row assignment at a time,
reducing chunk rewrite churn and filesystem pressure.
Before batch flush, incoming fixed-shape arrays are coerced to the canonical
dataset shapes so variable layer-count payloads do not break batch stacking.
Placement-heavy tiles still carry real per-placement mask painting cost, but
the builder no longer reparses the same placement catalog twice per tile for
masks and placement-array export.
The `stats` command now reports logical raw array size versus on-disk Zarr
size, including per-array compression ratios and whole-store savings.
The dedicated `validate_v16_training_ready.py` command now answers a separate
question: can the current `V16Dataset`, `DataLoader`, and `V16Model` actually
consume the finalized store without shape, dtype, or finite-value surprises.
If older final stores have bad `tile_x` / `tile_y` bookkeeping, the
`repair-index` command now rewrites `index.parquet` in place from a
metadata-only re-stream of the staged client. This leaves the Zarr arrays and
placements table untouched.
Future streamed NPZ metadata now carries explicit `tile_x` / `tile_y`, and the
Python builder trusts those explicit fields before falling back to `tile_name`
or `source_adt_path` parsing.
For legacy Alpha blobs that lacked explicit coordinates, fallback parsing now
also recognizes `#alpha-tile(x,y)` markers, so quilt coordinates can still be
recovered deterministically.

### Streaming Protocol

```
[4 bytes: "NPZB"] [4 bytes: length LE uint32] [length bytes: NPZ data]
[4 bytes: "NPZB"] [4 bytes: length LE uint32] [length bytes: NPZ data]
...
[4 bytes: "ENDS"] [4 bytes: 0x00000000]
```

All stdout output from the harvester in streaming mode is this binary protocol.
All diagnostic text goes to stderr.

## Training

```bash
cd wow-viewer/data-harvester
uv run python scripts/validate_v16_training_ready.py --build 3_3_5_12340
uv run python scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 4_0_0_11927 \
    --train-max-tiles 2000 \
    --val-max-tiles 256

# resume existing run from latest checkpoint
uv run python scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 4_0_0_11927 \
    --train-max-tiles 2000 \
    --val-max-tiles 256 \
    --run-name <existing-run-name> \
    --resume-from auto
```

The `V16Dataset` class reads from Zarr stores, using the Parquet index for
train/val splitting and the `has_*` columns for feature masking.

Geometric augmentation (hflip/vflip/rot90) is applied at training time with
correct normal vector transforms. Per-tile height z-score normalization uses
`height_mean` and `height_std` from the index.

Trainer-side subset curation is now first-class. If `--train-max-tiles` or
`--val-max-tiles` is set, `train_v16.py` samples a deterministic no-replacement
subset from the split and records chain-of-evidence artifacts under:

- `models/v16/runs/<run>/evidence/curation_manifest.json`
- `models/v16/runs/<run>/evidence/train_selection.jsonl`
- `models/v16/runs/<run>/evidence/val_selection.jsonl`
- `models/v16/runs/<run>/evidence/train_epoch_orders.jsonl`

Curation excludes placeholder map labels (`memory`, `<memory>`, empty,
unknown) by default so bad metadata rows are not selected; pass
`--include-placeholder-map-tiles` to opt in.

Validation snapshots now include one labeled composite overview image per
validation epoch:

- `models/v16/runs/<run>/validation/epoch_XXXX/validation_overview.png`

Checkpoint policy:

- `models/v16/runs/<run>/checkpoints/v16_last.pt` is written every epoch.
- `models/v16/runs/<run>/checkpoints/v16_best.pt` tracks best `val_h`.
- `models/v16/runs/<run>/checkpoints/v16_final.pt` is written at run end.

The training-readiness validator writes:

- `wow-viewer/output/datasets/v16/validation/<build>.training_readiness.json`

It validates the signals the current trainer actually uses:

- `input`, `height`, `normals`, `normal_mask`, `alpha`, `holes`, `liquid`, `mcly_ids`, `mcly_mask`, `weight`

`weight` is derived from `object_filtered_mask` when available, falling back to
`1.0 - object_mask` only for legacy stores that predate filtered masking.

It also proves that `instance_mask` remains readable by the dataset layer, but
the current trainer still does not supervise:

- `instance_mask`

## Training Contract Matrix (Spec vs Code)

This section is the executable truth surface for first training attempts.
Every expected item below maps to a concrete file in `data-harvester/`.

### Core training path (must exist)

| Expected surface | Implemented in | Status |
|---|---|---|
| Zarr dataset loader (`<build>.zarr`, `index.parquet`) | `src/harvester/v16_dataset.py` | Implemented |
| Model used by V16 trainer | `src/harvester/v16_model.py` (`V16Model`) | Implemented |
| V16 trainer entrypoint | `scripts/train_v16.py` | Implemented |
| Training-readiness gate | `scripts/validate_v16_training_ready.py` | Implemented |
| Inference bridge for post-train patch flow | `scripts/infer_v16.py` | Implemented |

### Targets currently supervised by `train_v16.py`

| Target | Source tensor key from `V16Dataset` | Loss path | Status |
|---|---|---|---|
| Height (257x257) | `height` | weighted L1 | Implemented |
| Normals (257x257x3) | `normals`, `normal_mask` | cosine loss, per-sample `has_normals` mask | Implemented |
| Alpha (256x256x4) | `alpha` | weighted L1, per-sample `has_alpha` mask | Implemented |
| Holes (16x16) | `holes` | weighted L1, per-sample `has_holes` mask | Implemented |
| Liquid mask (256x256) | `liquid` | weighted L1, per-sample `has_liquid` mask | Implemented |
| MCLY classes | `mcly_ids`, `mcly_mask` | masked cross-entropy, per-sample `has_mcly` mask | Implemented |

### Signals present in V16 dataset but not yet supervised

| Signal | Availability | Current use |
|---|---|---|
| `mcnk_flags_16` | present in Zarr builder | liquid provenance / QA signal, not read by current terrain trainer |
| `mddf_mask` | present in Zarr builder | QA / future object-loss experiments |
| `modf_mask` | present in Zarr builder | QA / future object-loss experiments |
| `object_filtered_mask` | present in Zarr builder | consumed indirectly as terrain-loss `weight`; not a standalone model target |
| `object_instance_mask` / dataset `instance_mask` | present in Zarr + dataset loader | readable, not yet in loss |
| `object_precise_mask` | present in Zarr builder | not consumed by `V16Dataset` yet |
| `liquid_height` | present in Zarr + dataset loader | intentionally deferred (future liquid-refinement model) |
| `shadow_mask` | present in Zarr builder | archived auxiliary signal; not consumed by default terrain losses |

### Planned Liquid Refinement Model (separate lane)

The project should keep terrain reconstruction and liquid reconstruction in
separate models:

- Terrain lane (current V16): optimize terrain geometry/material channels and
  use object/liquid masks as exclusion weighting for terrain losses.
- Liquid lane (planned): optimize liquid placement + liquid height quality from
  minimap-centric inputs.

Planned liquid-lane contract:

- Inputs:
  - `minimap_rgb`
  - optional liquid priors (`liquid_mask`, WL* hints, other map-level liquid cues)
- Targets:
  - `liquid_mask`
  - `liquid_height` (supervised only where liquid is present)
- Outputs:
  - `liquid_pred_mask_256`
  - `liquid_pred_height_256`

Boundary rule:

- Terrain model should not own liquid-height fidelity.
- Liquid model should not own terrain geometry.
- Terrain training continues to use liquid/object masks as loss gating so
  non-terrain pixels do not contaminate terrain supervision.

### Explicit note on model file naming

- Canonical V16 model module: `src/harvester/v16_model.py`.
- Legacy `src/harvester/v15_model.py` now exists only as a compatibility shim
  for older V15 imports.

## Compression Benchmarks (illustrative)

| Array | Uncompressed | Light Blosc (`lz4`/1/`shuffle`) | Expected trend |
|-------|-------------|-------------|-------|
| height_257 (257×257 f32) | 264 KB | materially smaller | moderate compression |
| minimap_rgb (256×256×3 u8) | 196 KB | somewhat smaller | limited compression |
| alpha_256 (256×256×4 f32) | 1024 KB | materially smaller | good compression |
| liquid_mask (256×256 f32) | 256 KB | much smaller on sparse tiles | very good compression |

The important contract is qualitative, not these exact numbers: future builds
should keep light Zarr chunk compression on, while avoiding the old per-tile
NPZ/zip compression overhead.

## V16 Model

ConvNeXt V2 Nano encoder (15.6M, pretrained ImageNet) with U-Net skip fusion
decoder.

| Head | Output shape | Loss |
|------|-------------|------|
| height | (B, 1, 257, 257) | L1, object-masked |
| normals | (B, 3, 257, 257) | 1 - cosine similarity, normal-masked |
| alpha | (B, 4, 256, 256) | L1, object-masked × has_alpha |
| holes | (B, 1, 16, 16) | L1, object-masked × has_holes |
| liquid_mask | (B, 1, 256, 256) | L1, object-masked × has_liquid |
| mcly | (B, 4, 16, 16, 16 logits) | Cross-entropy, masked by `mcly_layer_mask` |

## Normalization

- **Height**: z-score per tile: `h' = (h - mean) / (std + 1e-8)`. mean and std
  stored in the Parquet index for denormalization at inference.
- **Normals**: unit vectors in [-1, 1]. Zero vectors masked.
- **Alpha**: raw [0, 1].
- **Liquid mask**: raw [0, 1].
- **Minimap RGB**: [0, 255] → [0, 1] at training time.

## Files

| File | Purpose |
|------|---------|
| `scripts/build_v16_dataset.py` | Build pipeline: stream from harvester → Zarr, resume, rejected-tile reporting |
| `scripts/backfill_v16_resume_state.py` | Backfill `_resume_state.json` into older completed final stores |
| `scripts/inspect_v16_dataset.py` | Backfill `_dataset_summary.json` and sample visualizations from existing stores |
| `scripts/validate_v16_training_ready.py` | Validate dataset readability through the current V16Dataset/DataLoader/model stack |
| `scripts/infer_v16.py` | Deterministic V16 inference to `<build>.pred.zarr` + patch-ready summaries |
| `scripts/run-data-harvester-python.ps1` | Repo-local launcher for `.venv` packages when the venv stub is broken |
| `src/harvester/v16_dataset.py` | PyTorch Dataset reading from Zarr stores |
| `src/harvester/v16_model.py` | Current V16 terrain model implementation (`V16Model`) |
| `src/harvester/v15_model.py` | Compatibility shim for legacy V15 imports |
| `docs/architecture/v16-terrain-model-spec-2026-05-16.md` | This document |

## Inference Contract (Paired Input/Output Stores)

V16 training stores are now the canonical **input** dataset contract.
Inference must write a matching **output** dataset contract so every input tile
has a deterministic predicted counterpart and can flow into ADT patch tooling.

### Input Store (existing)

- Path: `wow-viewer/output/datasets/v16/<build>.zarr/`
- Authority:
  - `index.parquet` (`tile_id`, `build`, `map`, `tile_x`, `tile_y`)
  - `placements.parquet` (placement rows + asset paths)
  - fixed arrays (`height_257`, `normal_xyz`, `alpha_256`, and others)

### Output Store (new requirement)

- Path (target): `wow-viewer/output/datasets/v16_inference/<run_name>/<build>.pred.zarr/`
- Rule: `index.parquet` is copied from the input store with identical row order
  and identical `tile_id` values.
- Rule: output arrays are one-to-one with input rows (same `N`).
- Rule: output stores are append-forbidden after finalization; rebuild to change.

Required prediction arrays per tile:

| Array | Shape per tile | dtype | Notes |
|-------|----------------|-------|------|
| `height_pred_257` | 257x257 | float32 | denormalized world Z |
| `normal_pred_xyz` | 257x257x3 | float32 | normalized to unit length |
| `alpha_pred_256` | 256x256x4 | float32 | clamped [0,1] |
| `holes_pred_16` | 16x16 | float32 | [0,1] probability |
| `liquid_pred_mask_256` | 256x256 | float32 | [0,1] probability |
| `mcly_pred_logits_16x16x4x16` | 16x16x4x16 | float32 | raw logits for layer-class choices |

Required inference metadata sidecar:

- `_inference_run.json` with:
  - `model_version`, `checkpoint_path`, `checkpoint_sha256`
  - `dataset_build`, `input_store_path`, `input_index_sha256`
  - `seed`, `device`, `torch_version`, `started_at`, `finished_at`
  - `amp_enabled`, `compile_enabled`

### Determinism Rules

For "same input -> same output" guarantees, inference runs must:

1. Use `model.eval()` and disable data augmentation.
2. Set fixed seeds for Python/NumPy/Torch and log them.
3. Disable any random sampling or stochastic post-processing.
4. Keep a single checkpoint for one run.
5. Emit an input index hash and checkpoint hash into `_inference_run.json`.

If checkpoint hash + input index hash + seed are identical, output bytes should
be treated as expected-to-match for the same hardware/software stack.

## Reconstruction Contract (ADT/WDT Tooling)

Inference output stores are not just visualization artifacts; they are the
source contract for terrain patch/synthesis tooling.

### Mode A: Patch Existing ADTs

Goal: apply predicted terrain channels onto an existing staged ADT while
preserving non-terrain structures unless explicitly replaced.

Patch input contract:

- source staged client root under `output/tmp/wowarchive-clients/`
- source map/tile coordinates from input/output index rows
- prediction arrays from `<build>.pred.zarr`
- patch policy (`replace_height`, `replace_normals`, `replace_alpha`,
  `replace_holes`, `replace_liquid`)

Patch output contract:

- LK split-ADT files under a bounded output root
- patch report per tile (`old_hash`, `new_hash`, replaced channels)

### Mode B: Synthesize New Terrain Tiles

Goal: create brand-new terrain outputs from model predictions:

- LK ADT families (`root`, `_tex0`, `_obj0`) for new map regions
- or Alpha tile content routed through existing alpha writer surfaces
  (without changing `AlphaWdtWriter` semantics)

Synthesis output contract:

- generated tile files
- emitted WDT/WDL metadata required by target client format
- synthesis report (`tile count`, `format`, `channel provenance`)

### Tooling Surfaces (Current + Remaining)

Already implemented and usable now:

- `wow-viewer/data-harvester/scripts/infer_v16.py`
  - runs deterministic V16 inference
  - emits `<build>.pred.zarr`
  - emits per-tile `inference_summary.json` + `predicted_height_257.npy` for patch tooling
- `WowViewer.Tool.Converter terrain-patch-adt`
  - patches LK ADT terrain (MCVT/MCNR) from per-tile inference summaries
- `WowViewer.Tool.Converter convert-lk-to-alpha`
  - converts LK terrain outputs to alphaWDT outputs
- `WowViewer.Tool.Converter convert-alpha-to-lk`
  - converts alphaWDT outputs back to LK terrain outputs

Still missing (future ergonomic wrapper work):

- a single command that consumes `.pred.zarr` directly without the per-tile
  summary staging convention
- a one-shot "infer + patch + optional alpha conversion" pipeline command
- direct ADT liquid chunk patching in the converter path (`MH2O`/`MCLQ` write
  integration) so predicted liquid mask+height are emitted into terrain outputs

## Input/Output Pairing Policy

Every training build should have an optional paired inference build:

- Input: `v16/<build>.zarr`
- Output: `v16_inference/<run_name>/<build>.pred.zarr`

This pairing allows:

- tile-level regression checks (`gt` vs `pred`)
- deterministic replay of model outputs
- direct downstream ADT patch or synthesis workflows

Ground-truth training stores remain immutable. Inference stores are versioned by
`run_name` so multiple checkpoints can be compared without mutating the source
dataset.
