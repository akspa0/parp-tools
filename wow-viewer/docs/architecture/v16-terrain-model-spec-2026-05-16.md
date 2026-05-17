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
using blosc-zstd compression and standardized feature arrays. Data flows directly
from the C# harvester through a pipe into the Zarr writer — **no intermediate
NPZ files on disk**.

## Dataset Layout

```
wow-viewer/output/datasets/v16/
  <build_key>.zarr/              # One Zarr v3 LocalStore per client build
    zarr.json                    # Group metadata
    index.parquet                # Tile index: (tile_id, map, tx, ty, has_* flags, height stats)
    height_257/                  # (N, 257, 257) float32 — blosc-zstd-5
    normal_xyz/                  # (N, 257, 257, 3) float32
    normal_mask/                  # (N, 257, 257) bool
    alpha_256/                    # (N, 256, 256, 4) float32
    holes_16/                     # (N, 16, 16) bool
    liquid_mask/                  # (N, 256, 256) float32
    liquid_height/                # (N, 256, 256) float32
    object_mask/                  # (N, 257, 257) bool
    minimap_rgb/                  # (N, 256, 256, 3) uint8
    shadow_mask/                  # (N, 256, 256) float32
    mcly_texture_ids/             # (N, 16, 16, 4) int32
    mcly_layer_mask/              # (N, 16, 16, 4) float32
```

### Key Design Decisions

1. **Flat arrays indexed by tile number** — no per-tile files. The Parquet index
   maps row position → (map, tile_x, tile_y).
2. **Standardized feature set** — every tile has every array. Missing signals
   are stored as zero-filled arrays with a `has_<signal>` boolean column in
   the index. This eliminates per-sample feature-gating complexity.
3. **Blosc-zstd compression (level 5, bitshuffle)** — excellent compression
   ratio for the repetitive terrain data while keeping random-access fast.
4. **One Zarr store per build** — builds can be loaded independently or merged
   for cross-client training via `--builds`.
5. **Liquid data is mandatory when present** — `liquid_mask` and `liquid_height`
   are zero-filled for tiles without water, and `has_liquid_mask` in the index
   marks which tiles have real liquid data.
6. **No intermediate files** — the C# harvester streams NPZ blobs over a pipe
   directly to the Python builder. The Zarr store is the only on-disk artifact.

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
| `minimap_rgb` | 256×256×3 | uint8 | `minimap_rgb_256` | [0,255] RGB |
| `shadow_mask` | 256×256 | float32 | `mcsh_shadow_mask_256` | [0,1] |
| `mcly_texture_ids` | 16×16×4 | int32 | `mcly_texture_ids` | Texture layer IDs |
| `mcly_layer_mask` | 16×16×4 | float32 | `mcly_layer_mask` | Layer visibility |

### Derived Fields

- **`normal_mask`**: computed from `mcnr_normal_xyz` where
  `|nx| + |ny| + |nz| > 1e-6`. Zero-vectors (pad/gap vertices) are replaced
  with (0,0,1) before training and masked out of the normals loss.
- **Liquid arrays** are zero-filled for tiles where `unified_liquid_mask` is
  absent from the NPZ shard. The `has_liquid_mask` index column marks real data.

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
| `has_normal_xyz` | bool | normal_xyz present and nontrivial |
| `has_alpha_256` | bool | alpha_256 present |
| `has_holes_16` | bool | holes_16 present |
| `has_liquid_mask` | bool | unified_liquid_mask present with nonzero values |
| `has_shadow_mask` | bool | shadow_mask present |
| `has_mcly_texture_ids` | bool | mcly_texture_ids present |

## Build Pipeline

The V16 build pipeline has zero intermediate files. Data flows from the C#
harvester through a length-prefixed binary protocol over stdout directly into
the Python Zarr writer.

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
2. The harvester opens the MPQ archives, iterates tiles for each map, and writes
   length-prefixed NPZ blobs to stdout (format: 4-byte magic `NPZB` + 4-byte
   little-endian length + NPZ bytes). All diagnostics go to stderr.
3. The Python builder reads the binary stream, decodes each NPZ blob in memory,
   normalizes the arrays, and writes them into the Zarr store.
4. When the harvester finishes all tiles, it writes an `ENDS` sentinel.
5. The Python builder finalizes the Zarr store, writes the Parquet index, and
   trims arrays to the actual tile count.

**No temporary NPZ files are written to disk.** The only on-disk artifact is the
final Zarr store.

During builds, the Python side forwards harvester stderr live and prints
periodic progress lines with streamed tile counts, placement counts, raw NPZ
volume, and staged store size. The dataset is written to
`<build>.zarr.partial/` first and is only promoted to `<build>.zarr/` after
successful finalization, so interrupted runs do not silently poison the final
dataset path.
When `--maps` is omitted, the builder now calls `WowViewer.Tool.Harvest
discover-maps` and keeps only maps whose WDT summaries show terrain plus at
least one readable tile; pure WMO-only and zero-tile maps are skipped.

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
.\scripts\run-data-harvester-python.ps1 scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 4_0_0_11927
```

The `V16Dataset` class reads from Zarr stores, using the Parquet index for
train/val splitting and the `has_*` columns for feature masking.

Geometric augmentation (hflip/vflip/rot90) is applied at training time with
correct normal vector transforms. Per-tile height z-score normalization uses
`height_mean` and `height_std` from the index.

## Compression Benchmarks (target)

| Array | Uncompressed | Blosc-zstd-5 | Ratio |
|-------|-------------|-------------|-------|
| height_257 (257×257 f32) | 264 KB | ~50 KB | 5:1 |
| minimap_rgb (256×256×3 u8) | 196 KB | ~20 KB | 10:1 |
| alpha_256 (256×256×4 f32) | 1024 KB | ~80 KB | 13:1 |
| liquid_mask (256×256 f32) | 256 KB | ~5 KB | 50:1 |

Expected dataset size for one build (~5000 tiles): ~500 MB compressed.

## V16 Model

ConvNeXt V2 Nano encoder (15.6M, pretrained ImageNet) with U-Net skip fusion
decoder. Total ~27.4M parameters with the liquid head.

| Head | Output shape | Loss |
|------|-------------|------|
| height | (B, 1, 257, 257) | L1, object-masked |
| normals | (B, 3, 257, 257) | 1 - cosine similarity, normal-masked |
| alpha | (B, 4, 256, 256) | L1, object-masked × has_alpha |
| holes | (B, 1, 16, 16) | L1, object-masked × has_holes |
| liquid | (B, 1, 256, 256) | L1, object-masked × has_liquid |

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
| `scripts/build_v16_dataset.py` | Build pipeline: stream from harvester → Zarr |
| `scripts/run-data-harvester-python.ps1` | Repo-local launcher for `.venv` packages when the venv stub is broken |
| `src/harvester/v16_dataset.py` | PyTorch Dataset reading from Zarr stores |
| `src/harvester/v16_model.py` | V16Model (ConvNeXt V2 Nano + U-Net + liquid head) |
| `docs/architecture/v16-terrain-model-spec-2026-05-16.md` | This document |
