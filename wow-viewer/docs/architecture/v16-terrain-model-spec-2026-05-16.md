# V16 Terrain Model — Dataset Specification

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
using blosc-zstd compression and standardized feature arrays.

## Dataset Layout

```
wow-viewer/output/datasets/v16/
  <build_key>.zarr/              # One Zarr v3 LocalStore per client build
    zarr.json                    # Group metadata
    index.parquet                # Tile index: (tile_id, map, tx, ty)
    height_257/                  # (N, 257, 257) float32
    normal_xyz/                  # (N, 257, 257, 3) float32
    normal_mask/                  # (N, 257, 257) bool
    alpha_256/                    # (N, 256, 256, 4) float32
    holes_16/                     # (N, 16, 16) bool
    liquid_mask/                  # (N, 256, 256) float32
    liquid_height/                # (N, 256, 256) float32
    object_mask/                  # (N, 257, 257) bool
    minimap_rgb/                  # (N, 256, 256, 3) uint8
    shadow_mask/                  # (N, 256, 256) float32  [optional]
    mcly_texture_ids/             # (N, 16, 16, 4) int32   [optional]
    mcly_layer_mask/              # (N, 16, 16, 4) float32 [optional]
```

### Key Design Decisions

1. **Flat arrays indexed by tile number** — no per-tile files. The index maps
   row position → (map, tile_x, tile_y).
2. **Standardized feature set** — every tile has every array. Missing signals
   are stored as zero-filled arrays with a `has_<signal>` boolean column in
   the index. This eliminates per-sample feature-gating complexity.
3. **Blosc-zstd compression (level 5, bitshuffle)** — excellent compression
   ratio for the repetitive terrain data while keeping random-access fast.
4. **Optional arrays** — `shadow_mask`, `mcly_texture_ids`, `mcly_layer_mask`
   are stored when available but not required. The index marks availability.
5. **One Zarr store per build** — builds can be loaded independently or merged
   for cross-client training.
6. **Liquid data is mandatory when present** — `liquid_mask` and `liquid_height`
   are zero-filled for tiles without water, and `has_liquid` in the index marks
   which tiles have real liquid data.

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
  absent from the NPZ shard. The `has_liquid` index column marks real data.

## Index (Parquet)

| Column | dtype | Description |
|--------|-------|-------------|
| `tile_id` | int64 | Row index into Zarr arrays |
| `map` | string | Map name (e.g. "Azeroth", "Northrend") |
| `tile_x` | int32 | Tile X coordinate |
| `tile_y` | int32 | Tile Y coordinate |
| `has_height` | bool | height_257 present and nonzero |
| `has_normals` | bool | normal_xyz present and nontrivial |
| `has_alpha` | bool | alpha_256 present |
| `has_holes` | bool | holes_16 present |
| `has_liquid` | bool | unified_liquid_mask present with nonzero values |
| `has_shadow` | bool | shadow_mask present |
| `has_mcly` | bool | mcly_texture_ids present |

## Build Pipeline

```bash
cd wow-viewer/data-harvester

# Step 1: Extract all tiles for a build (uses existing C# harvester)
uv run python scripts/build_v16_dataset.py extract \
    --client-root I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft \
    --build 3_3_5_12340 \
    --output-dir ../output/datasets/v16/temp_npz

# Step 2: Consolidate into Zarr + index
uv run python scripts/build_v16_dataset.py consolidate \
    --input-dir ../output/datasets/v16/temp_npz/3_3_5_12340 \
    --output ../output/datasets/v16/3_3_5_12340.zarr \
    --build 3_3_5_12340

# Step 3: Clean up temp NPZ files
uv run python scripts/build_v16_dataset.py cleanup \
    --input-dir ../output/datasets/v16/temp_npz/3_3_5_12340

# Or: do all steps at once
uv run python scripts/build_v16_dataset.py build \
    --client-root I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft \
    --build 3_3_5_12340
```

## Training

```bash
cd wow-viewer/data-harvester
uv run python scripts/train_v16.py \
    --dataset-dir ../output/datasets/v16 \
    --builds 3_3_5_12340 4_0_0_11927
```

The `V16Dataset` class reads from Zarr stores, using the Parquet index for
train/val splitting and the `has_*` columns for feature masking.

## Compression Benchmarks (target)

| Array | Uncompressed | Blosc-zstd-5 | Ratio |
|-------|-------------|-------------|-------|
| height_257 (257×257 f32) | 264 KB | ~50 KB | 5:1 |
| minimap_rgb (256×256×3 u8) | 196 KB | ~20 KB | 10:1 |
| alpha_256 (256×256×4 f32) | 1024 KB | ~80 KB | 13:1 |
| liquid_mask (256×256 f32) | 256 KB | ~5 KB | 50:1 |

Expected dataset size for one build (~5000 tiles): ~500 MB compressed.

## V16 Model

Same architecture as V15 (ConvNeXt V2 Nano encoder, U-Net decoder) with the
addition of the liquid head. Total ~27.4M parameters.

| Head | Output shape | Loss |
|------|-------------|------|
| height | (B, 1, 257, 257) | L1, object-masked |
| normals | (B, 3, 257, 257) | 1 − cosine similarity, normal-masked |
| alpha | (B, 4, 256, 256) | L1, object-masked × has_alpha |
| holes | (B, 1, 16, 16) | L1, object-masked × has_holes |
| liquid | (B, 1, 256, 256) | L1, object-masked × has_liquid |

## Normalization

- **Height**: z-score per tile: `h' = (h - μ) / (σ + 1e-8)`. μ and σ stored in
  the index for denormalization at inference.
- **Normals**: unit vectors in [-1, 1]. Zero vectors masked.
- **Alpha**: raw [0, 1].
- **Liquid mask**: raw [0, 1].
- **Minimap RGB**: [0, 255] → [0, 1] at training time.