# V11 Terrain Model Architecture

## Why Not V10

V10 was a two-stage pipeline (Stage 1: minimap→height_17, Stage 2: multi-resolution refinement). The archive-backed extraction path had MpqArchiveCatalog hash probe bugs causing hangs on real MPQs. The training code was stripped down — no audit, no curation, no hard replay, no early stopping.

V11 goes back to v9's proven single-stage approach with a modern backbone.

## Model

**Encoder:** ConvNeXt V2 Tiny (28.6M params, from `timm`)
- Stem: 4×4 conv stride 4 (256→64)
- Stage 0: 3 blocks, 96ch, 64×64
- Stage 1: 3 blocks, 192ch, 32×32
- Stage 2: 9 blocks, 384ch, 16×16
- Stage 3: 3 blocks, 768ch, 8×8
- Normalization: LayerNorm (batch-size agnostic), activation: GELU

**Decoder:** U-Net style with ConvNeXt refinement blocks
- 8×8 → 16×16 (skip from stage 2) → 32×32 (skip from stage 1) → 64×64 (skip from stage 0) → 256×256
- 3× ConvNeXt refinement blocks at 256×256 for detail

**Heads (all from 256×256 shared features):**
- height_17: adaptive pool → 1×1 conv
- height_65: adaptive pool → 1×1 conv
- height_257: 3× 3×3 conv → 1ch output
- mcal_alpha: 2× 3×3 conv → sigmoid → 4ch
- mcly_class: pool to 16×16 → 1×1 conv → 3×3 conv → N classes
- hole_mask: pool to 16×16 → 1×1 conv → logits

**Total:** 35.5M params (~142MB fp32, ~71MB bf16)

## Input Signals (26 channels)

| # | Signal | Source NPZ array | Dropout |
|---|--------|-----------------|---------|
| 0-2 | minimap_rgb | `minimap_rgb_256` | 1× |
| 3-6 | mcal_alpha | `mcal_alpha_pack_256` | 1× |
| 7-9 | mcnr_normal | `mcnr_normal_xyz` / `normal_rgb_256` | 1× |
| 10-12 | mccv_rgb | `mccv_rgb` | **3×** (artist-painted, no geometric link) |
| 13 | coarse_height | `height_17` | 1× |
| 14 | liquid_mask | `unified_liquid_mask` / `liquid_mask_257` | 1× |
| 15 | liquid_height | `unified_liquid_height` / `liquid_height_257` | 1× |
| 16 | object_mask | `object_mask_257` | 1× |
| 17 | object_precise | `object_precise_mask_257` | 1× |
| 18 | pm4_path | `pm4_path_mask` / `pm4_mask_257` | 1× |
| 19 | pm4_building | `pm4_building_footprint_mask` | 1× |
| 20 | pm4_mprl | `pm4_mprl_mask` | 1× |
| 21 | hole_mask | `hole_mask_16` / `hole_mask_16x16` | 1× |
| 22 | luma | derived from minimap | 1× |
| 23 | gradient | derived from minimap | 1× |
| 24 | height_range | derived from coarse height | 1× |
| 25 | detail_energy | placeholder | 1× |

Signals not used: shadow masks (never present on minimap tiles).

## Output Targets

| Head | Shape | Loss | Notes |
|------|-------|------|-------|
| height_17 | 1×17×17 | L1 + gradient | Z-scored by dataset |
| height_65 | 1×65×65 | L1 + gradient | Z-scored |
| height_257 | 1×257×257 | L1 + gradient | Z-scored |
| mcal_alpha | 4×256×256 | L1 | Sigmoid output [0,1] |
| mcly_labels | 16×16 | CE | Vocabulary of N texture classes |
| hole_mask | 16×16 | BCE | Logits |

## Loss

Uncertainty-weighted multi-task loss with learned per-task sigma:

```
L_total = L_height / (2·σ_h²) + log(σ_h)
        + L_mcal / (2·σ_m²) + log(σ_m)   [if mcal present]
        + L_mcly / (2·σ_c²) + log(σ_c)   [if mcly present]
        + L_hole / (2·σ_o²) + log(σ_o)   [if hole present]
```

Height loss aggregates L1 at 3 scales + gradient L1. Missing signals are masked per-batch.

## Training

- **Optimizer:** AdamW or Lion, lr=2e-4, weight_decay=0.05
- **Schedule:** Linear warmup (5 epochs) → Cosine annealing
- **EMA:** Decay 0.999, best_ema.pt saved separately
- **AMP:** bf16 mixed precision via GradScaler
- **Gradient clipping:** 1.0
- **Signal dropout:** 15% base (MCCV at 45%)
- **Augmentation:** random horizontal/vertical flips
- **Batch:** 32 (fits 17GB RTX 4070 Ti), gradient accumulation for effective batch 64
- **Data:** LRU cache (2GB cap), MCAL downsampled to 256×256 at load

## Inference

```
python scripts/infer_v11.py <checkpoint.pt> <shard_paths> --export-obj
```

Produces: NPZ with predicted arrays, OBJ+MTL+texture mesh, JSON report per tile.

## Data Extraction

Two working paths produce NPZ shards for training:

**V9 pipeline** (recommended for cross-client extraction):
```
dataset-scan → dataset-audit → dataset-curate → dataset-build-cache
```
Staged client roots on filesystem. Now includes MCAL/MCLY arrays.

**V10-native single-pass** (quick for development tiles):
```
dataset-build-v10-stage1 --input-dir <adt_dir> --minimap-root <minimap_dir>
```
No temp files, filesystem-only.
