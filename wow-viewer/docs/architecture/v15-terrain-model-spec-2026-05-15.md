# V15 Terrain Model Specification

## Purpose

Given a single 256×256 RGB minimap image, predict a complete terrain patch:
heightmap, normals, texture alpha masks, and hole flags.
No priors required at inference — all knowledge is learned from
game-extracted training data.

## Input

| Signal | Shape | Source |
|--------|-------|--------|
| `minimap_rgb_256` | 256×256×3 uint8 | MPQ minimap BLP, decoded to RGB |

## Outputs

| Head | Shape | Range | Target (NPZ key) | Loss |
|------|-------|-------|-------------------|------|
| `height` | 257×257 float32 | world-space Z (m), normalised to μ=0 σ=1 per tile | `height_257` | L1, object-masked |
| `normals` | 257×257×3 float32 | [-1,1] unit vectors | `mcnr_normal_xyz` | 1−cosine similarity, valid-normal-masked × signal-available |
| `alpha` | 256×256×4 float32 | [0,1] blend weights | `mcal_alpha_pack_256` | L1, object-masked × signal-available |
| `holes` | 16×16 float32 | [0,1] hole probability | `hole_mask_16` | L1, object-masked × signal-available |

## Architecture

```
Minimap (3×256×256)
  │
  ▼
ConvNeXt V2 Nano (15.6M, pretrained ImageNet, features_only)
  ├─ e0: 80ch @  64×64  (stride 4)
  ├─ e1: 160ch @ 32×32  (stride 8)
  ├─ e2: 320ch @ 16×16  (stride 16)
  └─ e3: 640ch @  8×8   (stride 32)
  │
  ▼
Bottleneck: ConvBlock(640→640) @ 8×8
  │
  ▼
Decoder (U-Net skip fusion):
  dec3: UpFuse(640 + 320→320) @ 16×16
  dec2: UpFuse(320 + 160→160) @ 32×32
  dec1: UpFuse(160 + 80→80)   @ 64×64
  dec0: ConvBlock(80→64)       @ 64×64
  │
  ├─ height: Upsample→257 + Conv→1
  ├─ normals: Upsample→257 + Conv→3 + Tanh
  ├─ alpha: Upsample→256 + Conv→4 + Sigmoid
  └─ holes: AdaptivePool→16 + Conv→1 + Sigmoid
```

Total: ~27.3M parameters.

## Training

### Data

NPZ shards at `output/datasets/d1_reharvest/shards/<build>/<map>/<file>.npz`.
Each shard contains all signals extracted from one ADT tile.
Tiles missing any required training signal are excluded from the dataset.

### Sample selection

- 1000 training tiles randomly sampled from all D1-eligible shards
- Validation tiles from the pre-selected holdout set (validation_selection.json)
- `object_mask_257` is kept at native 257×257 and resized per-head as needed for pixel loss weighting
  (pixels with placed objects = weight 0, terrain-only pixels = weight 1)
- `mcnr_normal_xyz` contains ~50% zero vectors (pad/gap vertices). A per-pixel `normal_mask` identifies
  valid normals (sum(|xyz|) > 1e-6). Zero vectors are replaced with (0,0,1) before training.
  Only valid pixels contribute to the normals loss.

### Loss

```
L = L1_height + 2.0 × Cos_normals × [has_normals] + L1_alpha × [has_alpha] + L1_holes × [has_holes]
```

Normals loss uses **cosine similarity** (`1 − cos_sim`) on unit-length vectors, masked to
valid (non-zero) pixels only. L1 terms are weighted by `(1 − object_mask)`.
Signal-availability multipliers zero out terms for tiles that lack the corresponding ground truth.

Height is z-score normalised per tile: `h' = (h - μ) / (σ + 1e-8)`.
μ and σ are saved per tile for denormalisation at inference.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 2e-4 |
| Weight decay | 0.05 |
| Batch size | 8 |
| Epochs | 200 |
| LR schedule | Cosine annealing |
| Mixed precision | bf16 AMP (CUDA) |
| Gradient clip | 1.0 |
| Seed | 42 |

## Inference

```
uv run python scripts/test_v15.py --checkpoint checkpoints/v15_best.pt --npz-path <path>
```

Outputs per tile:

| File | Content |
|------|---------|
| `input_minimap.png` | The input minimap tile |
| `height_gt.png` | Ground-truth heightmap (grayscale) |
| `height_pred.png` | Predicted heightmap (grayscale) |
| `normals_gt.png` | Ground-truth normals (RGB) |
| `normals_pred.png` | Predicted normals (RGB) |
| `alpha_gt_ch0.png` → `ch3` | Ground-truth MCAL alpha channels |
| `alpha_pred_ch0.png` → `ch3` | Predicted alpha channels |
| `holes_gt.png` | Ground-truth hole mask |
| `holes_pred.png` | Predicted hole mask |
| `object_mask.png` | Object footprint mask (training loss weight) |
| `metrics.json` | Per-tile L1 errors for height/normals/alpha/holes |
| `<tile>_terrain.obj` | OBJ mesh from predicted heightmap |

## Validation During Training

Every `--val-interval` epochs (default 10), the trainer:
1. Computes numeric validation metrics (L1 for height/normals/alpha)
2. Saves per-tile comparison images to `logs/val_epoch<NNNN>/tile_00/`, `tile_01/`, etc.
3. Each tile directory contains:
   - `height_gt.png` / `height_pred.png` — visual height comparison
   - `normals_gt.png` / `normals_pred.png` — visual normals comparison
   - `alpha_gt_ch0.png` / `alpha_pred_ch0.png` — alpha mask comparison
   - `object_weight.png` — which pixels are excluded from loss
   - `metrics.json` — numeric L1 errors for this tile at this epoch

Track progress by comparing these images across validation snapshots.
Numeric metrics are logged in `logs/v15_training_log.json`.
The best checkpoint (by val height L1) is saved to `checkpoints/v15_best.pt`.

## Training Artifacts

| File | Purpose |
|------|---------|
| `checkpoints/v15_best.pt` | Best validation checkpoint |
| `checkpoints/v15_final.pt` | Final epoch checkpoint |
| `logs/v15_training_log.json` | Per-epoch train/val metrics |
| `logs/val_epoch<NNNN>/` | Validation snapshot images per tile |
| `docs/architecture/v15-terrain-model-spec-2026-05-15.md` | This document |
