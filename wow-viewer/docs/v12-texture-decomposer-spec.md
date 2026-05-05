# V12 Texture Decomposer — Specification (Stage 1)

## Problem

A WoW minimap is a baked composition of terrain geometry, texture painting, lighting, shadows, and object placement. Training a height model directly on the minimap forces the model to "see through" the paint. Instead, decompose first: isolate the texture paint (MapTexture) from the geometry signal (residual).

## Approach

Train a model to peel the minimap into its component layers:

```
minimap (256×256 RGB)
    → mcal_alpha (4ch, 256×256)       — per-pixel blend weights for up to 4 texture layers
    → mcly_labels (16×16, N-class)     — which texture is used in each of 256 chunks
    → texture_residual (3ch, 256×256)  — shading, lighting, objects, everything that isn't paint
```

Stage 1 **only** produces the decomposition. The residual feeds Stage 2 (height model).

## Why Not Tileset Input Channels

The original V12 plan used 17-channel input (3 minimap RGB + luma + gradient + 12 tileset texture reference channels). This has a fatal flaw: at inference you need MCLY to construct the tileset channels, but you need tileset channels to predict MCLY — circular dependency.

Instead, this model uses **only 3-channel RGB minimap input**. Texture awareness comes implicitly through:
- **MCLY supervision**: chunk-level texture classification (the model learns "this visual pattern → texture A")
- **MCAL supervision**: pixel-level alpha values (the model learns "given this chunk's appearance, the alpha for texture A is X")
- **Residual supervision**: pre-computed via ground-truth MCAL × known textures (factors out texture paint, leaving geometry)

The model never needs to see tilesets directly. At inference, 3ch → decomposition. No circular dependency.

## Architecture

```
Input: 3ch RGB (256×256) — ImageNet normalized (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
  │
  └─ SegFormer B2 backbone (pretrained on ImageNet-1K + ADE20K, 25M params)
       │  stage 0: (64ch, 64×64)   @ 1/4 scale
       │  stage 1: (128ch, 32×32)  @ 1/8 scale
       │  stage 2: (320ch, 16×16)  @ 1/16 scale  ───→ MCLY head
       │  stage 3: (512ch, 8×8)    @ 1/32 scale
       │
       ├─ 1×1 projectors to 256ch
       │
       └─ U-Net decoder (progressive upsampling with stage skip connections)
            8×8 → 16×16 → 32×32 → 64×64 → 128×128 → 256×256
            │ (ConvTranspose2d + concat skip + Conv+GELU)
            │
            ├─ MCAL head (Conv+GELU+Conv → 4ch → sigmoid)
            │   (B,4,256,256) in [0,1]
            │
            └─ Residual head (Conv+GELU+Conv → 3ch → linear)
                (B,3,256,256) unconstrained

MCLY head:
  256ch @ 16×16 → Conv1 → GELU → Conv1 → (B,N_classes,16,16) logits
```

### Decoder detail

| Stage | In | Skip | Out | Op |
|-------|----|------|-----|----|
| up3 | 256ch @ 8×8 | 256ch @ 16×16 | 256ch @ 16×16 | ConvT(2,2) → concat skip → Conv3 |
| up2 | 256ch @ 16×16 | 256ch @ 32×32 | 256ch @ 32×32 | ConvT(2,2) → concat skip → Conv3 |
| up1 | 256ch @ 32×32 | 256ch @ 64×64 | 256ch @ 64×64 | ConvT(2,2) → concat skip → Conv3 |
| to_full | 256ch @ 64×64 | — | 64ch @ 256×256 | ConvT(2,2) → GELU → ConvT(2,2) → GELU |
| merge | 64ch @ 256×256 | p[0] upsampled to 256 | 64+256=320ch | interpolate skip, concat |

### Param count

- SegFormer B2 backbone (pretrained, fine-tuned): 24.7M
- Projectors + decoder + heads: 4.9M
- **Total: 29.6M** (~118MB fp32)

## Training Data

Same NPZ shards as V11, with pre-computed composited MapTextures as residual targets:

| Array | Shape | Source |
|-------|-------|--------|
| `minimap_rgb_256` | (256,256,3) uint8 | Rendered minimap tile |
| `mcal_alpha_pack_256` | (256,256,4) float32 | Ground-truth MCAL (from ADT) |
| `mcly_texture_ids` | (16,16,4) int32 | Ground-truth MCLY (which texture per chunk per layer) |
| `texture_residual_256` | (256,256,3) float32 | Pre-computed `real_minimap − composite(gt_MCAL × BLPs)` |

Pre-computed via `scripts/precompute_maptextures.py`:
```python
# For each tile:
synthetic = composite(gt_MCAL × tileset_BLPs at native resolution)
residual = real_minimap - bilinear_resize(synthetic, 256×256)
```

Residuals stored as `_composited.npz` sidecar files in `output/tmp/maptextures/`.

### MCAL properties

- 4 layers per pixel, each in [0, 1]
- ~78% of pixels have MCAL = 0 (sparse — only 1-2 layers active per pixel)
- Active layers typically have α ∈ [0.3, 1.0]
- Valid for ~6800 tiles (of ~7000 total)

### MCLY properties

- Per-chunk (16×16 grid, 256 chunks per tile)
- Only **first layer** used for classification target
- Chunks with ground-truth label −1 (untextured) are ignored in loss
- ~27 unique texture classes in a typical 2000-sample training set

### Residual properties

- Range: approximately [-0.7, 1.0] after /255 normalization
- Contains: terrain shading, building silhouettes, tree shadows, water, and other non-terrain elements
- Positive values: minimap is brighter than the composite (e.g., lit terrain, colorful objects)
- Negative values: minimap is darker than the composite (e.g., shadows, dark objects)

## Loss

```
L = 5.0 × L_mcal_l1 + 0.2 × L_mcly_ce + 0.5 × L_residual_l1
```

| Component | Weight | Type | Masking |
|-----------|--------|------|---------|
| MCAL L1 | 5× | L1 on alpha weights | Sample-level: ignore tiles without MCAL data |
| MCLY CE | 0.2× | Cross-entropy on 16×16 labels | Per-pixel: `ignore_index=-100` for unlabeled chunks |
| Residual L1 | 0.5× | L1 on residual | Sample-level: ignore tiles without composited data |

MCAL gets 5× weight because it's sparse — only ~22% of pixels contribute. Without the extra weight, predicting all zeros would give a deceptively low loss.

## Training Schedule

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Peak LR | 2e-4 |
| Weight decay | 0.05 |
| Warmup | 5 epochs linear (1% → 100%) |
| Decay | Cosine, floor at 1% of peak |
| Batch size | 16 |
| Mixed precision | AMP (GradScaler) |
| Samples | 2000 curated from manifest |
| Train/val split | 88/12 |
| Epochs | 200 |
| Augmentation | Random H/V flips |

## Inference (Single Pass)

```
minimap (256×256 RGB) → V12Model → {
    mcal:      (4, 256, 256) sigmoid — alpha per layer
    mcly:      (N, 16, 16) logits — argmax for texture class per chunk
    residual:  (3, 256, 256) linear — geometry signal for Stage 2
}
```

No tilesets needed. No circular dependency. Single forward pass.

## Stage 2 Connection

The residual output of Stage 1 is the **clean input** for Stage 2 (height model):

```
Stage 1: minimap → (MCAL, MCLY, residual)
Stage 2: residual → heightmap → OBJ mesh
```

Stage 1's MCLY predictions also enable data filtering for Stage 2: only train height on chunks with terrain textures (skip water, buildings, etc.).

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| MCAL L1 (val) | < 0.05 | Per-pixel alpha error, ×5 weight in total loss |
| MCLY CE (val) | < 1.0 | Chunk texture classification |
| Residual L1 (val) | < 0.04 | Shading/object prediction |
| Visual | Residual shows geometry, not paint | Look at a few tiles — residual should look like shaded terrain, not colored texture |

## File Layout

| File | Purpose |
|------|---------|
| `scripts/train_v12.py` | Model + dataset + training loop |
| `scripts/precompute_maptextures.py` | Offline residual computation from MCAL + tilesets |
| `docs/v12-texture-decomposer-spec.md` | This document |
| `output/tmp/v11_cache/` | NPZ shards with MCAL/MCLY ground truth |
| `output/tmp/maptextures/` | Pre-computed composited MapTexture residuals |
