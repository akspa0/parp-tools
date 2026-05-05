# V12 Texture Decomposer — Specification

## Problem

A WoW minimap is a baked composition of terrain geometry, texture painting, lighting,
shadows, and object placement. Training a height model directly on the minimap forces
the model to "see through" the paint. Instead, decompose first.

## Approach

Train a model to peel the minimap into its component layers:

```
minimap (256×256 RGB)
    → mcal_alpha (4ch, 256×256)    — texture blend weights per pixel
    → mcly_labels (16×16, N-class) — which texture per chunk
    → texture_residual (3ch, 256×256) — shading, lighting, objects
```

Once decomposed, the MapTexture can be re-composited:

```
MapTexture = composit(tileset_BLPs × mcal_alpha)
Residual   = real_minimap − MapTexture
```

The residual is the CLEAN input for the height model — it contains geometry shading,
lighting, and object silhouettes without the texture paint. A height model trained on
the residual learns terrain shape directly, without the minimap's texture noise.

## Architecture

- **Backbone:** ConvNeXt V2 Nano (15.6M params), 80-channel stages
- **Input:** 5 channels (minimap RGB + luma + Sobel gradient) @ 256×256
- **Decoder:** U-Net with skip connections from all 4 ConvNeXt stages
- **Task heads** (from 64ch @ 256×256 decoder features):
  - `mcal_head`: Conv3→GELU→Conv1→sigmoid → 4ch
  - `residual_head`: Conv3→GELU→Conv1→sigmoid → 3ch
  - `mcly_head`: Pool→16→Conv1→GELU→Conv3 → N-class (vocabulary-dependent)

## Training Data

Same NPZ shards as V11, augmented with pre-computed composited MapTextures:

```
synthetic_minimap_256  — composited from ground-truth MCAL + tilesets
texture_residual_256   — real_minimap − synthetic_minimap
```

Pre-computed via `scripts/precompute_maptextures.py` which reads the harvested
tileset PNGs (848 textures) and composites MapTextures for every tile with
MCAL/MCLY data (6803 of 6821 tiles).

## Loss

```
L_total = L_mcal_l1 + 0.2 × L_mcly_ce + 0.5 × L_residual_l1

L_mcal_l1:     L1 on alpha weights (masked, only where MCAL present)
L_mcly_ce:     Cross-entropy on per-chunk texture class (ignore_index for unknown)
L_residual_l1:  L1 on residual (masked, only where composited data available)
```

All losses per-sample masked — tiles without MCAL/MCLY/residual don't contribute.

## Training Schedule

- Optimizer: AdamW, lr=2e-4, weight_decay=0.05
- LR: 5-epoch linear warmup + cosine decay, floor at 1% of peak
- Batch: 16 (15.6M params, fits 8GB VRAM easily)
- Mixed precision: AMP (GradScaler)
- Epochs: 200
- Data: 2000 curated tiles, 12% validation split
- Augmentation: random horizontal/vertical flips

## Inference

```
minimap → V12Model → (mcal_pred, mcly_pred, residual_pred)
mcal_pred + mcly_pred + tilesets → Model2 compositor → MapTexture
MapTexture + residual_pred → reconstructed minimap (should match input)
MapTexture → clean input for height model (no paint, just geometry)
```

## Success Criteria

- MapTexture visually resembles the real minimap minus objects and shadows
- Residual contains geometry shading and object silhouettes, not texture colors
- Reconstructed minimap (MapTexture + residual) is visually close to the original
- MCAL/MCLY predictions are within 0.15 L1 / 2.0 CE of ground truth (from V11.1 baseline)

## File Layout

| File | Purpose |
|------|---------|
| `scripts/train_v12.py` | Training script |
| `scripts/precompute_maptextures.py` | Offline MapTexture compositing |
| `scripts/synthesize_minimap.py` | On-the-fly compositing (V11 era) |
| `docs/v12-texture-decomposer-spec.md` | This document |
| `output/tmp/tilesets/` | 848 harvested tileset PNGs |
| `output/tmp/maptextures/` | Pre-computed composited outputs (per tile) |
