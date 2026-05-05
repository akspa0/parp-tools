# V12 Texture Decomposer — Training Guide

## Prerequisites

- Harvested tileset PNGs at `output/tmp/tilesets/` (848 textures)
- NPZ cache with MCAL/MCLY data at `output/tmp/v11_cache/`

## Step 1 — Pre-compute MapTextures

Composites ground-truth MCAL × tilesets for every tile with texture data:

```powershell
& '.venv-train/Scripts/python.exe' scripts/precompute_maptextures.py `
  output/tmp/v11_cache/v9_tensor_cache_manifest.json `
  --harvest-dir output/tmp/tilesets --output-dir output/tmp/maptextures
```

This writes `_composited.npz` sidecar files next to each NPZ shard with
`synthetic_minimap_256` and `texture_residual_256` arrays.

## Step 2 — Train

```powershell
& '.venv-train/Scripts/python.exe' scripts/train_v12.py `
  output/tmp/v11_cache/v9_tensor_cache_manifest.json `
  --output-dir runs/v12_prod --epochs 200 --batch-size 16 `
  --max-samples 2000
```

## Step 3 — Validate

The checkpoint at `runs/v12_prod/best.pt` contains model weights + MCLY vocabulary.
Load it and run inference on a few tiles to check decomposition quality.

## Expected Output

| Epoch | mcal_l1 | mcly_ce | residual_l1 | Notes |
|-------|---------|---------|-------------|-------|
| 10 | 0.08 | 2.5 | 0.12 | MCAL converges fast |
| 50 | 0.05 | 1.8 | 0.08 | MCLY catching up |
| 100 | 0.04 | 1.5 | 0.06 | Residual settling |
| 200 | 0.03 | 1.3 | 0.05 | Ready for height model |

## What Comes After

1. V12 decomposes the minimap into texture (MapTexture) + geometry (Residual)
2. A new height model trains on the residual — sees pure terrain shading without paint
3. At full inference: minimap → V12 → (MCAL, MCLY, residual) → composit MapTexture → height model on residual → OBJ mesh

## Troubleshooting

- **No MCLY data found:** Run precompute_maptextures first
- **Residual loss NaN:** Check tileset PNGs exist and BLP→PNG conversion was correct
- **Empty composite:** Texture names in sidecar don't match harvest filenames
- **OOM:** Reduce batch-size to 8 (15.6M model is light, should fit easily)
