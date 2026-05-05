# V12 Improvements & Overhead Object Library — Plan

## Part 1: What's Wrong With V12

### Layer semantics are ignored

The model predicts 4 independent alpha channels, but terrain layers have strict semantics:

```
L0 (base):     Dirt, grass, sand — always present, fills gaps
L1 (overlay):  Grass tufts, light stone, paths — contrasts with base
L2 (features): Heavy stone, roads, rivers — stands out from L0/L1
L3 (detail):   Moss, highlights, snow caps, decorative detail
```

Blending order: `L0 → mix(L0, L1, α1) → mix(result, L2, α2) → mix(result, L3, α3)`

### Current model problems

1. **No layer-awareness**: All 4 channels are symmetric in the architecture. L0 isn't treated as special (always-on base).
2. **No blending prior**: The model doesn't know layers blend sequentially (not additive).
3. **Residual still has texture**: The reconstruction loss we just added fixes this at the loss level, but the model still doesn't reason about layers hierarchically.
4. **p(mcal|mcly) is implicit**: The model learns texture appearance indirectly through MCAL L1 gradients, rather than through explicit texture lookup.

### Fix: Hierarchical decoder heads

Predict MCAL sequentially: first L0 (base coverage), then L1 (overlay on L0), then L2, then L3. Each head sees the previous layer's predictions.

```
Shared encoder features
  → L0 head: predicts α0 (base coverage, typically high/sparse)
  → L1 head: sees features + α0, predicts α1
  → L2 head: sees features + α0+α1, predicts α2
  → L3 head: sees features + α0+α1+α2, predicts α3 (detail)
```

This encodes the blending hierarchy directly into the architecture.

## Part 2: Overhead Object Library

### Goal

A versioned library of top-down orthographic renders of every WMO/MDX that appears on minimaps, for object identification and masking in the residual.

### What already exists

| Component | Status |
|---|---|
| `wowviewer-app mdx-gpu-frame` | Headless MDX render to PNG via `MdxGpuPreviewCaptureRunner` |
| `wowviewer-app m2-gpu-frame` | Headless M2 render to PNG via `M2GpuPreviewCaptureRunner` |
| `WmoGpuPreviewRenderer` | Renders to FBO, **NO CaptureBmp** |
| `PreviewCameraPresets` | Has `top` preset (azimuth=0, elevation=90) |
| `PreviewCameraPlanner` | Computes orthographic view from bounding box |
| Legacy `ScreenshotRenderer` | Multi-angle MDX screenshots, **skips WMO** |

### What needs building

#### A. WMO CaptureBmp

Add `CaptureBmp(string outputPath, int width, int height)` to `WmoGpuPreviewRenderer`. Pattern follows `MdxGpuPreviewRenderer.CaptureBmp`:
- Read from internal color texture → pixel buffer
- Write via `ImageOutputWriter.WriteRgbaImage()`
- Extra: render ID group index selection for multi-group WMOs

#### B. Harvest Tool

New CLI tool: `wowviewer-app object-harvest`

```
object-harvest
  --client-root <dir>          # Path to game client
  --client-version <string>    # e.g. "3_3_5_12340"
  --output-dir <dir>           # Where signature images go
  --model-list <path>          # Optional: specific models to render (WMO root paths + MDX paths)
  --render-size <int>          # Output resolution (default 256)
  --workers <int>              # Parallel renders
  --format png                 # PNG output
```

Pipeline:
1. Scan client root for WMO root files and MDX files in `World\` directory tree
2. Load each model via existing preview loaders
3. Set camera to top-down orthographic (`PreviewCameraPresets.Top`)
4. Render to 256×256 PNG via invisible Silk.NET window
5. Save as `{md5_of_virtual_path}__{client_version}.png`
6. Build index JSON: `{virtual_path → hash, version, md5, render_size}`

#### C. Index manifest

```json
{
  "client_version": "3_3_5_12340",
  "client_root": "H:\\CLIENTS\\3.3.5.12340",
  "count": 2847,
  "entries": [
    {
      "virtual_path": "World\\Minimap\\SI\\SI\\MinimapSI.blp",
      "hash": "a1b2c3d4e5f6...",
      "signature_path": "a1b2c3d4e5f6__3_3_5_12340.png",
      "model_type": "wmo_root",
      "version": "3_3_5_12340"
    }
  ]
}
```

### How it integrates with V12

1. Object identification model matches residual regions against the library
2. Object masks clean up the residual → better Stage 2 training
3. Object type labels enable auxiliary segmentation loss in future V12 versions

## Part 3: Action Items

| Priority | Item | Depends on | Effort |
|----------|------|------------|--------|
| P0 | Add `CaptureBmp` to `WmoGpuPreviewRenderer` | — | ~1 day |
| P0 | Write object-harvest CLI tool | WMO CaptureBmp | ~2 days |
| P1 | Refactor V12 to hierarchical MCAL heads | — | ~1 day |
| P1 | Retrain V12 with hierarchical heads + reconstruction loss | — | ~1 day GPU |
| P2 | Run harvest across all staged clients (3.3.5, 4.x, 5.x, 6.x) | object-harvest tool | ~1 day GPU |
| P2 | Build object identification model (ViT + Siamese) | Harvested library | ~2 days |
