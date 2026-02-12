# V6.1 Training Pipeline - Complete Specification

> **Last Updated**: January 16, 2026
> **Status**: Implementation Complete (Object Rendering Pending)

---

## Overview

This document describes the complete V6.1 heightmap regression training pipeline for WoW terrain reconstruction from minimap images.

## Model Architecture

### Input Channels (8 total → 9 with objects)

| Channel | Description | Source |
|---------|-------------|--------|
| 0-2 | Minimap RGB | `{tile}.png` |
| 3-5 | Normalmap RGB | `{tile}_normalmap.png` (rendered from MCNR) |
| 6 | WDL height hint | `wdl_heights.outer_17` → 17×17 → 256×256 bilinear |
| 7 | Height bounds hint | `height_min_norm` broadcast to 256×256 |
| 8* | Object mask (planned) | Bounding box silhouettes |

### Output Heads

- **Heightmap Decoder**: 2 channels (global + local) at 256×256
- **Alpha Mask Decoder**: 4 channels (texture layer masks)
- **Bounds Regression**: 4 scalars (tile_min, tile_max, global_min, global_max)

### Backbone

- **Current**: Custom U-Net encoder (7 layers)
- **Future**: ResNet-34 encoder with ImageNet pretrained weights

---

## Loss Function (V6.1)

```python
LOSS_WEIGHTS = {
    "heightmap_global": 0.15,
    "heightmap_local": 0.35,
    "alpha_masks": 0.10,
    "bounds": 0.05,
    "ssim": 0.05,
    "gradient": 0.05,
    "edge": 0.25,
}
```

### Loss Components

| Loss | Purpose | Implementation |
|------|---------|----------------|
| `heightmap_global` | Global height accuracy | L1 loss |
| `heightmap_local` | Local detail accuracy | L1 loss |
| `alpha_masks` | Texture layer blending | L1 loss |
| `bounds` | Height range prediction | MSE loss |
| `ssim` | Perceptual similarity | Custom SSIM with Gaussian windowing |
| `gradient` | Smooth transitions | L1 on finite differences |
| `edge` | Sharp boundaries | Sobel-based edge detection |

---

## Dataset Structure

```
053_azeroth_v11/
├── dataset/
│   └── {MapName}_{X}_{Y}.json    # Terrain data, objects, metadata
├── images/
│   ├── {tile}.png                # Minimap (input)
│   ├── {tile}_normalmap.png      # Rendered from MCNR
│   ├── {tile}_heightmap.png      # Local normalized
│   └── {tile}_heightmap_global.png  # Global normalized
├── shadows/                       # Per-chunk shadow maps
├── masks/                         # Per-chunk alpha masks
└── stitched/                      # Full-map stitched images
```

### JSON Schema (per tile)

```json
{
  "image": "images/{tile}.png",
  "terrain_data": {
    "adt_tile": "{MapName}_{X}_{Y}",
    "heights": [{ "idx": 0, "h": [145 floats...] }, ...],
    "chunk_layers": [{ "idx": 0, "normals": [435 bytes...], ... }],
    "objects": [
      {
        "name": "arathitree03",
        "unique_id": 209539,
        "x": 20828.436, "y": 95.82, "z": 18159.25,
        "rot_x": -12, "rot_y": 120.27, "rot_z": -9,
        "scale": 0.74,
        "category": "m2"
      }
    ],
    "wdl_heights": { "outer_17": [289 shorts] },
    "height_min": 119.02,
    "height_max": 557.82,
    "height_global_min": -524.58,
    "height_global_max": 896.78
  }
}
```

---

## Training Configuration

```yaml
training:
  batch_size: 4
  learning_rate: 1e-4
  epochs: 500
  early_stop_patience: 50
  optimizer: adamw
  weight_decay: 1e-5
  scheduler: reduce_lr_on_plateau

validation:
  split: map_level  # No tile leakage
  fraction: 0.1
  holdout_maps: []  # Optional specific maps

augmentation:
  color_jitter: {brightness: 0.15, contrast: 0.15, saturation: 0.15, hue: 0.05}
  horizontal_flip: true
  # vertical_flip: off (terrain asymmetry)
```

---

## Scripts

### Dataset Preparation

```bash
# Validate dataset for V6 requirements
python scripts/prepare_v6_datasets.py --dataset path/to/dataset --validate

# Render normalmaps from JSON terrain data
python scripts/prepare_v6_datasets.py --dataset path/to/dataset --render-normalmaps

# Fix all issues (normalmaps, WDL format)
python scripts/prepare_v6_datasets.py --dataset path/to/dataset --fix-all
```

### Training

```bash
# Start V6.1 training
python src/WoWMapConverter/scripts/train_height_regressor_v6_absolute.py
```

---

## Object Rendering (Planned Phase)

### Problem

Minimaps show objects (buildings, trees, props) but heightmaps are terrain-only. This causes loss mismatch.

### Solution

1. **Extract mesh bounding boxes** from WMO/M2 files
2. **Render object silhouette masks** (256×256)
3. **Add as 9th input channel** OR use for loss weighting
4. **Optionally composite object heights** into target heightmap

### Implementation Files

| File | Status | Purpose |
|------|--------|---------|
| `VlmDatasetExporter.cs` | Modify | Add `bbox_min`, `bbox_max` to object JSON |
| `render_object_masks.py` | Create | Render silhouette masks from bboxes |
| `train_height_regressor_v6_absolute.py` | Modify | Add 9th channel, loss weighting |

---

## Known Issues & Solutions

| Issue | Solution | Status |
|-------|----------|--------|
| Missing normalmaps in v11 | Render from JSON `chunk_layers.normals` | ✅ Script exists |
| Missing WDL data | Fallback to grayscale (chunk height average) | ✅ Implemented |
| 7 vs 8 input channels | Added bounds hint channel | ✅ Fixed |
| Missing SSIM/Edge losses | Implemented custom loss functions | ✅ Fixed |
| Object-heightmap mismatch | Object mask channel (planned) | 🔧 Pending |

---

## File Locations

- **Training Script**: `src/WoWMapConverter/scripts/train_height_regressor_v6_absolute.py`
- **Dataset Prep**: `scripts/prepare_v6_datasets.py`
- **Config**: `scripts/configs/v6_resnet34.yaml`
- **Datasets**: `test_data/vlm-datasets/053_*/`
- **Output**: `J:\vlm_output\wow_height_regressor_v6_absolute\`
