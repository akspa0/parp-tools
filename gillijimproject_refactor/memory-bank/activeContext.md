# Active Context

## Current Focus: V6.1 Training Pipeline (Jan 16, 2026)

### Status Summary

**V6.1 IMPLEMENTATION COMPLETE**: Training script updated with all V6 spec requirements:
- ✅ 8-channel input (added bounds hint channel)
- ✅ SSIM loss for perceptual quality
- ✅ Edge loss using Sobel filters
- ✅ Loss weights matching `v6_resnet34.yaml` config
- ✅ WDL fallback mode for missing data
- ✅ Dataset validation script (`scripts/prepare_v6_datasets.py`)

**Next Phase**: Object rendering to address minimap-heightmap mismatch.

### Key Documents

- **Full V6.1 Spec**: `memory-bank/planning/V6_TRAINING_PIPELINE.md`
- **Object Rendering Plan**: `memory-bank/planning/OBJECT_RENDERING_PLAN.md`
- **Training Script**: `src/WoWMapConverter/scripts/train_height_regressor_v6_absolute.py`
- **Dataset Prep**: `scripts/prepare_v6_datasets.py`

---

## V6.1 Training Configuration

### Model Architecture
- **Backbone**: Custom 8-channel U-Net
- **Input**: minimap(3) + normalmap(3) + WDL(1) + bounds_hint(1) = 8 channels
- **Output**: heightmap_global(1) + heightmap_local(1) + alpha(4) + bounds(4)

### Loss Weights
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

### Dataset Validation (Azeroth v11)
- **Total tiles**: 685
- **Complete for V6**: 678/685 (98.9%)
- **Height range**: [-524.6, 896.8]

---

## Object Rendering Plan (Next Phase)

### Problem
Minimaps show objects (trees, buildings) but heightmaps are terrain-only → loss mismatch.

### Solution
1. Extract bounding boxes from WMO/M2 meshes (C#)
2. Render object silhouette masks (Python)
3. Add as 9th input channel + loss weighting

### Files to Modify
- `VlmDatasetExporter.cs` - Add bbox extraction
- `render_object_masks.py` - New script
- `train_height_regressor_v6_absolute.py` - 9-channel support

---

## Scripts Usage

```bash
# Validate dataset
python scripts/prepare_v6_datasets.py --dataset test_data/vlm-datasets/053_azeroth_v11 --validate

# Render normalmaps
python scripts/prepare_v6_datasets.py --dataset test_data/vlm-datasets/053_azeroth_v11 --render-normalmaps

# Start training
python src/WoWMapConverter/scripts/train_height_regressor_v6_absolute.py
```

---

## Dataset Exports

| Map | Version | Tiles | Status |
|-----|---------|-------|--------|
| Azeroth | v11 | 685 | ✅ Ready for V6.1 |
| Kalimdor | v6 (pending) | ~951 | 🔧 Needs export |
| Shadowfang | v1 | ~20 | ✅ Complete |
| Deadmines | v2 | ~15 | ✅ Complete |

**Client Path**: `H:\053-client\`

---

## Key Files

| File | Purpose |
|------|---------|
| `train_height_regressor_v6_absolute.py` | V6.1 training script |
| `prepare_v6_datasets.py` | Dataset validation/fixing |
| `render_normalmaps.py` | Normalmap rendering from MCNR |
| `VlmDatasetExporter.cs` | C# dataset export |
| `VlmTrainingSample.cs` | JSON model definitions |

---

## Technical Notes

- **Alpha MCVT format**: 81 outer (9×9) FIRST, then 64 inner (8×8)
- **MCNR padding**: Alpha=448 bytes, LK=435 bytes (truncate to 435)
- **WDL upsampling**: 17×17 → 256×256 bilinear interpolation
- **Height normalization**: `(h - global_min) / (global_max - global_min)`

