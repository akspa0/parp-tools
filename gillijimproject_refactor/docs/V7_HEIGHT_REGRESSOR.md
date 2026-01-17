# WoW Height Regressor V7 - Complete Documentation

## Overview

The V7 Height Regressor is a deep learning model that predicts terrain heightmaps from WoW map tile data. It uses a 9-channel input and produces high-resolution 512x512 heightmaps.

---

## Part 1: Dataset Generation (C# VLM Exporter)

### Command
```bash
dotnet run -- vlm-export --client "H:\053-client\" --batch-all --out "J:\vlm-datasets"
```

### Output Structure
```
053_Azeroth_v30/
├── dataset/           # JSON metadata per tile
│   └── Azeroth_32_48.json
├── images/            # Minimaps (512x512) + Normal Maps + Heightmaps
│   ├── Azeroth_32_48.png          # Minimap (upscaled via NearestNeighbor)
│   ├── Azeroth_32_48_normal.png   # Normal Map (from MCNR)
│   └── Azeroth_32_48_heightmap.png # Heightmap (from MCVT)
├── liquids/           # Water masks and heights
│   ├── Azeroth_32_48_liq_mask.png
│   └── Azeroth_32_48_liq_height.png
├── shadows/           # Per-chunk shadow maps
└── masks/             # Per-chunk alpha layer masks
```

### Key Data Channels Generated
| File | Source | Resolution | Purpose |
|------|--------|------------|---------|
| `{tile}.png` | Minimap BLP | 512x512 | Visual context (roads, water, zones) |
| `{tile}_normal.png` | MCNR chunk | 512x512 | Terrain slope/orientation |
| `{tile}_heightmap.png` | MCVT chunk | 512x512 | Height Ground Truth |
| `{tile}_liq_mask.png` | MCLQ/MH2O | 512x512 | Binary water presence |

---

## Part 2: V7 Model Architecture

### Input Tensor: 9 Channels
| Channel | Name | Source | Purpose |
|---------|------|--------|---------|
| 0-2 | Minimap RGB | `images/{tile}.png` | Zone/texture context |
| 3-5 | Normal Map RGB | `images/{tile}_normal.png` | Slope orientation |
| 6 | WDL Height | JSON `wdl_heights` | Global elevation hint |
| 7 | Bounds Hint | JSON `height_min` | Altitude guidance |
| 8 | Water Mask | `liquids/{tile}_liq_mask.png` | Flat-water indicator |

### Network: 5-Level U-Net (SR-UNet)
```
Input: [B, 9, 512, 512]
    ↓ Encoder
    enc1: 64 ch  (512x512)
    enc2: 128 ch (256x256)
    enc3: 256 ch (128x128)
    enc4: 512 ch (64x64)
    enc5: 1024 ch (32x32)
    ↓ Bottleneck
    2048 ch (16x16) → GlobalPool → FC → Height Bounds [4]
    ↓ Decoder (Skip Connections)
    dec5 → dec4 → dec3 → dec2 → dec1
    ↓
Output: [B, 6, 512, 512]  (2 heightmaps + 4 alpha channels)
```

### Output Tensor: 6 Channels
| Channel | Name | Purpose |
|---------|------|---------|
| 0 | Global Heightmap | Normalized to [-1000, 3000] world units |
| 1 | Local Heightmap | Normalized to tile-local min/max |
| 2-5 | Alpha Layers | Texture blend weights (future use) |

---

## Part 3: Training Console Output Guide

### Example Output
```
Epoch 15/500: 100%|████| 250/250 [02:30<00:00, 1.67batch/s]
  Train Loss: 0.0234 | Val Loss: 0.0189 | Best: 0.0185
  HM_G: 0.012 | HM_L: 0.018 | Edge: 0.045 | SSIM: 0.034
  LR: 9.5e-05 | Patience: 12/50
```

### Key Metrics Decoded

| Metric | Description | Good Value | Bad Sign |
|--------|-------------|------------|----------|
| **Train Loss** | Combined training loss | < 0.05 | Stuck > 0.1 |
| **Val Loss** | Validation loss | ≤ Train Loss | Val >> Train = Overfitting |
| **HM_G** | Global heightmap MSE | < 0.02 | > 0.05 |
| **HM_L** | Local heightmap MSE | < 0.02 | > 0.05 |
| **Edge** | Edge preservation loss | < 0.10 | > 0.20 |
| **SSIM** | Structural similarity | < 0.05 | > 0.10 |
| **LR** | Learning rate | Decays over time | Stuck = plateau |
| **Patience** | Early stop counter | Resets on improvement | 50/50 = Stop |

### Loss Weights (Default)
```python
LOSS_WEIGHTS = {
    "heightmap_global": 0.15,  # Global altitude accuracy
    "heightmap_local": 0.35,   # Local detail accuracy
    "alpha_masks": 0.10,       # Texture prediction
    "bounds": 0.05,            # Height range prediction
    "ssim": 0.05,              # Structural similarity
    "gradient": 0.05,          # Smooth transitions
    "edge": 0.25,              # Sharp cliff preservation
}
```

---

## Part 4: Visual Validation

### Training Previews
Saved to: `vlm_output/previews/epoch_{N}_best.png`

**Layout per row:** `Minimap | Normal | Water | Predicted | Ground Truth`

### What to Look For
| Issue | Symptom | Fix |
|-------|---------|-----|
| Flat output | All gray, no variation | Check normalization, increase LR |
| Noisy output | Random speckles | Reduce LR, add blur |
| Blurry cliffs | Soft edges | Increase `edge` weight |
| Water troughs | Deep holes in water | Check water mask loading |
| Checkerboard | Grid pattern | Reduce batch norm momentum |

---

## Part 5: Inference

### Command
```bash
python infer_v7.py --model best.pt --dataset 053_Azeroth_v30 --out results --debug
```

### Debug Output
Each tile produces:
- `{tile}.obj` - Textured 3D mesh
- `{tile}_height.png` - 16-bit heightmap
- `{tile}_debug.png` - Composite: `MM | NM | WDL | Water | Pred`

---

## Quick Reference

### Start Training
```bash
python train_v7.py
```

### Monitor Progress
1. Check `vlm_output/previews/` for visual samples
2. Watch for Val Loss improvement
3. Training auto-stops after 50 epochs without improvement

### Resume Training
The script auto-resumes from `best.pt` if present.
