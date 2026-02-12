# WoW Height Regressor V7.1 - Complete Guide

## Quick Start

### 1. Export Dataset
```bash
cd src/WoWMapConverter/WoWMapConverter.Cli
dotnet run -- vlm-export --client "H:\053-client\" --batch-all --out "J:\vlm-datasets"
```

### 2. Train Model
```bash
cd src/WoWMapConverter/scripts
python train_v7.py
```

### 3. Run Inference
```bash
python infer_v7.py --model vlm_output/best.pt --dataset "J:\vlm-datasets\053_Azeroth_v30" --out results --debug
```

---

## Model Architecture: 11-Channel V7.1

### Input Channels
| Ch | Name | Source | Purpose |
|----|------|--------|---------|
| 0-2 | Minimap RGB | `images/{tile}.png` | Zone colors, roads, water |
| 3-5 | Normal Map RGB | `images/{tile}_normal.png` | Terrain slope orientation |
| 6 | WDL Height | JSON `wdl_heights` | Low-res global elevation |
| 7 | H_Min Mask | JSON `height_min` | Tile minimum altitude |
| 8 | H_Max Mask | JSON `height_max` | Tile maximum altitude |
| 9 | Water Mask | `liquids/{tile}_liq_mask.png` | Where water = flat terrain |
| 10 | Object Footprint | JSON `objects` → bounds | Where buildings = flat terrain |

### Output Channels
| Ch | Name | Purpose |
|----|------|---------|
| 0 | Global Heightmap | Normalized to world height range |
| 1 | Local Heightmap | Normalized to tile height range |
| 2-5 | Alpha Layers | Texture blend weights (future) |

### Network: 5-Level U-Net
- Input: 512×512 × 11 channels
- Bottleneck: 16×16 × 2048 channels
- Output: 512×512 × 6 channels
- Skip connections at each level
- Auxiliary output: 4-value height bounds predictor

---

## Training Console Output

### What You'll See
```
Epoch 15/500: 100%|████| 250/250 [02:30<00:00, 1.67batch/s]
  Train Loss: 0.0234 | Val Loss: 0.0189 | Best: 0.0185
  HM_G: 0.012 | HM_L: 0.018 | Edge: 0.045 | SSIM: 0.034
  LR: 9.5e-05 | Patience: 12/50
```

### What Each Metric Means
| Metric | Good | Concerning | Action |
|--------|------|------------|--------|
| **Train Loss** | < 0.05 | > 0.1 after 20 epochs | Increase LR or check data |
| **Val Loss** | ≤ Train Loss | Val >> Train | Overfitting - add dropout |
| **HM_G** | < 0.02 | > 0.05 | Global height issues |
| **HM_L** | < 0.02 | > 0.05 | Local detail issues |
| **Edge** | < 0.10 | > 0.20 | Blurry cliffs - increase edge weight |
| **SSIM** | < 0.05 | > 0.10 | Structural similarity poor |
| **Patience** | Resets on improvement | 50/50 | Training will stop |

### Signs Training is Working
1. **Val Loss decreasing** - Model is learning
2. **Patience resets** - New best model saved
3. **Preview images** show recognizable terrain in `vlm_output/previews/`
4. **HM_G and HM_L both decreasing** - Both global and local heights improving

### Signs of Problems
1. **Val Loss stuck** - Learning rate too low or data issue
2. **Val >> Train** - Overfitting, need more augmentation
3. **Previews all gray** - Normalization issue
4. **Previews noisy** - Learning rate too high

---

## When Training is Complete

Training auto-stops when:
- **50 epochs without improvement** (patience exhausted)
- OR **Val Loss < 0.01** (convergence target reached)

### Files Created
```
vlm_output/
├── best.pt              # Best model weights
├── checkpoint.pt        # Latest checkpoint
├── training_log.json    # Full training history
└── previews/            # Visual samples per epoch
    ├── epoch_001_best.png
    ├── epoch_015_best.png
    └── ...
```

### Ready for Inference When
1. `best.pt` exists
2. Val Loss < 0.02 (good quality)
3. Preview images show proper terrain shapes

---

## Running Inference

### Basic Usage
```bash
python infer_v7.py --model vlm_output/best.pt --dataset PATH_TO_DATASET --out results
```

### Debug Mode (Recommended First Run)
```bash
python infer_v7.py --model vlm_output/best.pt --dataset PATH_TO_DATASET --out results --debug
```

### Output Files
```
results/
├── Azeroth_32_48.obj      # 3D mesh
├── Azeroth_32_48_height.png  # 16-bit heightmap
└── Azeroth_32_48_debug.png   # Composite (debug mode only)
```

### Debug Image Layout
`Minimap | Normal | WDL | Water | Objects | Prediction | GroundTruth`

---

## Configuration

### Training Hyperparameters (in `train_v7.py`)
```python
DATASET_ROOTS = ["J:/vlm-datasets/053_Azeroth_v30"]  # Add more datasets
OUTPUT_DIR = Path("./vlm_output")
BATCH_SIZE = 4          # Reduce if OOM
LEARNING_RATE = 1e-4    # Starting LR
EPOCHS = 500            # Max epochs
PATIENCE = 50           # Early stop patience
```

### Loss Weights
```python
LOSS_WEIGHTS = {
    "heightmap_global": 0.15,  # Global altitude
    "heightmap_local": 0.35,   # Local detail (most important)
    "alpha_masks": 0.10,       # Texture layers
    "bounds": 0.05,            # Height range
    "ssim": 0.05,              # Structure
    "gradient": 0.05,          # Smoothness
    "edge": 0.25,              # Cliff preservation
}
```

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|--------------|-----|
| No `_normal.png` files | Old export | Re-run vlm-export |
| Missing water masks | Alpha 0.5.3 limitation | Normal - fallback to zeros |
| Objects all `null` bounds | MDX files not found | Check client path |
| Training OOM | BATCH_SIZE too high | Reduce to 2 |
| Previews show checkerboard | Learning artifacts | Lower LR, more epochs |
| Val Loss = NaN | Data corruption | Check dataset JSONs |
