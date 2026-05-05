# V12 Texture Decomposer — Training Guide (Stage 1)

## Prerequisites

- NPZ cache with MCAL/MCLY data at `output/tmp/v11_cache/` (~7000 tiles)
- Pre-computed composited residuals at `output/tmp/maptextures/`

If residuals are missing, generate them:

```powershell
& 'gillijimproject_refactor/.venv-train/Scripts/python.exe' `
  wow-viewer/scripts/precompute_maptextures.py `
  output/tmp/v11_cache/v9_tensor_cache_manifest.json `
  --harvest-dir output/tmp/tilesets --output-dir output/tmp/maptextures
```

## Train

```powershell
& 'gillijimproject_refactor/.venv-train/Scripts/python.exe' `
  wow-viewer/scripts/train_v12.py `
  output/tmp/v11_cache/v9_tensor_cache_manifest.json `
  --output-dir runs/v12_stage1 --epochs 200 --batch-size 16 --max-samples 2000 `
  --residual-dir output/tmp/maptextures --num-workers 2
```

### Arguments

| Arg | Default | Note |
|-----|---------|------|
| `input` | (required) | NPZ manifest or paths |
| `--output-dir` | `runs/v12_stage1` | Checkpoint directory |
| `--epochs` | 200 | Total epochs |
| `--batch-size` | 16 | Fits ~8GB VRAM (29.5M model) |
| `--lr` | 2e-4 | Peak learning rate |
| `--max-samples` | 2000 | Limit training tiles (set higher for full data) |
| `--num-workers` | 2 | DataLoader workers |
| `--residual-dir` | None | Path to `_composited.npz` files |

## Checkpoints

Saved to `--output-dir`:
- **`best.pt`**: Lowest validation MCAL L1. Contains `{model, epoch, vocab}`.
- **`last.pt`**: Latest epoch. Contains `{model, optimizer, epoch, vocab}`.

MCLY vocabulary maps texture IDs to class indices. Saved with checkpoint.

## Expected Training Curve

### Total loss (val, MCAL component ×5 removed for readability)

| Epoch | MCAL L1 | MCLY CE | Residual L1 | Total (raw) |
|-------|---------|---------|-------------|-------------|
| 0 | ~0.40 | ~2.9 | ~0.15 | ~2.7 |
| 10 | ~0.08 | ~2.5 | ~0.06 | ~0.9 |
| 50 | ~0.05 | ~2.0 | ~0.04 | ~0.6 |
| 100 | ~0.04 | ~1.6 | ~0.03 | ~0.5 |
| 200 | ~0.03 | ~1.3 | ~0.025 | ~0.45 |

### Interpretation

- **MCAL L1 < 0.05**: Good — model learns alpha patterns, not all-zeros
- **MCLY CE < 1.5**: Texture classification is working. Random baseline for 27 classes is ln(27) ≈ 3.3
- **Residual L1 < 0.04**: Residual captures shading well

## What Comes After

```
Stage 1 (this model):
  minimap → { MCAL, MCLY, residual }

Stage 2 (height model, not built yet):
  residual → heightmap (256×257 float)

Inference pipeline:
  minimap → Stage 1 → residual → Stage 2 → heightmap → OBJ mesh
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `BadZipFile` | Corrupted NPZ | Handled gracefully — sample skipped via `has_*` masks |
| CUDA assert `cur_target >= 0` | MCLY target out of range | Should not happen with `ignore_index=-100`. If it does, a non-vocab ID leaked through. |
| MCAL all zeros at E10 | Model hasn't converged yet | Pretrained backbone, but MCAL head is random. Give it 20-30 epochs. |
| Residual all zeros | `--residual-dir` not set | Add `--residual-dir output/tmp/maptextures` |
| OOM | Batch too large | Reduce `--batch-size` to 8 |
| Slow data loading | `--num-workers 0` | Set to 2-4 for parallel loading |
| Val loss not improving after E50 | Learning rate too high | Lower `--lr` to 5e-5 or add gradient clipping |
| MCLY CE = ln(N_classes) | Model predicting uniform random | Check `--max-samples` — too few samples may cause overfitting |
