# Implementation Plan: Fractal-Aware Height Loss

**Branch**: `068-fractal-aware-height-loss` | **Date**: 2026-06-20 | **Spec**: `specs/068-fractal-aware-height-loss/spec.md`

## Summary

Add two cheap, O(N log N) loss terms to V21/V21c height training: (1) spectral loss that matches radial power spectrum of predicted vs ground truth height in Fourier domain, and (2) fractal dimension aux head that predicts local Hurst exponent per 16×16 patch as auxiliary supervision. Both are <5% training overhead, directly encode the fractal terrain priors we've repeatedly observed in WoW procedural terrain.

## Technical Context

**Language/Version**: Python 3.11+, PyTorch 2.x

**Primary Dependencies**: `torch.fft.rfft2` (built-in, no new deps)

**Storage**: No new Zarr arrays — fractal dim target computed on-the-fly from `height_norm` in the loss function

**Testing**: Compare V21c val L1 with `--spectral-weight 0.1 --fractal-dim-weight 0.05` vs baseline

**Target Platform**: Training server (CUDA)

**Project Type**: Training loss function + model head

**Performance Goals**: <5% epoch time increase vs baseline

**Constraints**: Must not change V21 model IO shape (aux head is internal — output is still `(B, 1, 257, 257)`). Both losses are gated by train_mask and liquid_mask. Zero-weight = disabled = no behavior change.

**Scale/Scope**: 2 loss terms, ~100 lines of new Python total

## Constitution Check

- **PASS**: All code in `wow-viewer/data-harvester/` — repo independent
- **PASS**: Single-signal model (height_257 output). Aux head is internal backbone supervision, not a second model output.
- **PASS**: No `H:\CLIENTS` references
- **PASS**: Loss terms default to weight=0 — no regression for existing configs

## Project Structure

### Source Code

```text
wow-viewer/data-harvester/
├── scripts/
│   └── train_v16_1_common.py   # _v21_height_loss: add spectral + fractal loss terms
└── src/
    └── harvester/
        └── v16_1_models.py     # V21HeightModel: add fractal_dim_head
```

## Implementation Phases

### Phase 1 — Spectral Loss

**Goal**: Add `--spectral-weight` CLI flag and FFT-based radial power spectrum loss to `_v21_height_loss`.

**Files**:
- `scripts/train_v16_1_common.py` — add `--spectral-weight` arg, implement `_spectral_loss()` function, wire into `_v21_height_loss`

**Approach**:
1. Add `--spectral-weight` (float, default=0.1) to arg parser (only for v21/v21c tasks)
2. In `_v21_height_loss`: if `spectral_weight > 0`, compute:
   - `pred_masked = pred * train_mask`, `target_masked = target * train_mask`
   - Radial power spectrum via `torch.fft.rfft2()` → complex magnitude squared → average over radial annuli
   - `loss = MSE(log(spectrum_pred + eps), log(spectrum_target + eps))`
   - `total_loss += spectral_weight * loss`
3. Add `spectral_loss` to returned metrics dict

**Validation**: Run `train_v18.py v21c --spectral-weight 0.1` for 5 epochs, verify loss decreases and radial power spectrum plots show convergent slopes.

---

### Phase 2 — Fractal Dimension Aux Head

**Goal**: Add a small 16×16 → 1 conv head on the backbone bottleneck, plus Hurst-exponent target computation via variogram.

**Files**:
- `src/harvester/v16_1_models.py` — add `_FractalDimHead` module, add to `V21HeightModel.__init__` and `forward`
- `scripts/train_v16_1_common.py` — add `--fractal-dim-weight` arg, implement `_fractal_dim_loss()`, wire into `_v21_height_loss`

**Approach**:
1. Add `_FractalDimHead(nn.Module)`:
   - Input: `pooled16` (B, 32, 16, 16) from backbone
   - Conv2d(32, 8, 3, padding=1) + ReLU + Conv2d(8, 1, 1) → (B, 1, 16, 16) = H per patch
   - Sigmoid output → H ∈ [0, 1]
2. Modify `V21HeightModel.__init__` to create `self.fd_head` and store `self.fd_enabled` flag
3. Modify `V21HeightModel.forward` to return aux H when `fd_enabled`, else None
4. In `_v21_height_loss`: compute target H per 16×16 patch from `target * train_mask`:
   - For each patch: variogram at lag-1 (mean squared diff of adjacent pixels), Hurst = 0.5 * log2(variogram_lag2 / variogram_lag1), clamped to [0.1, 0.9]
   - Mask patches with <50% train_mask coverage → zero-weight
   - `loss = L1(pred_H, target_H)` with patch weighting
5. Add `--fractal-dim-weight` (float, default=0.05) to arg parser (v21/v21c only)

**Validation**: Train with `--fractal-dim-weight 0.05`, verify aux head outputs H ∈ [0.2, 0.8] and higher H (smoother) on flat terrain regions.

---

### Phase 3 — Integration & Baseline Comparison

**Goal**: Run baseline vs fractal-enabled comparison, verify <5% overhead and val L1 improvement.

**Approach**:
1. Run baseline: `train_v18.py v21c --multiscale-weight 1.0 --gradient-weight 0.05 --normal-consistency-weight 0.05` (existing best config)
2. Run fractal: same + `--spectral-weight 0.1 --fractal-dim-weight 0.05`
3. Compare val L1 trajectory, epoch wall time, radial spectra of predictions

**Validation**:
- Fractal variant should match or beat baseline val L1
- Epoch time <5% overhead
- Radial power spectrum slope β within 10% of ground truth

## Complexity Tracking

None — all constitution checks pass without violations.
