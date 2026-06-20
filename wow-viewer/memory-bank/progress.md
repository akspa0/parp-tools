# Progress — wow-viewer

## 2026-06-20 — Spec 068: fractal-aware height loss + curation hardening

### What landed

**Spectral Loss (Phase 1)**:
- `_spectral_loss()`: masked height → `torch.fft.rfft2` → log-log MSE of radial power spectra
- `--spectral-weight 0.1` CLI flag, wired into `_v21_height_loss`

**Fractal Dim Aux Head (Phase 2)**:
- `_FractalDimHead`: tiny 32→8→1 conv head on backbone pooled16 → H per 16×16 patch
- `V21HeightModel` always creates fd_head, stores `_pooled16` during forward
- `_fractal_dim_target()`: Matheron variogram at lags 1/2, clamp H ∈ [0.1, 0.9], patch weight from mask coverage
- `--fractal-dim-weight 0.05` CLI flag

**Curation Hardening (Phase 3)** — critical fix:
- Fixed mask precedence in `v16_1_dataset.py:278-283`: `object_precise_mask` now checked before `object_filtered_mask` (was reversed, training used coarse AABB instead of precise mesh)
- `--curation-max-liquid-coverage` default 1.0 → 0.05
- Auto-enforced sensible curation defaults for height/v21 tasks when manifest provided
- Added `liquid_coverage` to curation gate logging

### Previous progress (June 18-19)
- V20 Multi-Modal Chained Terrain Intent — segmentor training setup
- V19 minimal-signal height regressor — dataset patching, model, loss functions
- PM4 surface correlation matcher — collision fingerprints, generator validation, WMO surface DB
- Hull fingerprint matcher (ABANDONED — false positives)
- PM4 simplification algorithm reverse-engineered
