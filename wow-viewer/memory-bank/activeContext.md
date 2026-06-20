# Active Context — wow-viewer

**Last updated**: 2026-06-20 | **Focus**: Spec 068 — training failed, need to debug loss landscape

## Current State

All code for spec 068 is implemented and smoke-tested:
- Spectral loss (FFT radial power spectrum MSE)
- Fractal dim aux head (16×16 Hurst via variogram)  
- Curation hardening (mask precedence, liquid filter defaults)

### Training Result — FAILED

V21c full run with all losses: **val loss 5.66**, stalled immediately, never converged. Baseline 3-channel model achieved 0.69.

### Suspected Root Causes

1. **Spectral weight=0.1 too high** — log-MSE between random init (flat spectrum) and terrain (1/f spectrum) is O(10-50), dominating loss gradient
2. **Multi-scale L1 at weight=1.0 inflates total** — 5 scales sum to ~5x single-scale L1
3. **No per-component val metrics** — can't distinguish L1 vs spectral vs fd contribution
4. **V21c 10-channel model may need different LR or more data**

### Next Session Plan

1. Add per-component loss metrics to validation output
2. Run bare V21c baseline (no extra losses) to get comparable val L1
3. Lower spectral weight to 0.01 and re-run
4. Remove multi-scale L1 or lower its weight to 0.2
5. Only add fractal losses back once baseline is stable

### Files Changed (uncommitted)

- `data-harvester/scripts/train_v16_1_common.py`
- `data-harvester/src/harvester/v16_1_models.py`
- `data-harvester/src/harvester/v16_1_dataset.py`
