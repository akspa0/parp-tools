# Research: the pinned v7 contract (T002/T003)

**Read-only sources** (never modified): `gillijimproject_refactor/src/WoWMapConverter/scripts/`
`v7_model.py`, `train_v7.py` (V7.5.1 header), `v7_losses.py`, `infer_v7.py`.

Pinned 2026-07-13 by reading the sources directly. This document is the contract the
`wow-viewer/data-harvester/src/harvester/spec103/` port implements.

## 1. The exact 13-channel input order (train_v7.py `__getitem__`, ~line 1494)

| ch | signal | v7 construction | spec103 source (V18 store) |
|----|--------|-----------------|------------------------------|
| 0–2 | minimap RGB | ToTensor → ×(1 − recovery·0.85) → ImageNet Normalize | `minimap_rgb` (256×256×3 u8) |
| 3–5 | normal RGB | ToTensor → ×(1 − recovery·0.70) → ImageNet Normalize; flat (128,128,255) when missing | `normal_xyz` (257×257×3 float in [−1,1]) → RGB = (n+1)/2, resized 257→256 |
| 6 | WDL height prior | `outer_17` (17×17) → `(h − global_min)/range` clip [0,1] → bilinear **align_corners=True** to input size; **0.5 constant fill when missing** | derived: `height_257[::16, ::16]` (the verified outer transform) |
| 7 | height-min hint | constant plane = `(tile_height_min − global_min)/range` | from `height_257` per tile (mode `gt`), or from the prior (mode `wdl`), or 0 (mode `none`) |
| 8 | height-max hint | constant plane = normalized tile height max | same as ch 7 |
| 9 | liquid mask | binary (>0.1) | `liquid_mask` (256×256) |
| 10 | liquid height prior | 16-bit normalized heightmap × liquid mask | `liquid_height` (world units) → normalized × mask |
| 11 | object footprint mask | context ellipses / precise mask | `object_precise_mask` (257→256, >0.5) |
| 12 | brush imprint mask | binary | zeros (V18 has no brush imprints) |

The plan.md table's guess ("alpha, holes_16, chunk metadata" for 7–12) was wrong — the real v7
aux set is the above. `alpha_256`/`holes_16` are **not** v7 inputs. `v7_losses.derive_recovery_mask_from_inputs`
hard-codes ch 9 = liquid, ch 11 = object, ch 12 = brush; this layout must not be reordered.

**Recovery mask** (train + loss): `max(object, liquid, brush·0.5)` → 5×5 max-pool dilation → clip [0,1].
Applied twice: attenuates RGB (×0.85 factor) and normals (×0.70) *before* ImageNet normalization,
and re-derived inside `combined_loss` to up-weight recovery regions (RECOVERY_FOCUS_GAIN 3.5).

**ImageNet normalization**: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) on ch 0–5 only.

## 2. Model (v7_model.py, ported unchanged)

- `MultiChannelUNetV7`: 5-level residual U-Net 64→128→256→512→1024, bottleneck 2048.
  GroupNorm (16 groups, divisor-resolved), reflect padding, bilinear+1×1-conv upsampling, MaxPool2d(2).
- Heads: `out_conv` → 2 channels (global, local); V7.7 adds an optional 3rd detail channel.
  `height_bounds_fc`: 2048-pool → 512 → 4 values.
- **Trestle** (`use_wdl_global_trestle=True`, default head mode `legacy_clamped`):
  `global = clamp(wdl_base + tanh(raw)·0.20, 0, 1)` where `wdl_base = inputs[:, 6:7]`.
  `linear_unclamped_train` mode: `global = wdl_base + raw·0.20`, clamped only at eval.
- Output is bilinearly interpolated to `OUTPUT_SIZE` if needed. Original constant = 512.
- Model variants: `wdl-trestle-reflect-v1` (trestle), `legacy-absolute-v1`, `wdl-trestle-reflect-v77` (trestle+detail).

## 3. Losses (v7_losses.py, ported unchanged)

`combined_loss` = weighted sum: global L1 (0.08), local L1 (0.14), detail aux (0.08·weight),
bounds MSE (0.04), gradient L1 (0.10), SSIM (0.05), Sobel edge (0.12), log-FFT frequency (0.08),
Laplacian (0.12), transition-focus (0.10, gain 3.0), tile-edge border (0.12, width 12),
recovery-focus (0.16, gain 3.5), adversarial (0.12·scale, only when a PatchGAN is attached).

Spec 103 default: `combined_loss` **without** the PatchGAN (adv term absent) — "the way v7 worked",
minus the GAN complexity. `--loss l1` gives pure height regression for ablation.
**No object-mask loss gating** (the recovery term up-weights object regions; it never masks them out).

## 4. Targets, normalization, bounds

- Height normalization: `HEIGHT_GLOBAL_MIN = −1000`, `HEIGHT_GLOBAL_MAX = 3000` (matches the
  converter's constants). `global_target = clip((h − gmin)/(gmax − gmin), 0, 1)`.
- `local_target = (h − tile_min)/(tile_max − tile_min)` (within-tile normalized detail).
- Target tensor = `cat([global, local])`, at OUTPUT_SIZE.
- Bounds target = `[tile_min_norm, tile_max_norm, 0.0, 1.0]` (last two are the global range
  normalized against itself — constants by construction).

## 5. Resolution decision (Phase 0 item 3)

v7 ran INPUT_SIZE = OUTPUT_SIZE = 512 (upscaled 256 minimaps). Our signals are native
256 (minimap/liquid) and 257 (height/normals/precise-mask vertex grids). **Decision: work at 256.**

- The U-Net needs input divisible by 32; 256 is native minimap resolution; 512 doubled memory for
  no information gain (v7 upscaled with bilinear).
- The ported model takes `output_size` as a constructor argument (default 256). This is the **only
  deviation** from the reference: a hardcoded module constant became a parameter. Architecture,
  channel widths, heads, and the trestle path are unchanged.
- 257-grid ↔ 256-raster convention: vertex grids (height_257, normal 257) are resampled with
  bilinear **align_corners=True** (corners map to corners); binary 257 masks with nearest.
  Predicted 256 rasters are resampled back to 257 with the same convention before WDL sampling
  or OBJ export.

## 6. The WDL prior and the paired lattice

- **Input prior (ch 6)** uses the **outer 17×17 only**, exactly as v7's `_render_wdl` did
  (`outer_17` reshaped 17×17, normalized, bilinear align_corners=True upsample).
- The verified paired-lattice transform (`WdlWriter.ExtractTileHeightsFromAlpha`):
  `outer = height257[::16, ::16]` (17×17), `inner = height257[8::16, 8::16]` (16×16).
  The inner grid is not an input channel; it is part of the exported/validated lattice (spec FR-005).
- `wdl_height_33` (`height_257[::8]`) is prohibited everywhere.
- **Missing/dropped prior = 0.5 constant fill** (v7's own fallback). WDL-prior dropout re-uses this:
  a dropped tile gets `ch6 = 0.5` and (in hint mode `wdl`) neutral hints, so one model learns both
  prior-present refinement and prior-absent full prediction.
- **Spec 108 handoff:** the RGB-only prior model writes a row-addressed generated-prior archive
  containing the paired 17×17/16×16 grids. `infer_spec103_v7.py --generated-wdl-priors` uses its
  outer grid for ch6 and derives ch7/8 from that generated grid; it rejects missing rows rather
  than substituting a ground-truth prior. The remaining V8 auxiliary signals are a separate lane.

## 7. Honest caveats (recorded, not hidden)

- v7 consumes height-derived inputs (prior, hints, normals) — it is the reconstruction back half,
  not an image-only model. The image-only front-end (image → generated prior) is a separate lane.
- Height hints (ch 7/8) in mode `gt` leak per-tile GT min/max; mode `wdl` derives them from the
  prior (deployment-consistent). The trainer exposes `--height-hints gt|wdl|none` (default `gt`,
  faithful to v7; use `wdl` for deployment-shaped runs).
- The synthetic capture now uses a canonical one-tile top-down orthographic projection with the
  dataset row/column orientation and a hash-bound lighting sidecar. The store builder also retains
  the old, explicitly labeled procedural fallback (`--synthesize-minimaps`) only as a pipeline
  smoke. Neither fallback is silently presented as a native game minimap; captured LIT colors and
  authored color variants retain different evidence states and rights classes.

## 8. Synthetic caveat catalog (T011 — fill after the user's training run)

| caveat | status |
|---|---|
| resolution / resize convention | decided above (256 + align_corners=True for vertex grids) |
| channel order | pinned above; guarded by tests/spec103/test_v7_sanity.py |
| trestle behavior on synthetic priors | pending first synthetic run |
| loss terms that matter on clean data | pending first synthetic run |
| prior-dropout tiles still resolving | pending first synthetic run |
| shadow↔height correlation (T018) | pending user capture run |
