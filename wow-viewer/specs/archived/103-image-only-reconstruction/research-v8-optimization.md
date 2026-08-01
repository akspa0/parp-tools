# Research — v8 optimization lane: slimming the v7 regressor with 2025–2026 techniques

**Date**: 2026-07-13 | **Status**: DECIDED + IMPLEMENTED (same day) — **USER decision: v8 is
the primary lane.** The blocking pain was time-to-signal: a v7 run costs ~26 h before the model
proves sound or not. v7 stays available as `--arch v7` for ablation only; no baseline-first
gatekeeping. Implemented as `src/harvester/spec103/v8_model.py` (`V8LeanUNet`,
`v8-lean-convnextv2-v1`): **measured 6,204,198 params (25 MB fp32), 16.4 GFLOPs @256** —
18.9× fewer params, 7.3× fewer FLOPs than v7. Trainer default `--arch v8`; checkpoints record
the arch; inference auto-resolves; 6 CPU sanity tests (13/13 suite green).
Nothing here weakens the pinned 13-ch contract.

## 1. The measured problem

`MultiChannelUNetV7` at `output_size=256` (measured 2026-07-13, `.venv` torch):

| stage | params | note |
| --- | ---: | --- |
| enc1–enc4 | 4.70M | fine |
| enc5 (512→1024) | 14.16M | 16×16 res |
| **bottleneck (1024→2048)** | **56.64M** | **8×8 res** |
| **dec5 (2048→1024)** | **28.32M** | 16×16 res |
| dec4–dec1 + ups | 10.09M | |
| bounds FC (2048→512→4) | 1.05M | |
| **total** | **117.06M** | 468 MB fp32, **119.9 GFLOPs @256²** |

73% of the parameters (bottleneck + dec5 + up5) operate at 8×8–16×16 resolution. The task —
smooth 257² height field from a 256² minimap *with a 17×17 WDL prior already supplied as the
trestle base* — is low-frequency residual regression. This is the 2015-era "double width every
level" habit; modern dense-prediction practice caps widths at 256–512 and spends capacity on
better blocks, not wider bottlenecks. With 2,253 tiles in the recorded all-gates clean manifest
(2,650 object-free before blank/signal filtering), 117M params is also a memorization risk the
curation pass can't fix.

## 2. 2025–2026 landscape (papers/code surveyed 2026-07-13)

**Directly on-task (height-from-image with a coarse prior):**
- *Seamless High-Resolution Terrain Reconstruction* (arXiv 2507.09681) — RGB + low-res SRTM
  prior → high-res DEM, <5 m MAE vs LiDAR. Validates our exact pattern (coarse height prior +
  image → refined height). ViT-based; the *pattern* transfers, the size doesn't need to.
- *PhiSat-2 / TSONet* (arXiv 2603.29245, 2026) — building height @256², group-wise convs
  throughout, dual-stream decoder, dataset on HuggingFace, code on GitHub. Modern lightweight
  height-regression reference.
- *TSE-Net* (arXiv 2511.13552) — semi-supervised monocular height estimation; relevant later if
  we want to exploit unharvested maps' minimaps.

**Efficient dense-prediction architectures:**
- **ConvNeXt-V2-style U-Net** (ConvUNeXt; Sci Reports 2025 lightweight-UNet variants): 7×7
  depthwise + pointwise MLP + GRN blocks, capped widths, pixel-shuffle decoder. 3–10M params,
  fully deterministic, plain PyTorch — no exotic kernels, works on Windows + RunPod unchanged.
- **EfficientViT (MIT han-lab)** (arXiv 2205.14756): multi-scale *linear* attention for
  high-res dense prediction; pretrained weights on HF; global receptive field at conv cost.
  Would need the 13-ch stem inflated (zero-init extra channels).
- **Mamba U-Nets** (LightM-UNet ~1M params; UNetMamba for remote sensing): linear complexity,
  tiny — but custom CUDA scan kernels are painful on Windows, determinism less audited.
  Exploratory only.

**Training recipe (2026):**
- **Muon / schedule-free Muon / AMUSE** (arXiv 2605.23061, 2509.15816): spectral optimizer for
  conv/linear weights (AdamW for norms/bias), ~2× data-efficiency claims, no LR schedule to
  tune. Low-risk to offer as `--optimizer muon` beside the existing AdamW+cosine.
- Existing trainer conveniences (AMP/EMA/warmup+cosine/early-stop, prior dropout,
  `val_no_prior`) carry over unchanged; add `torch.compile` + bf16 where the GPU allows.

**Excluded (with reasons):**
- **Depth-Anything family** — blacklisted for terrain work (non-deterministic outputs; half a
  month lost). Not as backbone, not as teacher.
- **DepthPro / UniDepth / other depth foundations** — 100M+ ViT foundations; fine-tuning them
  on 2.6k stylized minimap tiles is the same overkill in new clothes, and violates the spirit
  of the image-only law's determinism/auditability.
- **Diffusion dense predictors** (Lotus, Marigold-E2E, DenseDiT) — sampling is
  non-deterministic; label-free validation (FR) needs reproducible outputs.

## 3. Options, ranked

| option | params | risk | notes |
| --- | ---: | --- | --- |
| **A. v8-lean** — ConvNeXt-V2-ish U-Net: widths 32-64-128-256-384, 7×7 DW + GRN, pixel-shuffle up, keep trestle + bounds head + 13-ch + loss unchanged | ~4–8M | low | recommended primary; pure PyTorch drop-in behind the existing trainer |
| **B. v7-slim** — v7 blocks verbatim, widths ÷2 (→~29M) or ÷4 (→~7.3M) | 7–29M | minimal | cheapest ablation; isolates "was it ever the width?" — one constructor arg |
| **C. EfficientViT-B1 encoder + light fusion decoder** | ~9–15M | medium | pretrained global attention; stem surgery for 13 ch; HF weights |
| **D. LightM-UNet-style Mamba** | ~1–5M | high | Windows kernel pain; exploratory only |

All options preserve: 13-ch input order (loss reads ch 9/11/12 — load-bearing), WDL trestle
residual (`global = wdl_base + tanh(δ)·scale`), 0.5 prior fill + dropout, bounds head, 256
output, `v7_losses.combined_loss`. That keeps every existing script (store builders, inference,
mesh export, label-free harness) untouched and results directly comparable.

## 4. Sequence (as decided — v8 first, USER runs all training)

1. **v8-lean is the primary lane** (implemented, option A): synthetic corpus first
   (quickstart §1, minutes-to-signal), then curated V18 (§3). `val_previews/` +
   `noprior_l1_g` tell sound/not-sound early instead of at hour 26.
2. **v7 (`--arch v7`) is ablation only**, run if and when v8's result needs a reference point.
3. Optimizer ablation (AdamW+cosine vs Muon/schedule-free) piggybacks later if wanted.
4. v8 feeds T019 distillation as the STUDENT skeleton — a slim student was the plan anyway.

**v8-lean as built:** ConvNeXt-V2 blocks (7×7 reflect-padded depthwise + pointwise MLP + GRN),
widths 32-64-128-256-384, depths 1-1-2-2-2, pixel-shuffle decoder (checkerboard-free), pooled
global-context mixer + bounds head at the 16× stage. Head/trestle/clamp semantics copied
verbatim from v7 — `combined_loss`, trainer, inference, previews, mesh export, and the
label-free harness all run unchanged.

## 6. Banding investigation (2026-07-14) — where v7's reported banding likely came from

The user reported v7's outputs had visible banding and asked two questions: (1) does v8 avoid
it, (2) is precise numeric data (height, WDL prior) accidentally being routed through 8-bit
image encoding somewhere, since only the deployment minimap should be image data.

**Verified against the live V18 zarr store** (`zarr.open_group(...); a.dtype` per array):
`height_257`, `normal_xyz`, `liquid_mask`, `liquid_height`, `object_precise_mask` are all
**float32**. Only `minimap_rgb` is `uint8` — correct, since it is the one signal that is
genuinely the deployment image. The WDL prior (`v7_inputs.wdl_lattice_from_height257`) is a
`::16` subsample of the float32 height array, done at batch time — never rasterized. The
`wdl_height_33` prohibition already recorded in this lane is exactly this class of mistake,
caught previously. **Conclusion: no, precise data is not going through image quantization.**
One quantization source is real but external: WoW's own ADT `MCNR` chunk stores terrain
normals as signed bytes per axis (~256 directions) — a client file-format limit inherited by
`normal_xyz`, not introduced by this pipeline, and not fixable without different source data.

**Two real, code-level causes were found and fixed:**

1. **`output_head_mode` was wired into both model constructors but never exposed to the
   trainer** — every run silently used `legacy_clamped`: `global_delta = tanh(raw) * scale`,
   then `clamp(wdl_base + global_delta, 0, 1)` applied every step, including during training.
   tanh saturates quickly, so `global_delta` clusters near ±`global_residual_scale` rather than
   spanning it continuously — a plausible mechanism for terraced/banded output relative to the
   WDL base. Fixed: `--output-head-mode {legacy_clamped, linear_unclamped_train}` is now a
   trainer flag (default stays `legacy_clamped` for v7-parity), recorded in checkpoints, and
   resolved automatically by `infer_spec103_v7.py`. `linear_unclamped_train` drops the tanh,
   clamps only at eval time, and is cheap to A/B on the same data/checkpoint layout.
2. **v8-specific, self-introduced risk: PixelShuffle without ICNR init.** Random-init
   pixel-shuffle decoders reliably show checkerboard/grid artifacts (Aitken et al. 2017,
   arXiv:1707.02937) — a failure mode v7 never had (it upsamples via bilinear + 1×1 conv, not
   PixelShuffle). Fixed: `_icnr_init` in `v8_model.py` makes each contiguous group of 4 channels
   feeding one 2×2 PixelShuffle output block start identical (a smooth upsample at init, not
   noise); verified live (sub-kernels bit-identical at init) and guarded by
   `test_icnr_init_avoids_pixelshuffle_checkerboard`.

**One structural contributor left as-is (shared equally by v7 and v8, not a bug):** the WDL
prior is a 17×17 grid bilinearly upsampled to 256×256 — visibly faceted every ~16px, since
bilinear upsampling of a coarse grid is only C0-continuous at the original control points. The
global residual head only gets ±`global_residual_scale` (0.20 default) of room to correct this,
so the prior's facets dominate large-scale structure. This is the documented trestle design
(`global = wdl_base + residual`), not a bug — changing it is a bigger design call than this pass
scoped, but it is the most likely remaining source of any residual faceting/banding tied to the
16px WDL grid spacing, worth watching for in `val_previews/` on the real-data run.

## 7. Sources

- https://arxiv.org/abs/2507.09681 (prior-based terrain reconstruction)
- https://arxiv.org/html/2603.29245v2 (PhiSat-2 TSONet; PHDataset on HF)
- https://arxiv.org/pdf/2511.13552 (TSE-Net)
- https://arxiv.org/abs/2205.14756 (EfficientViT)
- https://arxiv.org/html/2403.05246v2 (LightM-UNet), https://arxiv.org/html/2408.11545v1 (UNetMamba)
- https://www.nature.com/articles/s41598-025-16683-1 (lightweight DWSC U-Net, 2025)
- https://www.sciencedirect.com/science/article/pii/S0031320324008197 ("Pixel shuffling is all you need")
- https://arxiv.org/pdf/2605.23061 (schedule-free spectral/Muon), https://arxiv.org/pdf/2509.15816 (Muon convergence)
