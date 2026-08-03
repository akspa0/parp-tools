# Active Context — wow-viewer

Last updated: 2026-08-03

## Session: Spec 125/126 training — residual extractor + stacked height model

- **Spec 125 US7 residual extractor** trained and validated. Best epoch 54, val_mae=0.0893, beats_baseline=True. Guidance losses (multiscale/sobel/spectral/laplacian) added to trainer but showed marginal improvement vs baseline. Extractor converges at ~0.089 MAE on the curated rolling+steep regimes.
- **Residual→height feed-forward** (Spec 125 US4) failed conclusively: two runs (uncurated and curated) never beat the tile-mean baseline. The "learns then unlearns" oscillation confirmed the target is not learnable from single-view shading.
- **Forward-model-as-referee refinement** (Spec 126 US7) built and tested: differentiable hillshade forward model (height→normals→Lambert shading), optimization loop (Adam, shape_loss + affine-fit L1 + TV). Shading fits to 0.0103 MAE (92.9% better than flat), but recovered height correlates at r=0.0024 with ground truth — proving single-view shading does not carry enough information to recover height.
- **Direct minimap→height** (Spec 114): existing `direct_cnn_v112` (U-Net-lite, 1.56M params) and `mit_b0_regression` (HF SegFormer, 3.7M params) checkpoints exist but none beat the tile-mean baseline on their own validation splits. The v1/cnn model: best_val_mae=0.1878, beats_baseline=false. The v3-deconfounded: best_val_mae=0.1723, beats_baseline=false.
- **Stacked height model** (new): `direct_cnn_v112` extended to accept 4 input channels (RGB + frozen residual extractor output). The `--residual-checkpoint` flag loads the trained extractor as a frozen preprocessing step. The `HeightRelativeNet` now accepts `in_channels` parameter. The `build_geometry_model` allows 4-channel `direct_cnn_v112`. All spectral/fractal guidance, OneCycle, AMP, gradient clipping preserved.
- **Residual extractor inference** visual-review tool built: `residual_extractor_infer.py` produces 4-panel contact sheets (minimap, predicted residual, ground-truth residual, stripped albedo).
- **Deploy pipeline** built: `v50_deploy_height_to_mesh.py` chains minimap→MiT-B0→height→OBJ mesh export.

## Tile archaeology (2026-08-03) — parked, see `weak-signal-tile-archaeology.md`

Built and validated a weak-signal/white-plate tile pipeline (inventory, per-tile synthesis, 4-mode
whole-map composites incl. liquid, cross-build version diff). Ran on 0.5.3 (4 maps) and 4.0.0.11927
(Azeroth, Kalimdor, Deephome, Gilneas2, LostIsles). Full-corpus campaign across all clients was
dropped: ~237 GB needed vs 148.5 GB free, and level counting is biased on non-alpha clients until
the MCVT-delta issue is fixed.

**The one training-relevant result**: `surviving_height_levels` (count of distinct heights per tile)
should gate curation in BOTH directions — it excludes 127 tiles currently in the corpus that hold
<=64 distinct heights (four Azeroth tiles hold **2** across a 516-unit range, which under a per-tile
min-max target becomes a perfect binary step function), and admits 26 compressed-rich tiles whose
target is already correct and which are only excluded by curation. Not yet implemented.

## Active specs

| Spec | State |
|------|-------|
| 125 minimap-dxt1-inversion | US7 (residual extractor) trained and validated. US4 (residual→height) proven to not work. |
| 126 minimap-terrain-reconstruction | US7 (forward-model-as-referee) built and tested; single-view shading proven insufficient for height recovery. |
| 114 direct-terrain-reconstruction | Stacked height model (direct_cnn_v112 + residual channel) implemented; needs training. |

## Key files created/modified this session

- `src/harvester/v50/terrain_lighting_torch.py` — differentiable forward model (height→normals→Lambert shading)
- `src/harvester/v50/residual_extractor_infer.py` — visual-review contact sheets for extractor
- `scripts/v50_refine_height_from_residual.py` — forward-model-as-referee refinement
- `scripts/v50_deploy_height_to_mesh.py` — minimap→MiT-B0→height→OBJ deploy
- `src/harvester/v50/direct_geometry_model.py` — modified: direct_cnn_v112 accepts 4 input channels
- `src/harvester/v50/height_relative_model.py` — modified: HeightRelativeNet accepts `in_channels` parameter
- `src/harvester/v50/direct_geometry_train.py` — modified: `--residual-checkpoint` flag, frozen extractor preprocessing
- `src/harvester/v50/residual_extractor_train.py` — modified: added `--multiscale-weight`, `--sobel-weight`, `--spectral-weight`, `--laplacian-weight` guidance flags

## Stacked-height trainer: crash FIXED, ready to run (2026-08-03)

`--residual-checkpoint` crashed at the first best-epoch checkpoint: `expected input[1, 3, 256, 256]
to have 4 channels`. The residual channel was built ONLY in the trainer's own `RowDataset`; every
evaluation path (preview, final eval, road-region, object-region) rebuilt inputs independently with
RGB+features and handed 3 channels to a 4-channel model. Fixed with one shared builder,
`height_relative_evaluate.build_model_input_channels`, now the single source of truth for channel
order (`RGB -> residual -> features`), used by all five call sites. Also: extractor now loads AFTER
the dry-run gate (was allocating CUDA on plan-only runs); `direct_cnn_v112` 4-channel now hashes to a
distinct `config_sha256` (was colliding with the RGB-only baseline, 3-channel hash unchanged);
checkpoints record `residual_extractor` + `input_channels`.

**Known mismatch before running**: every extractor on disk (v2-v5) trained on `minimap_rgb_dxt1`;
`curriculum-0_5_3_3368-dual_v3.zarr` only has `minimap_rgb`. The trainer warns and records
`input_array_matches_training: false`. Treat the residual channel's contribution as a lower bound.

## Known open issues

- Stacked height model has not been trained yet — the crash above blocked it; it is now runnable.
- Existing direct minimap→height checkpoints (v1, v3) don't beat baseline.
- The residual→height feed-forward path is proven dead (beat_baseline=false, r=0.0024).
- The residual extractor is good for albedo-stripping, not height recovery.

## Durable constraints

- `gillijimproject_refactor` is read-only. New code lives in `wow-viewer`.
- User runs training, capture, client-backed proof, and heavy work.
- No DepthAnything/multi-head/shared-weight model paths.
- Constitution IV: per-signal evidence required.