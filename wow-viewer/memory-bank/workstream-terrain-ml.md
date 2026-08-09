# Workstream — terrain / minimap ML

Owner specs: 114 (direct terrain reconstruction), 125 (minimap DXT1 inversion),
126 (minimap terrain reconstruction), 111 (minimap lighting calibration).
Last updated: 2026-08-08. **Nothing is training right now.** The active first experiment is Spec 134's
small synthetic control corpus plus a bounded v50 real-mask/pair validation lane; no working v60
real-data corpus has been accepted or generated.

This file is the durable home for the terrain-ML workstream. `activeContext.md` links here and
stays short; put detail here, not there.

## Active route — Spec 134 v60 control corpus

- Start with project-owned deterministic controls, not a broad v50-derived harvest. The default
  control run is 27 families × 4 variants = 108 rows with family-level holdouts and the four
  `easy`/`medium`/`hard`/`pathological` complexity buckets.
- The generator also emits a sibling `object-sieve-v1` derivative with 540 rows. The control taxonomy
  includes mountainous relief, arbitrary-angle sheer drop-offs, zone-style blends, fBm, ridged
  fractal, dendritic lightning-burn terrain proxies, and
  `cross_tile_lightning`/`cross_tile_burn`. Each cross-tile family is one global 2×2 pattern whose
  four tile rows share one `pattern_id`; the validator rejects missing, duplicate, or mixed-ID
  quartets, and the visualizer emits a stitched atlas. Non-grid terrain records deterministic
  sub-cell offsets; only `chunk_grid` is exactly chunk-aligned.
- First signal contract remains `terrain_shadow_256` → `height_257`. Fractal and lightning-burn
  controls are shape probes, not claims about a literal client semantic. The goal is to expose
  partial patterns and high-complexity signals before attempting a tiny albedo-normalized 0.x/1.x
  transfer sample.
- The C# generator, Python validators, object-sieve model/loss variants, and visual reviews are
  implemented. User runs generation,
  client-backed transfer, and GPU training; Codex does not launch them.
- The object-sieve lane is emitted with the terrain controls. Its synthetic input is
  `objectified_terrain_shadow_256`; targets are clean `terrain_shadow_256` plus the distinct
  `object_contamination_mask_256`. Compare clean-only, auxiliary-mask-loss, and predicted-mask-
  guided variants. Do not conflate this screen-space contamination mask with the existing
  `object_geometry_visible_mask_257` numeric geometry target.
- The existing v50.1 mixed curriculum has 1,325 complete authored/legacy-flat same-tile pairs out
  of 1,330 groups. `v60_validate_real_synthetic_pairs.py` writes the validation-only JSON/atlas; the
  first 16-tile Azeroth slice measured mean RGB MAE 0.1812 and RMSE 0.2120. The absolute difference
  is a flat-maptexture diagnostic, not terrain-shadow ground truth. A fresh post-fix C# NPZ with
  `terrain_shadow_256` is required for shadow comparison. Real masks remain labels only; no GPU run
  has started.

## Settled — including the dead ends, which are the expensive part

- **Residual→height feed-forward is dead.** Two runs (uncurated and curated) never beat the
  tile-mean baseline. The "learns then unlearns" oscillation confirmed the target is not learnable
  from single-view shading.
- **Single-view shading cannot recover height.** The forward-model-as-referee (Spec 126 US7) fits
  shading to 0.0103 MAE — 92.9% better than flat — while the recovered height correlates with
  ground truth at **r = 0.0024**. That is the cleanest statement of the limit and should stop this
  family of ideas being re-proposed.
- **The residual extractor works, for albedo-stripping only.** Spec 125 US7, best epoch 54,
  val_mae 0.0893, beats_baseline true, on the curated rolling+steep regimes. Guidance losses
  (multiscale/sobel/spectral/laplacian) were added but only marginally helped.
- **No existing direct minimap→height checkpoint beats the tile-mean baseline.**
  `direct_cnn_v112` (U-Net-lite, 1.56M) v1 best_val_mae 0.1878; v3-deconfounded 0.1723. Both
  `beats_baseline: false`. `mit_b0_regression` (SegFormer, 3.7M) likewise.
- **MCSH is not in minimaps** (measured r = −0.006) and must never be a target or input. General
  rule: the target has to be visible in RGB.

## Ready to run, not yet run

**Stacked height model** — `direct_cnn_v112` extended to 4 input channels (RGB + frozen residual
extractor output) via `--residual-checkpoint`. `HeightRelativeNet` takes `in_channels`.

The crash that blocked it is fixed: the residual channel was built only in the trainer's own
`RowDataset`, while every evaluation path (preview, final eval, road-region, object-region) rebuilt
inputs with RGB+features and handed 3 channels to a 4-channel model. Now one shared builder,
`height_relative_evaluate.build_model_input_channels`, is the single source of truth for channel
order (`RGB -> residual -> features`) across all five call sites. Also: the extractor loads *after*
the dry-run gate (it was allocating CUDA on plan-only runs), and 4-channel `direct_cnn_v112` hashes
to a distinct `config_sha256` (it was colliding with the RGB-only baseline; the 3-channel hash is
unchanged).

**Known mismatch before running**: every extractor on disk (v2–v5) trained on `minimap_rgb_dxt1`,
but `curriculum-0_5_3_3368-dual_v3.zarr` only has `minimap_rgb`. The trainer warns and records
`input_array_matches_training: false`. **Treat the residual channel's contribution as a lower
bound.**

`output/runs/stacked-height-v1/` is the **crashed** attempt (reached epoch 1, val_mae 0.28126,
`input_channels` absent = pre-fix checkpoint). `require_new_output` refuses non-empty dirs, so the
next run needs a fresh path (`stacked-height-v2`). **Do not reuse or delete v1** — it is the negative
record of the crash.

## Ordered plan for the next training session

### Step 1 — decide the curation change BEFORE spending GPU time

Cheap, CPU-only, and it changes what the model sees. Details below.

### Step 2 — stacked height run

`--architecture direct_cnn_v112 --source authored --residual-checkpoint <extractor>`, fresh output
dir. Extractor candidates by best val_mae: **v4 = 0.08840**, v5-guided = 0.08929, v3 = 0.08941,
v2 = 0.09119. v5-guided carries the multiscale/sobel/spectral/laplacian guidance.

Prior split for reference: authored source, 1384 train / 245 val, onecycle + AMP, 100 epochs.

**Record the confound in the run identity rather than discovering it afterwards** — see the
`minimap_rgb_dxt1` vs `minimap_rgb` mismatch above. If the stacked run underperforms, that is the
first thing to rule out, **not** evidence the channel is useless.

### Comparison targets

- tile-mean and flat baselines (computed in-run)
- `SPEC112_FROZEN_BEST_VAL_MAE = 0.1492665126`
- SC-001 requires beating all three by 5% relative
- prior direct runs: v1/cnn 0.1878, v3-deconfounded 0.1723 — neither beat baseline

### Explicitly not on the path

- Full-corpus multi-client harvest — dropped, see `weak-signal-tile-archaeology.md`
- Residual→height feed-forward — dead (r = 0.0024, three approaches agree)
- Spec 127 viewer explorer — drafted, unimplemented, not a training dependency

## Curation change to decide before spending GPU time

`surviving_height_levels` (distinct heights per tile) should gate curation in **both** directions:

`surviving_height_levels` lives in `tile_inventory.py`. Measured on the 0.5.3 corpus:

- **Exclude**: 127 tiles currently classified usable/terrain_no_minimap hold ≤64 distinct heights.
  Four Azeroth tiles (29_24, 30_24, 31_24, 32_24) hold exactly **2** across a 516-unit range. Under
  the v112.1 per-tile min-max target these become perfect binary step functions — actively teaching
  the model that a texture edge is a 500-unit vertical wall.
- **Admit**: 26 compressed-rich tiles are excluded by curation today but their target is **already
  correct**. Amplification above `RANGE_FLOOR` is provably a no-op on the target (verified: ×1 and
  ×1,000,000 give bit-identical targets, because min-max normalisation cancels affine scale). They
  need un-excluding, nothing more.
- 7 further rich tiles sit below `RANGE_FLOOR = 1.0`; those need the floor gated on level count.
  **Do NOT lower the floor globally** — it would amplify the 144 two-to-eight-level tiles into
  full-range targets.

Not yet implemented. It came out of the tile-archaeology work (see `weak-signal-tile-archaeology.md`,
parked) and is the one training-relevant result from it.

## Key files

- `src/harvester/v50/terrain_lighting_torch.py` — differentiable forward model
  (height→normals→Lambert shading)
- `src/harvester/v50/residual_extractor_infer.py` — 4-panel visual-review contact sheets
- `src/harvester/v50/direct_geometry_model.py` — `direct_cnn_v112` accepts 4 input channels
- `src/harvester/v50/height_relative_model.py` — `HeightRelativeNet` accepts `in_channels`
- `src/harvester/v50/direct_geometry_train.py` — `--residual-checkpoint`, frozen extractor preprocessing
- `src/harvester/v50/residual_extractor_train.py` — guidance-loss flags
- `scripts/v50_refine_height_from_residual.py` — forward-model-as-referee refinement
- `scripts/v50_deploy_height_to_mesh.py` — minimap→MiT-B0→height→OBJ deploy

## Constraints specific to this workstream

- **The user runs all training, capture, and GPU work.** Hand over the exact command; never launch
  it.
- No DepthAnything / multi-head / shared-weight model paths.
- Never validate on PVPZone02 or Kalidar; use Kalimdor and Azeroth.
- Constitution IV: per-signal evidence. A strong signal must never mask a dead one, so every signal
  is reported against its own baseline, never rolled into an aggregate score.
