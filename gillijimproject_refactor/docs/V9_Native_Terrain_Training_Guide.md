# V9 Native Terrain Training Guide

This guide documents the current grounded `v9` terrain training path that is active in this repository.

Use this document when you need to understand:

- what the `v9` model is trying to learn
- which dataset signals feed it
- how the main corpus and development-map data are combined
- how to launch, monitor, and resume training
- what a trained model should and should not be expected to do

This is a Bring Your Own Data workflow. Do not ship client data, harvested corpora, model weights, or outputs derived from proprietary game data.

## What V9 Is

`v9` is a native terrain reconstruction model.

It does not try to generate an entire playable map from nothing. The current training claim is narrower and grounded:

- input: terrain-focused minimap evidence plus terrain priors and masking signals
- target: real terrain heights at `17x17`, `65x65`, and `257x257`
- objective: improve terrain recovery beyond the coarse WDL-style prior while staying anchored to real client data and real harvested masks

The model is currently intended to learn terrain shape reconstruction, not final object placement, final texturing, or a fully editable world package.

## Current Data Story

The active `v9` pipeline has two different data lanes.

### 1. Main Direct Corpus

The main training corpus comes from real game clients through `wow-viewer` shared readers.

Current recommended build coverage:

- `0_5_3_3368`
- `0_5_5_3494`
- `0_7_0_3694`
- `3_0_1_8303`
- `3_3_5_12340`
- `4_0_0_11927`

This lane is responsible for broad terrain-family coverage and for keeping the main corpus tied to real client roots instead of harvested image-folder ownership.

### 2. Development-Map Compatibility Lane

The development map remains a compatibility-fed dataset root under `datasets/original_development/development`.

It is still important because it carries object-rich and PM4-rich supervision that does not naturally appear in the direct archive corpus. In practice this means the development map is the current source of:

- `terrain_only_minimap`
- `no_liquid_minimap`
- `no_object_minimap`
- `no_mccv_minimap`
- explicit object masks
- explicit precise object masks
- explicit PM4 masks

The main direct corpus gives scale and breadth. The development map gives special supervision for reconstruction around placements, footprints, and development-era terrain cleanup cues.

## Harvested And Cached Features

The compatibility cache builder at `src/WoWMapConverter/scripts/build_v9_native_tensor_cache.py` currently exports the following training-facing surfaces when present:

- `minimap_rgb_256`
- `normal_rgb_256`
- `height_17`
- `height_65`
- `height_257`
- `wdl_17`
- `height_hints_v7`
- `liquid_mask_257`
- `liquid_height_257`
- `object_mask_257`
- `object_mask_precise_257`
- `pm4_mask_257`
- `brush_mask_257`
- `hole_mask_16x16`

It also records metadata used by audit and curation, including:

- `height_range`
- `detail_energy`
- `minimap_variance`
- `minimap_gradient`
- `liquid_coverage`
- `object_coverage`
- `precise_object_coverage`
- `pm4_coverage`
- `brush_coverage`
- `hole_coverage`
- `minimap_source`

### Minimap Precedence

When multiple minimap variants exist, the current compatibility cache builder prefers them in this order:

1. `terrain_only_minimap`
2. `no_liquid_minimap`
3. `no_object_minimap`
4. `no_mccv_minimap`
5. raw `image`

That preference is deliberate. The active `v9` line wants terrain evidence first and uses separate masks to explain away liquids, objects, PM4 footprints, brush imprints, and holes.

## Model Architecture

The optimized trainer lives at `src/WoWMapConverter/scripts/train_v9_optimized.py`.

### Input Contract

The current feature contract is `v9-native-inputs.v4`.

The named active input signals are:

- `terrain_only_or_no_liquid_or_no_object_or_no_mccv_or_image_minimap_rgb`
- `normal_rgb`
- `minimap_luma`
- `minimap_detail_gradient`
- `wdl_17_or_height_17_base_prior`
- `height_min_mask`
- `height_max_mask`
- `height_range_context`
- `detail_energy_context`
- `minimap_variance_context`
- `liquid_mask`
- `liquid_height_prior`
- `object_footprint_mask`
- `object_precise_mask`
- `pm4_footprint_mask`
- `brush_imprint_mask`
- `hole_mask_16x16`

Two of those signals are RGB triplets, so the actual model input width is `21` channels.

### Output Contract

The native targets are:

- `height_17`
- `height_65`
- `height_257`

The model predicts terrain in a coarse-to-fine hierarchy:

- coarse `17x17` terrain
- mid `65x65` terrain residual refinement
- full `257x257` terrain residual refinement

This keeps the model anchored to a large-scale base prior while still learning local shape detail.

### High-Level Network Shape

`V9TerrainModel` uses a convolutional encoder-decoder style backbone over the `21`-channel tensor stack.

At a practical level, the important architectural behavior is:

- the model sees terrain-focused RGB evidence instead of raw minimap only
- the model gets a coarse WDL or height prior instead of learning terrain from RGB alone
- masking channels tell it where liquid, objects, PM4 footprints, brush imprints, and holes exist
- context channels tell it whether a tile is flat, high-variance, or detail-heavy before it commits to full-resolution residuals

### Loss Stack

The optimized trainer combines multiple losses rather than a single final-height L1 term.

Current tracked components are:

- `full_l1`
- `mid_l1`
- `coarse_l1`
- `gradient`
- `mid_residual`
- `detail_residual`

This is why the trainer logs several component losses per epoch. The model is being pushed to match both absolute terrain and local slope/detail behavior, not only average height.

## Training Workflow

The optimized trainer follows this sequence:

1. load a cache manifest
2. audit entries against minimap, WDL, height-range, variance, gradient, and WDL-delta gates
3. curate a sane and diverse subset or keep the full sane pool
4. split into train and validation sets
5. optionally load a separate development holdout for dev-eval checkpoint selection
6. preload `.npz` shards into memory
7. optionally enable `torch.compile`
8. train with staged loss weighting and periodic detail-focus epochs
9. save checkpoints and preview images

### Detail-Focus Behavior

The optimized trainer can periodically bias epochs toward the highest-detail tiles and also strengthen detail-oriented losses when the run starts to flatten.

That is important for terrain reconstruction because a model can improve broad average error while still washing out cliffs, trenches, roads, and hand-shaped brush detail.

### Checkpoint Selection

There are two distinct evaluation surfaces:

- validation loss on the curated train-family split
- development-map dev-eval metrics from a separate holdout cache

The current PM4-mixed branch uses `--selection-metric dev_global_mae`, which means:

- `val_loss` still shows whether the model is fitting the main corpus better
- `dev_global_mae` decides which checkpoint is considered best

That is intentional. The development-map holdout is the closest thing we currently have to a stable reconstruction-oriented checkpoint gate for PM4-bearing terrain.

## Recommended Current Pipeline

The current recommended production-like flow is:

1. build the broad direct cache from real client roots with `wow-viewer/scripts/run_v9_direct_pipeline.ps1`
2. build a compatibility cache from `datasets/original_development/development`
3. run `wow-viewer` converter command `dataset-split-pm4` to split the development cache into:
   - PM4-bearing development tiles that get mixed into training
   - non-PM4 development tiles that remain a non-overlapping dev holdout
4. launch `train_v9_optimized.py` on the merged training manifest with the non-PM4 development manifest bound through `--dev-eval-cache-manifest`

This is the current best-understood way to keep PM4-bearing development supervision inside active training without letting the same tiles also decide checkpoint selection.

## What A Fully Trained Model Should Do

The expectation for a good `v9` checkpoint is:

- improve terrain detail over the coarse WDL-style base prior
- preserve large-scale terrain shape instead of inventing unrelated topography
- recover sharper terrain structure where minimap evidence and brush/detail cues support it
- avoid using object, liquid, PM4, and hole regions as if they were ordinary terrain evidence
- generalize across multiple client-era terrain families better than a single-build model

## What A Fully Trained Model Should Not Be Claimed To Do

Do not overclaim the output.

Even a strong `v9` checkpoint does not yet prove:

- final world reconstruction parity
- object placement reconstruction
- texture or alpha map regeneration parity
- full development-map rebuild closure
- runtime viewer signoff

The safe claim is narrower: a trained `v9` model should produce a better terrain height reconstruction than the coarse prior alone, using grounded signals from real client data and development-map compatibility masks.

## Run Outputs

A normal optimized run writes these key outputs under its run directory:

- `best_model.pt`
- `last_checkpoint.pt`
- `previews/`

`last_checkpoint.pt` carries the live training history and feature-contract metadata needed to inspect or resume the run. The optimized trainer now auto-resumes from this file when you relaunch the same run with the same `--output-dir`.

`best_model.pt` is the best checkpoint according to the active selection metric, which may be different from plain `val_loss`.

`previews/` is the first place to look for terrain-shape regressions that scalar losses may not make obvious.

## Resume Pattern

Relaunch optimized training with the same cache manifest and output directory to auto-resume from `last_checkpoint.pt` when it exists:

```powershell
& $PythonExe `
  (Join-Path $RepoRoot 'gillijimproject_refactor/src/WoWMapConverter/scripts/train_v9_optimized.py') `
  $TrainingManifest `
  --output-dir $RunOutputDir `
  --epochs 120
```

You can still override the checkpoint path explicitly with `--resume-from <path/to/last_checkpoint.pt>`.

When resuming, keep the same selection metric and dev-eval manifest unless you are intentionally starting a different experiment.

## Pause Behavior

The optimized trainer no longer pauses every 50 epochs by default.

Current defaults are:

- `--pause-every-epochs 0`
- `--pause-on-stall-epochs 50`

That means the run keeps going through the requested epoch budget unless:

- you enable periodic pauses explicitly, or
- validation has failed to produce a new best checkpoint for `50` epochs and the trainer stops cleanly after writing `last_checkpoint.pt`

## Practical Success Criteria

Treat a run as promising when all of these are true together:

- `val_loss` trends down over time
- dev-eval `model_global_mae` trends down on the non-overlapping development holdout
- the model starts closing the gap to WDL and ideally beats it on the dev holdout
- preview images show better terrain structure instead of smoother but less truthful terrain

If only train or validation loss improves while dev-eval stagnates or regresses, the run may still be overfitting to the main corpus rather than learning the reconstruction behavior we care about.
