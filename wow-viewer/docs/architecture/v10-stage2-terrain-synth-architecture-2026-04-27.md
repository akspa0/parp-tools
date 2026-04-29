# v10 Stage 2 Terrain Synth Architecture

This document tracks the current Stage 2 trainer in `wow-viewer/scripts/train_v10_stage2_terrain_synth.py`.

## Current Default

New runs now default to a small, iteration-oriented setup:

- Model variant: `slim_structured_v1`
- Parameters: `679,051`
- Input tensor: `23 x 256 x 256`
- Coarse-prior mode: `zero`
- Default epochs: `120`
- Compact curation default: `--max-selected-fraction 0.25`
- Era cap default: `--max-per-era 128`

The point is to stop burning time on repeated terrain examples. The current broad slim manifest proof selected `717` shards from `3,240` valid preselection shards while retaining all `41` pattern-annotated native v10 rows.

## Architecture Verdict

The architecture is sane for the next training run if we treat it as a compact terrain synthesizer, not a giant all-data memorizer.

The old `structured_fusion_v2` model was already modest at about 2.2M parameters, but the new default is intentionally smaller. `slim_structured_v1` keeps the useful split-stem idea and cuts width:

| Variant | Branch widths | Encoder widths | Params | Use |
|---|---|---|---:|---|
| `early_fusion_v1` | none | 32/64/96/128/160 | legacy | checkpoint compatibility |
| `structured_fusion_v2` | 24/16/16 | 32/64/96/128/160 | ~2.2M | heavier comparison |
| `slim_structured_v1` | 12/8/8 | 16/32/48/64/96 | `679,051` | current default |
| `multi_task_v3` | 24/16/16 | 32/64/96/128/160 | larger | experimental auxiliary heads |

The default should stay `slim_structured_v1` until a compact run clearly underfits.

## Input Contract

The current input tensor has 23 channels:

| Range | Signal | Notes |
|---|---|---|
| `0:3` | `minimap_rgb_256` | Required, `[0,1]` |
| `3:7` | `mcal_alpha_pack_256` | Optional, zeros when missing |
| `7:10` | `mccv_rgb` | Optional, resized to `256x256` |
| `10:13` | `mcnr_normal_xyz` | Optional; legacy `normal_rgb_256` is decoded to XYZ |
| `13:14` | `unified_liquid_mask` | Optional; aliases include `liquid_mask_257`, `wl_liquid_mask`, `mh2o_surface_height` |
| `14:15` | `unified_liquid_height` | Optional; aliases include `liquid_height_257`, `wl_liquid_height`, `mh2o_surface_height`, `mclq_surface_height` |
| `15:16` | `object_mask_257` | Optional placement-derived proxy |
| `16:17` | `object_precise_mask_257` | Optional; alias `object_mask_precise_257` |
| `17:18` | `pm4_path_mask` | Optional; alias `pm4_mask_257` |
| `18:19` | `pm4_building_footprint_mask` | Optional |
| `19:20` | `pm4_mprl_mask` | Optional |
| `20:21` | `hole_mask_16` | Optional; alias `hole_mask_16x16`, upsampled nearest |
| `21:22` | `mtxf_animated_mask` | Optional |
| `22:23` | `coarse_height_17_prior` | Default zeros; `--coarse-prior-mode target` restores old target-fed behavior |

The old default used target `height_17` as the coarse prior. That was useful for a refinement-only experiment, but it is not honest for minimap-driven generation unless inference supplies a Stage 1 prediction. The default is now `zero` to avoid target leakage.

## Signal Groups

`slim_structured_v1` still routes signals through split stems before fusion:

- `surface`: minimap, MCAL, MCCV, normals, coarse prior
- `structure`: object masks, PM4 masks, holes, MTXF
- `liquids`: unified liquid mask and height

This is enough structure to prevent sparse PM4/MCAL rows from being buried in a single first convolution, without making the model expensive.

## Targets And Loss

The height targets remain:

- `height_17`
- `height_65`
- `height_257`

Loss stack:

`full L1 + 0.5 mid L1 + 0.25 coarse L1 + 0.3 gradient + 0.3 mid residual + 0.3 detail residual`

The model does not currently predict MCAL, MCLY, object masks, liquid masks, or PM4 classes unless `multi_task_v3` is explicitly selected.

## Dataset Strategy

The current training problem is too much repeated data, not too little data. The compact curation path is now the default:

- Reject unusable shards first.
- Infer and preserve `era_tag`.
- Attach `pattern_detection` metadata from local Wave 2 dictionaries.
- Prefer quality and pattern-bearing rows.
- Cap selected rows to 25% of valid preselection.
- Cap each era at 128 rows.

Validated compact outputs:

- Full corpus: `output/ml-training/v10_curated/v10_full_corpus_slim_pattern_manifest.json`
  - `3,945` candidates
  - `3,240` valid preselection shards
  - `717` selected shards
  - `41` pattern-annotated native v10 rows retained
- Native development: `output/ml-training/v10_curated/v10_dev_slim_pattern_manifest.json`
  - `64` candidates
  - `41` valid preselection shards
  - `10` selected shards
  - all selected rows carry pattern hints

This is the right direction for iteration. If the model underfits, increase epochs first, then relax curation. Do not jump straight back to thousands of repeated shards.

## Pattern Hints

`pattern_detection` is metadata, not an input channel. Stage 2 uses it for:

- weighted sampling via `--pattern-signal-boost`
- validation subset reporting
- catalog visibility

This keeps existing checkpoint channel shapes stable while still biasing training toward rows where Wave 2 found real repeated structures.

## Proofs

Current sanity proofs:

- Python compile passed for `curate_v10_training_shards.py` and `train_v10_stage2_terrain_synth.py`
- Model instantiation proof: `slim_structured_v1 zero 23`, params `679,051`
- Full-corpus slim curation proof: `717` selected shards
- Native-dev slim curation proof: `10` selected shards
- CPU smoke: `output/ml-training/v10_stage2_slim_arch_smoke/checkpoints/best.pt`
  - `Model: slim_structured_v1`
  - `Params: 679,051`
  - `Input channels: 23`
  - `coarse_prior_mode: zero`

The one-epoch CPU smoke emitted a low prediction-variance warning. That is expected from a tiny one-epoch smoke and is not a model quality claim.

## Remaining Risks

- `ObjectMask257` and `ObjectPreciseMask257` are still placement-derived proxies, not rendered silhouettes.
- Pattern hints are currently strongest for native development rows; broad v9 rows do not yet carry equivalent pattern labels.
- `era_tag` is still `unknown` for native development shards until upstream Stage 1 metadata records a concrete build or era.
- `--stage1-checkpoint` is reserved but intentionally fail-loud until Stage 1 predicted coarse-prior wiring exists.
- Old `early_fusion_v1` ablation numbers remain useful only as a baseline. They do not prove the slim model's signal usage.

## Recommended Next Run

Use the slim compact manifest, more epochs, and the zero coarse-prior default:

```powershell
gillijimproject_refactor\.venv-train\Scripts\python.exe wow-viewer\scripts\train_v10_stage2_terrain_synth.py `
  output\ml-training\v10_curated\v10_full_corpus_slim_pattern_manifest.json `
  --output-dir output\ml-training\v10_stage2_slim_pattern_cuda_run `
  --epochs 160 --batch-size 4 --num-workers 4 --device cuda
```

Use `--coarse-prior-mode target` only when intentionally comparing against older refinement-only behavior.
