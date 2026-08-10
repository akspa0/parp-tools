# RGB Method Benchmark Contract

Schema identity: `v60-rgb-method-benchmark-v1`

## Source classes

- `authored`: authored minimap RGB prepared through `minimap_rgb_to_raw_luma_diagnostic_v1`; runtime-compatible for the current RGB-only observation branch.
- `object_library`: project-owned object-library overlays over `objectified_terrain_shadow_256`; useful as a controlled contamination/withheld-mask diagnostic, but not RGB-minimap runtime evidence.
- `both`: emits separate source reports and split identities; results must not silently merge the modalities.

## Conditions

- `no_mask`: the model receives observation channels only; no object mask is provided.
- `predicted_mask`: the model may receive only a separately produced predicted mask. A source-side or target-side mask cannot satisfy this condition.
- `withheld_mask`: a mask may exist for evaluation/supervision, but it is withheld from model input.

## Required baselines

Every plan declares:

1. `tile_mean_height_v1` for final-height comparison;
2. `identity_observation_v1` for clean/contaminated observation comparison;
3. `zero_predicted_mask_v1` for the predicted-mask condition.

## No-leak rules

- `model_input_arrays` MUST NOT contain `height_257`, `terrain_shadow_256`, `shadow_mask`, WDL, source-side object masks, or target-side object masks.
- `evaluation_only_arrays` MAY contain those arrays when the manifest identifies them as targets and the condition withholds them from the model.
- `runtime_compatible` MUST be false for `objectified_terrain_shadow_256` until a separately approved RGB-observation conversion exists.
- A plan with invalid source provenance fails before training command generation.
