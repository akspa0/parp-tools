# Contract: v7 Clean-Signal Model Lane

## Inference input

```text
clean_observation_luma_256       float32[1,256,256] in [0,1]
clean_observation_gradient_256   float32[2,256,256] finite, versioned gradient transform
clean_observation_confidence_256 float32[1,256,256] in [0,1]
```

The concatenated model input is exactly `[4,256,256]`. The model must not accept or inspect WDL,
height, normals, liquid, object, alpha, material, or target-derived arrays in inference mode.

## Model output

```text
coarse_relief_257      float32[257,257]
detail_residual_257    float32[257,257]
height_prediction_257  clamp(coarse + detail) under the relative-height contract
```

The coarse/detail fields are supervision and diagnostics. Only `height_prediction_257` is the
published terrain result.

## Loss profiles

- `parity`: final point loss plus first-derivative gradient loss.
- `v7_structural_v1`: parity plus full 2D log-spectrum, Laplacian, Sobel edge, transition focus,
  tile-border, and low/high-frequency band losses.

Every term has a recorded weight and a separate validation metric. Adversarial and object/recovery
terms are not part of `v7_structural_v1`.

## Provenance requirements

Every corpus row and run report must record schema versions, source group, split, albedo operation
identity, synthesis parameters, array hashes, architecture identity, loss profile, seed, and
forbidden-signal audit results. Missing or contradictory provenance fails closed.
