# Contract: v7 Clean-Signal Model Lane

## Inference input

```text
clean_observation_luma_256       float32[1,256,256] in [0,1]
clean_observation_gradient_256   float32[2,256,256] finite, versioned gradient transform
clean_observation_confidence_256 float32[1,256,256] in [0,1]
```

The concatenated model input is exactly `[4,256,256]`. The model must not accept or inspect WDL,
height, normals, liquid, object, alpha, material, or target-derived arrays in inference mode.

The input artifact uses `v7-clean-signal-input-v1`. Gradients use the deterministic
`finite-difference-edge-v1` transform. `confidence_status` is `measured` or `absent_explicit` for
admitted rows; the latter requires a zero-filled confidence channel. `rejected` and `quarantined`
rows remain visible to the gate report but cannot be assembled for inference.

## Model output

```text
coarse_relief_257      float32[257,257]
detail_residual_257    float32[257,257]
height_prediction_257  clamp(coarse + detail) under the relative-height contract
```

Training targets use `v7-clean-signal-target-v1`: per-tile relative height with a denominator
floor of `1.0`, followed by `box9-edge-replicate-v1`. The stored detail is exactly
`relative_height_257 - coarse_relief_257` within the published float tolerance; it is not an
inference input.

The model identity uses `v7-clean-signal-model-identity-v1` and binds architecture, profile,
feature widths, input/output schemas, detail scale, random-initialized status, parameter count,
and `config_sha256`. A checkpoint may be reconstructed only when the identity hash and parameter
count match the rebuilt model. No external or pretrained weights are admitted in this phase.

Corpus manifests use `v7-clean-signal-corpus-v1`. Each row stores the seven named arrays, SHA-256
hashes, source kind/group, split, confidence/gate status, observation provenance, and an explicit
empty forbidden-signal list.

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
