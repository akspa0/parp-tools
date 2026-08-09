# Research: Controlled Synthetic-to-Real Reconstruction Experiment

**Feature**: `134-v60-unified-dataset-model`
**Date**: 2026-08-08

## Decisions

### 1. Make project-owned controls the first dataset

The first experiment uses a small deterministic set of terrain families and variants. The current
`WowViewer.Tool.Harvest control-corpus` command already emits exact `terrain_shadow_256` and
`height_257` pairs, family-level validation splits, and per-array hashes. This is the right first
authority because it removes authored-minimap ambiguity and gives us a known target for every row.

The initial target is intentionally small: enough rows to test learning curves and family
generalization, not enough rows to pretend that a historical corpus has been solved. A default run
of 27 families and four variants produces 108 rows; the manifest controls any increase or focused
subset within the bounded experiment size. The three added families are mountainous relief,
arbitrary-angle sheer drop-offs, and zone-style blends.

### 2. Reuse the existing compositor for the synthetic input

`TerrainMinimapCompositor.ComposeShadowArray` remains the only lighting/shadow implementation for
the control lane. The C# `synthetic-minimap --textureless-residuals` path is useful evidence that a
textureless rendering surface exists, but this plan does not duplicate its equation in Python.

Python owns validation, experiment indexing, and reports. It does not regenerate the C# signal.

### 3. Treat real-minimap albedo removal as a new, measured operation

The existing `composite_texture_identity_albedo` helper is a metadata-derived texture identity
representation. It is not an inverse operation that removes albedo from an authored minimap pixel
image. Likewise, a C# synthetic textureless output is a known reference rendering, not proof that
an arbitrary captured minimap can be stripped successfully.

Therefore the real-data step is explicitly named **albedo normalization**:

```text
authored / texture-bearing minimap
    -> albedo estimate and removal
    -> canonical textureless minimap input
    -> textureless quality gate
    -> height reconstruction
```

The operation must emit its method/version, normalized image, estimated albedo or removal record,
and measurable residuals. It must not silently replace a failed result with zeros or with the
ground-truth synthetic shadow.

### 4. Gate non-textured data instead of forcing every tile through the model

The real-data lane processes only tiles whose normalized result passes a textureless quality gate.
The gate records accepted, rejected, and quarantined rows plus the reason and metrics. The first
threshold values are calibrated from synthetic positive controls and deliberately textured or
failed negative controls; they are not guessed once and hidden in code.

This keeps the first model honest: it sees a canonical signal close to the control input contract,
not arbitrary texture-bearing pixels mislabeled as textureless terrain.

### 5. Use a transfer ladder, not a direct leap from synthetic score to full expansion

Control performance is necessary evidence that the model and target relationship are learnable. It
is not sufficient evidence that an authored 0.x/1.x minimap can be normalized into the same domain.
The expansion order is:

1. synthetic controls and family holdout;
2. a tiny accepted 0.x/1.x sample after albedo normalization;
3. a small transfer comparison with the same input statistics and evaluation report;
4. broader processing only if the transfer gate passes.

If the tiny real sample fails, the next action is to diagnose albedo normalization or domain shift,
not to multiply the dataset size.

### 6. Evaluate limited dataset sizes explicitly

The first model experiment reports a small learning curve, such as 8, 16, and 32 training rows, or
the nearest sizes supported by the manifest. Every run uses the same held-out families, seed policy,
model version, and tile-mean baseline. Metrics are reported per family and per signal, with weak or
ambiguous terrain called out separately.

Codex prepares the command and offline checks. The user runs any GPU training, real-client
processing, or long synthesis operation.

### 7. Include fractal and cross-tile controls explicitly

The existing repository already treats terrain art as a full-map/cross-tile phenomenon: its fractal
canvas work preserves tile origins, seams, and region coverage, while the existing fractal helpers
use multi-frequency octaves. The v60 controls therefore add two related but distinct classes:

- local fBm/ridged-fractal and dendritic lightning-burn height patterns;
- cross-tile lightning/burn patterns evaluated in one global 2x2 coordinate system.

### 8. Deliberately break the chunk/cell lattice for authored-style variation

Real terrain can be copied, rotated, or blended without respecting the underlying chunk/cell
boundaries. The control generator therefore records deterministic sub-cell field offsets for every
non-grid family and reserves exact alignment for the explicit `chunk_grid` diagnostic. The
validator requires those offsets and rejects metadata that falsely presents a non-grid pattern as
chunk-aligned. Cross-tile patterns still share one global field and offset so their seams remain a
real continuity test.

The second class is important because a tile-local generator can accidentally teach the model that
every pattern begins and ends at a tile edge. Four rows share one pattern ID and occupy `(0,0)`,
`(1,0)`, `(0,1)`, and `(1,1)` positions. The visual reviewer stitches them before training so seam
continuity is human-visible.

The lightning-burn control is an analytic dendritic stroke modulated by multi-octave fractal field
detail. It is a terrain-shape proxy for branching electrical/charred patterns seen in authored
terrain; it is not a claim that the client stores a literal “wood burn” semantic.

### 8. Treat object removal as a decomposition problem, not a height-loss tweak

The repository already has `object_geometry_visible_mask_257` and loss-side object weighting, but
that target answers a different question: which terrain vertices are visibly occupied by decoded
geometry. A minimap sieve needs a screen-space contamination target: which 256x256 pixels contain
object appearance, object occlusion, or configured object-cast effects that must be removed to recover
the terrain-only shadow.

The first object lane therefore uses an `objectified_terrain_shadow_256` input with two exact outputs:
clean `terrain_shadow_256` and `object_contamination_mask_256`. The mask is both a separately reported
signal and an auxiliary loss target. A third ablation feeds the model's predicted mask into its clean
output head. It never feeds the ground-truth mask as an input, which would create an inference-only
oracle.

The initial object vocabulary is deliberately broad and procedural: tree/rock/building/bridge/
cluster-like silhouettes with controlled scale, density, overlap, and tile-boundary placement. Exact
client asset identity and instance recognition are deferred until binary contamination removal proves
useful. Clean-output error and mask quality are reported separately; a strong mask cannot conceal a
bad inpainted terrain signal.

### 9. Use the real v50 masks for the first object detector

The existing on-disk `curriculum-0_5_3_3368-obj_v1.zarr` contains 2,655 rows, with 1,325 authored
minimap rows and 1,330 synthetic counterparts. Its `object_mask`, `object_precise_mask`, and
`object_instance_mask` arrays are populated on sampled rows; the separate
`object_geometry_visible_mask_257` array is empty in the audit and sample inspection. The object
detector therefore uses authored `minimap_rgb` as input and trains against the real precise and
coarse masks. It does not use the empty geometry-visible array and does not pretend that the store
contains a clean terrain-only minimap target.

The first model is a compact multi-head U-Net with selectable `precise`, `footprint`, or `both`
heads. Checkpoint selection uses the minimum IoU across the requested heads, while reports retain
per-head metrics. RGB is the baseline; an explicit RGB-plus-edge channel is the only first ablation.
This is enough to answer whether real object appearance is detectable before albedo normalization,
without coupling the first experiment to an unproven inpainting target.

### 10. Use the existing same-tile flat rows as absolute-difference diagnostics

The v50.1 mixed curriculum contains 1,330 source groups: 1,325 have exactly one authored and one
synthetic minimap row, while five groups are incomplete. The authored and synthetic rows in a
complete group share the same map/tile identity and object-mask labels. The source manifest says
the synthetic minimap signal is produced by the legacy `synthetic-minimap` path; it is a flat fake
maptexture with no post-fix terrain-shadow target. A deterministic 16-row Azeroth holdout sample
measured mean authored-vs-flat RGB MAE `0.1812`, RMSE `0.2120`, and 69.4% of pixels differing by
more than `0.10` in normalized RGB. That absolute difference is useful observed evidence of what
the authored image adds over flat terrain, but it is not a clean shadow ground truth.

The pair lane therefore writes a validation-only report and visual atlas containing authored RGB,
flat synthetic RGB, and amplified absolute difference. A fresh NPZ from the current C# compositor
may be supplied; it must contain `terrain_shadow_256`, and the report compares its luminance pattern
with the absolute-difference luma as calibration evidence only. The old flat synthetic image is not
fed to the terrain model, and the real masks remain labels only.

## Alternatives rejected

### Full v50/v60 historical harvest first

Rejected for this phase. It makes provenance, era mixing, and missing-signal bugs the first problem,
while providing no controlled answer about whether the first height relationship is learnable.

### Treat all authored minimap pixels as valid input

Rejected. Texture and albedo can become shortcuts or distribution noise. The model should first be
tested on a canonical textureless signal and real tiles should be accepted only after normalization.

### Claim synthetic success proves real success

Rejected. A synthetic score can be high while the real albedo operation produces a different
distribution. The tiny 0.x/1.x transfer gate is required before expansion.

### Generate a fixed 4,000-row corpus

Rejected. Row count is an experimental variable. Increase it only when a measured failure mode says
the control family or variation space is insufficient.

## Open implementation questions resolved by the plan

- **Albedo method**: implement behind a versioned operation contract and compare candidate methods
  on controls plus negative controls; no method is declared correct by name alone.
- **Textureless threshold**: calibrate and persist thresholds in the gate report; no universal
  threshold is assumed before the first calibration run.
- **Real seed count**: begin with a tiny explicit 0.x/1.x manifest and expand only after transfer
  evidence. Later client builds remain out of the initial route.
- **Object guidance**: implement `clean_only`, `auxiliary_mask_loss`, and
  `predicted_mask_guided` ablations. Keep the mask as an exported output and loss-side target; do
  not commit to a joint height/object model until the sieve's clean-output and mask metrics justify
  the extra coupling.
- **Paired validation**: use `v60_validate_real_synthetic_pairs.py` for the small absolute-difference
  report and atlas before GPU work; supply `--shadow-npz-dir` only with fresh post-fix NPZs containing
  `terrain_shadow_256`. Do not use the legacy synthetic RGB as a terrain-shadow model input.
