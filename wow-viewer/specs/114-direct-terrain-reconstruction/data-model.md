# Data Model: Universal Image-to-Terrain Reconstruction

## Design rules

- Any decodable source raster is the deployment root. Every other inference input is generated and
  names the checkpoint that produced it.
- Training truth is never silently substituted for a generated upstream signal.
- Every crop, rendering, or style variant derived from one source shares one `source_group_id` and
  split. Whole visual/source families can be held out as a unit.
- Missing evidence is `unavailable`, never a zero-filled tensor.
- Each model-stage record describes exactly one output signal and one independently replaceable
  checkpoint.

## ReconstructionSourceRow

One leak-safe terrain row before it is expanded into authored/synthetic curriculum views.

| Field | Type | Rules |
|---|---|---|
| `build_id` | string | Exact client build, for example `0_5_3_3368` |
| `map_name` | string | Canonical map identifier |
| `tile_x`, `tile_y` | integer | Source ADT coordinates |
| `source_group_id` | string | Stable hash of build/map/tile and source-store identities |
| `split` | enum | `train`, `validation`, or `test`; immutable for all derived views |
| `authored_rgb` | signal reference | Required for authored-view training; lineage must be real |
| `synthetic_rgb_256` | signal reference | Optional; requires fixed-noon-white provenance |
| `synthetic_rgb_1024` | signal reference | Optional detail truth owned by Spec 113 |
| `relative_height_257` | signal reference | Numeric geometry truth plus decode metadata |
| `normal_257`, `liquid`, `material_ids`, `alpha_stack` | signal references | Optional per-stage truth with honest coverage |
| `object_visibility` | evidence reference | `populated`, `empty`, or `unavailable`; never inferred from RGB difference |
| `upstream_identities` | object | Store manifest, row lineage, renderer revision, and source hashes |

Validation rejects duplicate source identities, cross-split `source_group_id` reuse, synthetic RGB
without `NoonWhiteGlobal` provenance, and declared signals whose coverage action is not real.

## UniversalImageSource

One source raster or paired relief example outside the build-specific v50 terrain family.

| Field | Type | Rules |
|---|---|---|
| `source_id` | string | Stable content-derived identity for the original source |
| `source_group_id` | string | Shared by every crop/render/style/teacher label derived from the source |
| `visual_family` | string | Versioned family such as aerial, natural-photo, artwork, grayscale, procedural, or held-out custom |
| `split` | enum | `train`, `validation`, `test`, or `compatibility`; whole-family holdout policy is recorded |
| `raster` | signal reference | Decodable RGB/RGBA/grayscale image with original size/mode metadata |
| `relief_truth` | signal reference or null | Exact paired view-axis relief when available |
| `teacher_relief` | generated signal reference or null | Pinned teacher/revision/hash/orientation; never silently treated as exact truth |
| `license_or_byod` | object | Distribution/use authority; absent authority is rejected |
| `transform_lineage` | object | Parent source, crop, resize/pad/tile, spatial transform, and style transform |

At least one of exact `relief_truth` or explicitly labeled `teacher_relief` is required for a
training row. Unpaired sources are compatibility/review evidence only.

## ObjectVisibilityEvidence

Trusted top-down object coverage aligned to the authored minimap.

| Field | Type | Rules |
|---|---|---|
| `schema` | literal | `v50-object-visibility-v1` |
| `source_group_id` | string | Joins exactly one reconstruction row |
| `status` | enum | `populated`, `empty`, or `unavailable` |
| `mask` | signal reference or null | Binary or instance-aligned raster; null only when unavailable |
| `coverage_fraction` | number or null | `[0,1]`; required for populated/empty evidence |
| `renderer_revision` | string or null | Required for trusted available evidence |
| `placement_source_hashes` | string array | Verified object-placement/geometry inputs |
| `alignment_evidence` | object | Projection, dimensions, and fixture proof |

An empty mask is valid only when the renderer successfully processed the tile and proved zero
visible object coverage. Failure to render is `unavailable`, not `empty`.

## UniversalReliefCurriculumRow

One image view with one direct normalized view-axis-relief target.

| Field | Type | Rules |
|---|---|---|
| `row_id` | string | Stable identity for this view |
| `source_group_id` | string | Same for every derived view of one underlying source/terrain |
| `visual_family` | string | Drives whole-domain holdout reporting; never a model input |
| `split` | enum | Copied from `ReconstructionSourceRow` |
| `input_origin` | enum | Exact v50, procedural paired, exact external pair, teacher-labeled, or review-only |
| `raster` | signal reference | Preprocessed image plus original dimensions/mode and coverage transform |
| `generated_object_mask` | generated signal reference or null | Names checkpoint and inference run when used |
| `relative_relief` | signal reference | Exact or explicitly teacher-labeled normalized view-axis displacement |
| `target_authority` | enum | `exact_numeric` or `teacher_pseudo`; controls loss/reporting separation |
| `target_decode` | object | Relief orientation, normalization, invalid-value policy, extent, and scale metadata |

The base geometry model consumes the raster only. Teacher identity, visual family, transform
lineage, and all exact numeric guidance are training/evaluation metadata, never inference channels.
Mask-guided WoW geometry is a later, separately recorded ablation.

## TerrainMeshArtifact

Deterministic conversion of one predicted relief field into a terrain artifact.

- Original raster identity and coverage transform.
- Finite normalized relief grid plus output extent, vertical scale, and offset.
- Deterministic grid vertices/triangles, normals, boundary policy, and stitch metadata.
- UV coordinates that cover the complete source image after documented padding/cropping/tiling.
- Model/checkpoint/inference identity and truthful `top_down_height` or `view_axis_relief` semantics.

## TerrainFeatureLibrary

A versioned, deterministic label vocabulary derived from numeric evidence rather than texture names.

- `library_id` and immutable revision hash.
- Canonical classes, explicit `unknown` and `unavailable`, and human descriptions.
- Per-class derivation rules over relative height, slope, curvature, liquid, material, and alpha.
- Family-safe grouping key used to prevent class-family leakage.
- Evidence coverage and collision report for every source row.

The corresponding classifier emits exactly one semantic feature map plus confidence metadata.

## TextureFamilyLibrary and TextureFamilyTarget

`TextureFamilyLibrary` maps build-specific texture/tileset identities to versioned canonical
families. It records source hashes, aliases, unknowns, ordering rules, and family-safe partitions.

`TextureFamilyTarget` contains an ordered family tuple, layer-presence mask, confidence/ambiguity
metadata, and the exact library revision. The texture-family model predicts this target only; it
does not predict alpha weights.

## AlphaStackTarget

One ordered, compositor-compatible blend field.

- Shape/dtype and channel order for the existing four-layer contract.
- Layer-presence mask bound to a `TextureFamilyTarget`.
- Bounded blend values and the existing composition/sum semantics.
- Missing-MCAL, base-only, holes, border, and unsupported states.
- Ground-truth lineage for training plus generated family-selection identity for downstream runs.

The alpha model emits this one stack. Recomposition is evaluation evidence, not an additional
training output.

## ModelStageRecord

One training/evaluation identity for one output signal.

| Field | Type | Rules |
|---|---|---|
| `schema` | literal | `v50-model-stage-run-v1` |
| `stage` | enum | `direct_geometry`, `object_visibility`, `terrain_features`, `texture_families`, `alpha_stack` |
| `output_signal` | string | Exactly one persisted model output |
| `architecture` | object | Local architecture ID, parameter count, config hash |
| `pretrained_source` | object or null | Hub ID, revision/hash, license, and optionality |
| `curriculum` | object | Schema, identity/hash, row counts, split/group audit |
| `upstream_models` | array | Generated-signal checkpoint and inference-run identities |
| `checkpoint` | object | Path, content hash, best epoch, and device-independent config |
| `baselines`, `metrics`, `visual_evidence` | objects | Stage-specific comparison evidence |
| `promotion_verdict` | enum | `pending`, `promoted`, or `rejected` |

## State transitions

```text
candidate
  -> fixture_proven
  -> real_data_ready
  -> user_run_complete
  -> evaluated
  -> promoted | rejected
```

- Code/tests may advance a stage through `fixture_proven`.
- A real corpus audit advances it to `real_data_ready`.
- Only the user launches the heavy operation that produces `user_run_complete`.
- Promotion requires the numeric gates, lineage audit, and user visual review defined by the story.
- Rejected checkpoints remain immutable evidence and are never relabeled as another model line.
