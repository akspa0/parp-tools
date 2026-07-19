# Data Model: Direct Minimap-to-Terrain Reconstruction

## Design rules

- The authored minimap is the deployment root. Every other inference input is generated and names
  the checkpoint that produced it.
- Training truth is never silently substituted for a generated upstream signal.
- Authored and synthetic views of the same terrain row share one `source_group_id` and split.
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

## GeometryCurriculumRow

One image view with one direct relative-height target.

| Field | Type | Rules |
|---|---|---|
| `row_id` | string | Stable identity for this view |
| `source_group_id` | string | Same for authored and synthetic views of one terrain tile |
| `split` | enum | Copied from `ReconstructionSourceRow` |
| `input_origin` | enum | `authored` or `synthetic_noon_white` |
| `rgb` | signal reference | 256 input used by the model |
| `generated_object_mask` | generated signal reference or null | Names checkpoint and inference run when used |
| `relative_height_257` | signal reference | Exact numeric target |
| `target_decode` | object | Relative-height scale/offset and invalid-value policy |

The base geometry comparison uses RGB only. Mask-guided geometry is a later, separately recorded
ablation and must consume generated masks for training and evaluation.

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
