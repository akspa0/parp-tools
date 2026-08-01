# Data Model: Minimap Deconstruction Engine

## 1. Object Library

### 1.1 ObjectLibraryEntry

Canonical per-asset record.

Required fields:

- `library_id`: stable string ID
- `original_asset_path`: original path from ADT placement name tables
- `normalized_asset_path`: canonical lookup path used by loaders
- `asset_type`: `m2`, `mdx`, or `wmo`
- `capture_status`: `captured`, `failed`, `not_attempted`, `partial`
- `visibility_class`: `roof_visible`, `likely_visible`, `likely_hidden`, `clutter_filtered`, `unknown`
- `review_state`: `unreviewed`, `accepted`, `rejected`, `needs_followup`
- `source_builds`: list of build keys where the asset was observed
- `source_maps`: list of maps where the asset was observed
- `placement_observation_count`: integer
- `preferred_variant_id`: optional pointer to the best capture variant

### 1.2 ObjectCaptureVariant

Concrete captured artifact for one asset.

Required fields:

- `variant_id`: stable string ID
- `library_id`: foreign key to `ObjectLibraryEntry`
- `capture_build`: build key used for lookup/capture
- `capture_mode`: `orthographic_topdown`, `geometry_projection`, `hybrid`, or future value
- `asset_type`: repeated for convenience
- `image_key`: path/key into Zarr image arrays
- `mask_key`: path/key into Zarr mask arrays
- `bbox_local`: local crop bounds relative to the captured canvas
- `rot_x`, `rot_y`, `rot_z`
- `scale`
- `capture_notes`: freeform or nullable
- `capture_confidence`: float in `[0,1]`

### 1.3 Library Zarr Layout

Suggested store root:

`wow-viewer/output/datasets/object-library/<build-or-corpus>.zarr`

Suggested arrays/groups:

- `capture_rgb/` -> `(N, H, W, 3)` or tiled variable-size strategy
- `capture_mask/` -> `(N, H, W)`
- `capture_alpha/` -> optional future artifact
- `index.parquet` -> one row per `ObjectCaptureVariant`
- `assets.parquet` -> one row per `ObjectLibraryEntry`

## 2. Teacher Prior Dataset

### 2.1 TeacherPriorTileRecord

Per tile, generated from ADT-backed supervision.

Required fields:

- `build`
- `map`
- `tile_id`
- `tile_x`
- `tile_y`
- `raw_minimap_key`
- `teacher_object_mask_key`
- `teacher_object_confidence_key`
- `processed_prior_key`
- `has_teacher_objects`
- `teacher_object_cov`
- `filtered_mask_source`: `object_filtered_mask`, `object_precise_mask`, `object_mask`, or `none`

### 2.2 Teacher Prior Arrays

Minimum required arrays:

- `raw_minimap_rgb_256`: `(N, 256, 256, 3)`
- `teacher_object_mask_256`: `(N, 256, 256)`
- `teacher_object_confidence_256`: `(N, 256, 256)`
- `processed_minimap_prior_256`: `(N, 256, 256, C)` where `C` is documented explicitly by the phase

Phase-1 preferred channel philosophy:

- keep `C` small
- do not hide what each channel means
- prefer explicit channels over latent magic

Candidate phase-1 prior channels:

1. raw minimap RGB
2. object-suppressed/fill RGB
3. teacher object mask
4. teacher object confidence

The exact subset used for the first height-only model must be locked by the corresponding phase and documented in the trainer spec.

## 3. Height-Only Training Sample

### 3.1 HeightOnlyTrainingSample

Required fields:

- `input_prior`: `(C, 256, 256)`
- `height_257`: `(1, 257, 257)`
- `weight_257`: `(1, 257, 257)` terrain-valid or object-aware loss weighting; computed from `object_precise_mask` first, then `object_filtered_mask`, then `object_mask` only as fallback
- `meta_build`
- `meta_map`
- `meta_tile_id`

Non-goals in phase 1:

- no normal head
- no liquid head
- no object head
- no shared-weight multitask outputs

### 3.2 Coarse-To-Fine Residual Height Chain

The direct height model remains available for comparison, but the next active
terrain lane splits the problem into two independently trained models.

#### H0: HeightCoarseSample

Inputs:

- `input_prior`: `(C, 256, 256)` processed minimap prior channels
- optional `albedo_rgb`: `(3, 256, 256)` texture-identity sidecar
- optional `density_rgb`: `(3, 256, 256)` derived on the fly from minimap RGB

Target:

- `height_coarse_65`: `(1, 65, 65)`, area-downsampled from authoritative `height_257`
- `weight_coarse_65`: `(1, 65, 65)`, area-downsampled from `weight_257`

Output signal:

- `height_coarse_65` only

#### H1: HeightResidualSample

Inputs:

- same H0 source channels
- `base_height_257`: `(1, 257, 257)`, deterministic upsample of frozen H0 output

For H1 model input assembly, the 256x256 source channels are resized to the
257x257 base-height grid before concatenation so the residual model operates on
the same grid as its output and reconstruction loss.

Target:

- `height_delta_257 = height_257 - base_height_257`
- `weight_257`: unchanged terrain-valid/object-aware loss weighting

Output signal:

- `height_delta_257` only

Initialization invariant:

- H1's final residual projection is zero-initialized. A fresh H1 checkpoint
  therefore starts as a no-op delta and the composed validation loss should
  begin near the frozen H0 baseline instead of destroying it with random deltas.

Composition:

```text
height_refined_257 = base_height_257 + height_delta_257
```

H1 losses may inspect the composed height for gradient or normal guidance, but
H1 does not predict normals, liquids, objects, or any second terrain head.

## 4. Minimap-Only Inference Objects

### 4.1 InferenceObjectHypothesis

Predicted from minimap without ADT placements.

Required fields:

- `tile_id` or runtime tile name
- `instance_id`
- `mask_bbox`
- `mask_confidence`
- `asset_candidate_paths`: ordered list
- `asset_candidate_scores`: ordered list
- `pred_xy`
- `pred_yaw`

Deferred fields:

- `pred_z`
- `pred_pitch`
- `pred_roll`

### 4.2 RecoveredObjectPlacement

Recovered placement after terrain exists.

Required fields:

- `asset_path`
- `x`
- `y`
- `z_from_terrain`
- `yaw`
- `confidence`

Optional later fields:

- `pitch`
- `roll`
- `scale`

## 5. Review Artifacts

Every proof stage should be able to emit a review row containing:

- raw minimap
- teacher or predicted object mask
- processed prior
- height preview
- object-library exemplar or candidate panel

This keeps the deconstruction pipeline human-auditable at each stage.
