# Data Model: 089 — DA-V2-Small LoRA Height Predictor with Cross-Tile Consistency

**Phase 1 output. Companion to `plan.md` and `research.md`.**
**Date**: 2026-07-03.

This file defines the concrete entities V23 implementation work will create and consume. The contracts in `contracts/` are written against these entities.

---

## 1. Top-Level Flow

```text
V22 Zarr Store
  -> TilesetPruneTable
  -> V23HeightDataset
  -> V23DatasetSample
  -> V23HeightPredictor
  -> V23ModelOutput
  -> V23LossBreakdown / V23Checkpoint
  -> CAI stitched inference artifacts
  -> RunPod bundle + evidence files
```

The V23 model is a consumer of V22. It does not alter the V22 store contract.

---

## 2. Input Contract Entities

### 2.1 `InputMode`

Controls the active channel subset:

```text
InputMode
├── full
├── minimap_only
├── minimap_alpha
└── minimap_alpha_normal
```

Rules:

- `full` is the only default.
- All other modes are explicit diagnostics/degraded configurations.
- The replaced patch-embed conv is sized from the active mode's channel count.

### 2.2 `ChannelDescriptor`

One row in the documented channel contract.

| Field | Type | Meaning |
|---|---|---|
| `index_start` | int | first channel index in the packed tensor |
| `index_end` | int | last channel index in the packed tensor |
| `name` | str | logical channel name |
| `source_array` | str | V22 source array name |
| `channels` | int | number of channels contributed |
| `dtype` | str | pre-normalization dtype |
| `normalization` | str | normalization rule |
| `fill_policy` | str | zero-fill / OOV / pass-through rule |
| `required_in_default` | bool | whether `InputMode.full` requires it |

### 2.3 `TilesetPruneTable`

Stable mapping from V22 build-wide tileset ids to V23 one-hot indices.

| Field | Type | Meaning |
|---|---|---|
| `build_keys` | list[str] | builds included when the table was derived |
| `top_k` | int | requested retained-cardinality, default 256 |
| `tileset_id_to_index` | dict[int, int] | original V22 tileset id -> pruned index |
| `oov_index` | int | out-of-vocabulary bucket |
| `tileset_path_hashes` | dict[int, str] | optional audit hash per source path |

Validation rules:

- indices `0..top_k-1` are reserved for retained tilesets
- `oov_index == top_k`
- the same table is used for every build in one V23 training run

---

## 3. Dataset Entities

### 3.1 `V23DatasetSample`

Single tile record emitted by `V23HeightDataset`.

| Field | Shape / Type | Meaning |
|---|---|---|
| `tile_id` | int | row id in the V22 store |
| `build` | str | build key |
| `map_name` | str | map / continent name |
| `tile_x` / `tile_y` | int | tile coordinates |
| `input` | `float32[C,256,256]` | packed channel tensor |
| `target_height` | `float32[1,257,257]` | liquid-aware target height |
| `terrain_valid_mask` | `bool[1,257,257]` | loss mask on metric target |
| `channel_valid_mask` | `bool[C]` | whether each logical channel was source-backed |
| `metadata` | dict | optional audit fields, hashes, source-availability flags |

Rules:

- `target_height` uses `liquid_height` where `liquid_mask > 0`
- `terrain_valid_mask` excludes masked/object/liquid-invalid pixels
- absent optional source arrays zero-fill the corresponding input channels and set `channel_valid_mask=false`

### 3.2 `V23Batch`

Mini-batch consumed by the trainer.

| Field | Shape / Type | Meaning |
|---|---|---|
| `inputs` | `float32[B,C,256,256]` | packed model input |
| `target_height` | `float32[B,1,257,257]` | metric target |
| `terrain_valid_mask` | `bool[B,1,257,257]` | masked supervision |
| `samples` | list[`V23DatasetSample`] | original sample metadata |

When GPCT is enabled, one `V23Batch` expands into a grouped patch batch rather than changing the dataset contract.

### 3.3 `GpctPatchBatch`

Training-time grouped subtile batch derived from one `V23Batch`.

| Field | Type | Meaning |
|---|---|---|
| `subtile_inputs` | `float32[B*K,C,H,W]` | stacked overlapping subtiles |
| `subtile_targets` | `float32[B*K,1,257,257]` | optional supervision view |
| `overlap_pairs` | list[tuple] | overlap-coordinate descriptors |
| `parent_index` | `int[B*K]` | maps each subtile back to its tile |
| `crop_window` | `tuple[int,int,int,int][B*K]` | source crop window |

---

## 4. Model Entities

### 4.1 `DepthAnythingV2SmallEncoderConfig`

| Field | Type | Meaning |
|---|---|---|
| `model_id` | str | HF model id |
| `in_channels` | int | active V23 input channels |
| `lora_rank` | int | LoRA rank, default 16 |
| `lora_alpha` | int | LoRA alpha, default 32 |
| `lora_dropout` | float | adapter dropout |
| `target_modules` | list[str] | attention projection module names |
| `gradient_checkpointing` | bool | enable checkpointing in train mode |

### 4.2 `EncoderFeatureSchema`

Describes the feature pyramid emitted by the encoder.

| Field | Type | Meaning |
|---|---|---|
| `stage_names` | list[str] | ordered feature stages |
| `channels` | dict[str, int] | per-stage channel count |
| `stride` | dict[str, int] | per-stage downsample factor |
| `tensor_shape_template` | dict[str, tuple] | batch-agnostic shape template |

### 4.3 `V23HeadConfig`

| Field | Type | Meaning |
|---|---|---|
| `reassembly_width` | int | decoder internal width |
| `output_size` | tuple[int, int] | fixed `(257,257)` |
| `anchor_hidden_dim` | int | affine-anchor MLP width |
| `disparity_activation` | str | output activation for disparity |

### 4.4 `V23ModelOutput`

Combined forward result.

| Field | Shape / Type | Meaning |
|---|---|---|
| `disparity` | `float32[B,1,257,257]` | affine-invariant output |
| `affine_scale` | `float32[B,1]` | per-tile metric scale |
| `affine_shift` | `float32[B,1]` | per-tile metric shift |
| `metric_height` | `float32[B,1,257,257]` | `disparity * scale + shift` |
| `feature_pyramid` | dict[str, Tensor] | optional training/inference features |

Rules:

- `metric_height` is derived, not a second prediction head
- the model predicts only height/disparity; no normals/liquids/objects

---

## 5. Loss and Training Entities

### 5.1 `V23LossWeights`

| Field | Type | Default |
|---|---|---:|
| `lssi` | float | `1.0` |
| `lgm` | float | `0.5` |
| `sdc` | float | `0.1` |
| `gpct` | float | `0.1` |
| `bias_free_mask_ratio` | float | `0.15` |

### 5.2 `V23LossBreakdown`

Logged per step / validation batch.

| Field | Type | Meaning |
|---|---|---|
| `total` | float | weighted total |
| `lssi` | float | affine-invariant term |
| `lgm` | float | gradient term |
| `sdc` | float | spatial-distance term |
| `gpct` | float | overlap-consistency term |
| `masked_patch_ratio` | float | realized masking ratio |

### 5.3 `V23TrainingRunConfig`

| Field | Type | Meaning |
|---|---|---|
| `dataset_dir` | str | V22 dataset root |
| `builds` | list[str] | selected builds |
| `input_mode` | `InputMode` | active channel mode |
| `batch_size` | int | tiles per optimizer step |
| `gpct_k` | int | grouped subtiles per tile |
| `deterministic` | bool | strict deterministic mode |
| `seed` | int | training seed |
| `device` | str | training device |
| `optimizer_name` | str | expected `PagedAdamW8bit` |
| `target_vram_gb` | float | envelope target |

---

## 6. Inference Entities

### 6.1 `CaiRequest`

| Field | Type | Meaning |
|---|---|---|
| `tile_grid` | list[tuple[int,int]] | requested tile coordinates |
| `cai_r` | int | number of overlap shifts |
| `deterministic` | bool | inference determinism flag |
| `seed` | int | recorded seed |
| `fp16` | bool | inference precision mode |

### 6.2 `CaiAccumulator`

| Field | Type | Meaning |
|---|---|---|
| `sum` | Tensor | accumulated predictions |
| `count` | Tensor | per-pixel coverage count |
| `grid_shape` | tuple[int,int] | tile-grid dimensions |
| `output_shape` | tuple[int,int] | stitched output size |

### 6.3 `V23InferenceArtifact`

| Field | Type | Meaning |
|---|---|---|
| `predicted_disparity` | array | saved disparity output |
| `predicted_metric_height` | array | saved metric height output |
| `preview_png_path` | str | preview path |
| `edge_l1_metrics` | dict | optional seam metrics |
| `request` | `CaiRequest` | recorded invocation |

---

## 7. Checkpoint and Bundle Entities

### 7.1 `V23Checkpoint`

| Field | Type | Meaning |
|---|---|---|
| `config` | dict | full run config |
| `model_state` | dict[str, Tensor] | model weights |
| `optimizer_state` | dict | optimizer state |
| `epoch` | int | completed epoch |
| `commit_sha` | str | repo revision |
| `environment` | dict | torch/cuda/image/version metadata |
| `data_hashes` | dict | V22 store + prune-table hashes |

### 7.2 `RunPodBundleManifest`

| Field | Type | Meaning |
|---|---|---|
| `bundle_name` | str | archive identifier |
| `contains_game_client_files` | bool | must be `false` |
| `source_spec` | str | `089-dav2-height-predictor` |
| `dataset_subset` | dict | included build/tile counts |
| `paths` | list[str] | packaged file list |
| `tree_hash` | str | bundle content hash |

---

## 8. State Transitions

### 8.1 Dataset Sample Lifecycle

```text
V22 tile row
  -> source arrays resolved
  -> channel tensor packed
  -> liquid-aware target built
  -> valid masks attached
  -> V23DatasetSample emitted
```

### 8.2 Training Lifecycle

```text
Run config
  -> dataset + prune table loaded
  -> V23 batch emitted
  -> optional GPCT subtiles generated
  -> model forward
  -> loss breakdown computed
  -> checkpoint persisted
  -> validation previews written
```

### 8.3 Inference Lifecycle

```text
Checkpoint + CaiRequest
  -> per-tile or shifted subtile loads
  -> deterministic forward
  -> optional CAI accumulation
  -> disparity / metric outputs written
  -> preview + seam metrics recorded
```

---

## 9. Relationship to Existing Surfaces

| Existing surface | Role in V23 |
|---|---|
| `harvester.v22_zarr_io` | canonical V22 read surface |
| `docs/architecture/v22-dataset-signals-2026-06-30.md` | schema authority for source arrays |
| Spec 079 bundle contract | owner of Pod/bootstrap pattern |
| `data-harvester/src/harvester/v23/` | sole new V23 implementation owner |

*End of data model. Next: `contracts/` and `quickstart.md` are the operator-facing Phase 1 outputs.*
