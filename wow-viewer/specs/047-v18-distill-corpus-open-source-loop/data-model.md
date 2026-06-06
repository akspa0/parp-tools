# Data Model: V18 Focused Two-Build Terrain Reconstruction System

## FocusedV18Corpus

Represents the two canonical V18 training stores.

### Fields

- `build`: string
  - allowed values:
    - `0_5_3_3368`
    - `3_3_5_12340`
- `store_path`: path
- `index_path`: path
- `tile_count`: integer
- `has_minimap_rgb`: boolean
- `has_height_257`: boolean
- `has_normal_xyz`: boolean
- `has_normal_mask`: boolean
- `has_liquid_mask`: boolean

### Validation

- store must exist under `wow-viewer/output/datasets/v18/`
- `minimap_rgb`, `height_257`, and `normal_xyz` must be present for active
  terrain training
- `liquid_mask` is optional for a given row but required as an available corpus
  signal

## FocusedCurationManifest

Represents the filtered, scored row set used by focused V18 training.

### Fields

- `profile`: string
- `dataset_dir`: path
- `builds`: string[]
- `tiles_path`: path
- `kept_tiles_path`: path
- `summary_path`: path
- `source_manifest`: path | null
- `selection_recipe`: object | null

### Relationships

- one `FocusedV18Corpus` produces zero or more `FocusedCurationManifest` runs
- one `FocusedCurationManifest` may produce zero or more tiny derived manifests
  for scouting

## FocusedCurationRow

Represents one tile row in the focused curation manifest.

### Fields

- `build`: string
- `map`: string
- `tile_id`: integer
- `tile_x`: integer
- `tile_y`: integer
- `keep`: boolean
- `reject_reason`: string | null
- `quality_score`: float
- `usefulness_score`: float
- `difficulty_score`: float
- `difficulty_bucket`: string | null
- `height_std`: float
- `normal_cov`: float
- `terrain_valid_cov`: float
- `trainable_cov`: float
- `liquid_cov`: float
- `normal_edge_f1`: float
- `what_plate`: boolean

### Validation

- `keep == false` requires a non-null `reject_reason`
- `difficulty_bucket` must be one of:
  - `easy`
  - `medium`
  - `hard`
  - `pathological`
  - `null` only for rejected rows

## HeightModelRun

Represents one focused V18 height training run.

### Fields

- `run_name`: string
- `dataset_dir`: path
- `builds`: string[]
- `curation_manifest`: path | null
- `input_contract`: literal `minimap_rgb`
- `output_contract`: literal `height_257`
- `checkpoint_dir`: path
- `config_path`: path
- `training_log_path`: path
- `epoch_sampling_mode`: string
- `epoch_sampling_fraction`: number | null
- `best_val`: float | null
- `best_epoch`: integer | null

### Validation

- all focused runs must stay under `wow-viewer/models/v18/height/runs/`
- `epoch_sampling_fraction` must be in `(0, 1]` when `epoch_sampling_mode` is
  `bucket_rotation_fraction`

## NormalModelRun

Represents one focused V18 normal training run.

### Fields

- `run_name`: string
- `dataset_dir`: path
- `builds`: string[]
- `curation_manifest`: path | null
- `input_contract`: literal `minimap_rgb`
- `output_contract`: literal `normal_xyz`
- `checkpoint_dir`: path
- `config_path`: path
- `training_log_path`: path
- `epoch_sampling_mode`: string
- `epoch_sampling_fraction`: number | null
- `best_val`: float | null
- `best_epoch`: integer | null

### Validation

- all focused runs must stay under `wow-viewer/models/v18/normal/runs/`
- `epoch_sampling_fraction` must be in `(0, 1]` when `epoch_sampling_mode` is
  `bucket_rotation_fraction`

## TerrainQuiltJob

Represents the later-stage stitched terrain reconstruction job.

### Fields

- `run_name`: string
- `source_tiles`: list of `{build, map, tile_x, tile_y, tile_id}`
- `height_prediction_root`: path
- `normal_prediction_root`: path
- `quilt_layout`: object
- `border_constraints`: object
- `output_root`: path

### Relationships

- consumes one `HeightModelRun`
- consumes one `NormalModelRun`
- consumes many `FocusedCurationRow` / tile placements

### Validation

- every predicted tile must be placeable back into quilt coordinates
- quilt job must carry enough metadata for later ADT emission

## FocusedInferenceProofRun

Represents a focused minimap-only inference proof over the V18 stores.

### Fields

- `run_name`: string
- `dataset_dir`: path
- `build`: string
- `height_checkpoint`: path | null
- `normal_checkpoint`: path | null
- `resolved_input_contract`: literal `minimap_rgb`
- `prediction_store_path`: path

### Validation

- the forward pass must consume minimap RGB only
- hidden supervision-only tensors may be used later for offline scoring, but
  they are not part of the runtime input contract
