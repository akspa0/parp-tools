# V7.6 Predicted Output Dataset Specification

Updated Apr 14, 2026.

This document defines the intended packaging contract for V7.6 inference outputs.

The important distinction is that this is a predicted dataset specification, not a harvested dataset specification.

- harvested datasets are built from real client or map data and documented in `docs/VLM_DATASET_EXPORTER.md`
- V7.6 predicted datasets are derivative outputs produced by a model from an input image

That means the output package must preserve provenance very explicitly so no one mistakes prediction for harvested ground truth.

## Purpose

The V7.6 predicted dataset should package arbitrary-image inference results in a structured, inspectable form.

It is intended to replace the current loose-file output behavior in:

- `src/WoWMapConverter/scripts/inference_v7_6.py`
- `src/WoWMapConverter/scripts/stitch_full_map.py`

The goals are:

- keep source-image provenance explicit
- keep model-checkpoint provenance explicit
- package predicted height and albedo outputs in a stable layout
- attach mesh exports and stitched outputs without forcing users to guess filenames

## Non-Goals

This package is not meant to claim harvested truth.

It must not:

- overwrite or masquerade as the original harvested input dataset
- imply that predicted height or albedo are client-authored assets
- hide the fact that these files are model outputs

## Root Layout

The intended root layout is:

```text
<predicted-dataset-root>/
|- predictions/
|  |- <sample-id>.json
|- sources/
|  |- <sample-id>_input.png
|- heights/
|  |- <sample-id>_height_pred.png
|- albedo/
|  |- <sample-id>_albedo_pred.png
|- meshes/
|  |- <sample-id>.obj
|  |- <sample-id>.mtl
|- stitched/
|  |- <map>_full_height_pred.png
|  |- <map>_full_albedo_pred.png
|- metadata.jsonl
|- dataset_info.json
|- v76_prediction_manifest.json
```

Not every run needs every directory.

- `stitched/` is optional and exists only for map-quilt jobs
- `meshes/` is optional if mesh export is disabled

## Packaging Rules

- every path recorded in JSON files must be root-relative and use `/`
- every prediction must preserve a stable `sample_id`
- predicted outputs must never be written back into the harvested input dataset root
- if the input came from an existing harvested tile, the output must preserve that linkage explicitly
- if the input is arbitrary user imagery, the output must say so explicitly

## Sample Identity

Each prediction record uses a `sample_id`.

Recommended rules:

- if the source was a harvested tile, use the tile name such as `Azeroth_32_48`
- if the source was not a harvested tile, derive a stable slug from the input filename
- for batch or quilt jobs, keep the map/tile coordinate identity if it exists in the input filename

## Run-Level Manifest Contract

The run-level manifest is `v76_prediction_manifest.json`.

Recommended top-level fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Version of this prediction-dataset spec |
| `generated_at_utc` | Timestamp for the prediction run |
| `prediction_root` | Absolute root path of the predicted dataset |
| `model_family` | Model family label, for example `v7.6` |
| `checkpoint_path` | Checkpoint used for inference |
| `source_kind` | `harvested_tile`, `arbitrary_image`, or `map_tile_batch` |
| `sample_count` | Number of predictions in this package |
| `samples` | Array of per-sample summary records |

### Example Manifest

```json
{
  "schema_version": "wowterrain-v76-prediction-manifest.v1",
  "generated_at_utc": "2026-04-14T20:30:00Z",
  "prediction_root": "i:/parp/parp-tools/output/v76_predictions/demo_run",
  "model_family": "v7.6",
  "checkpoint_path": "output_v7_6/checkpoints/latest.pth",
  "source_kind": "arbitrary_image",
  "sample_count": 1,
  "samples": [
    {
      "sample_id": "demo_tile",
      "prediction_json_path": "predictions/demo_tile.json",
      "source_input_path": "sources/demo_tile_input.png",
      "height_prediction_path": "heights/demo_tile_height_pred.png",
      "albedo_prediction_path": "albedo/demo_tile_albedo_pred.png",
      "obj_path": "meshes/demo_tile.obj",
      "mtl_path": "meshes/demo_tile.mtl"
    }
  ]
}
```

## Per-Sample Prediction Contract

Each prediction lives at `predictions/<sample-id>.json`.

Recommended structure:

```json
{
  "schema_version": "wowterrain-v76-prediction-tile.v1",
  "sample_id": "demo_tile",
  "source": {
    "source_kind": "arbitrary_image",
    "input_image_path": "sources/demo_tile_input.png",
    "original_width": 512,
    "original_height": 512,
    "model_input_width": 512,
    "model_input_height": 512,
    "resized_for_model": false
  },
  "model": {
    "model_family": "v7.6",
    "checkpoint_path": "output_v7_6/checkpoints/latest.pth",
    "input_channels": 3,
    "output_height_channels": 1,
    "output_albedo_channels": 3
  },
  "predictions": {
    "height_prediction_path": "heights/demo_tile_height_pred.png",
    "albedo_prediction_path": "albedo/demo_tile_albedo_pred.png",
    "obj_path": "meshes/demo_tile.obj",
    "mtl_path": "meshes/demo_tile.mtl"
  },
  "geometry_assumptions": {
    "height_encoding": "uint16-normalized",
    "max_height_assumed": 1200.0,
    "tile_size_assumed": 533.3333
  }
}
```

## Required Source Provenance Fields

The `source` block should make it impossible to confuse predicted outputs with harvested inputs.

### Required Fields

| Field | Meaning |
| --- | --- |
| `source_kind` | `harvested_tile`, `arbitrary_image`, or `map_tile_batch` |
| `input_image_path` | Stored source image copy under `sources/` |
| `original_width` / `original_height` | Original source size |
| `model_input_width` / `model_input_height` | Actual size fed to the network |
| `resized_for_model` | Whether resizing occurred |

### Recommended Additional Fields For Harvested-Tile Inputs

When the source is a harvested tile, also record:

- `source_dataset_root`
- `source_tile_json_path`
- `source_tile_name`
- `source_map_name`
- `source_client_label` when known

Those fields preserve linkage back to the real harvested supervision surface.

## Output Asset Families

### Source Copy

- `sources/<sample-id>_input.png`

The source image should always be copied into the prediction package even if it already existed elsewhere.

### Height Prediction

- `heights/<sample-id>_height_pred.png`

Recommended encoding:

- `I;16` PNG
- normalized model output mapped into the stored height encoding used by the inference path

### Albedo Prediction

- `albedo/<sample-id>_albedo_pred.png`

Recommended encoding:

- `RGB` PNG
- direct post-model albedo output in image space

### Mesh Export

- `meshes/<sample-id>.obj`
- `meshes/<sample-id>.mtl`

Optional companion texture reference:

- the MTL should point to `../albedo/<sample-id>_albedo_pred.png` or a copied mesh-local texture if the export path requires it

### Stitched Outputs

For quilt jobs the package may also contain:

- `stitched/<map>_full_height_pred.png`
- `stitched/<map>_full_albedo_pred.png`

These should be treated as optional aggregate outputs linked from the per-sample records or a batch-level section in the manifest.

## `metadata.jsonl` Contract

To stay parallel with harvested dataset packaging, the predicted dataset should also emit `metadata.jsonl`.

Each line should include a compact summary of one prediction.

Recommended fields:

- `sample_id`
- `source_kind`
- `input_image_path`
- `height_prediction_path`
- `albedo_prediction_path`
- `obj_path`
- `checkpoint_path`

## `dataset_info.json` Contract

The predicted dataset should also emit `dataset_info.json` so consumers can inspect the package without opening every prediction JSON.

Recommended top-level fields:

- `dataset_type`: `v76-predicted-output`
- `model_family`
- `sample_count`
- `has_meshes`
- `has_stitched_outputs`
- `source_kind_breakdown`

## Distinguishing Harvested From Predicted

This distinction must remain explicit everywhere.

### Harvested Input Dataset

- built from real client or map data
- documented in `docs/VLM_DATASET_EXPORTER.md`
- suitable as training truth when validated

### Predicted Output Dataset

- built from model inference over source images
- documented here
- useful for reconstruction, review, translation, and downstream tooling
- not a substitute for harvested truth

## Current Implementation Gap

The checked-in V7.6 scripts do not yet emit this full package.

Current behavior today:

- `inference_v7_6.py` writes loose predicted PNG and OBJ files under `inference_output/`
- `stitch_full_map.py` writes loose stitched quilts and per-tile OBJ bundles under `stitched_output_v7_restore/`

This document defines the intended structured replacement for those outputs.