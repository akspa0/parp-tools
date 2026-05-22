# V16 Terrain Model Specification

## Purpose

V16 is the current terrain-AI dataset and training contract for `wow-viewer`.

It defines:

- the finalized per-build Zarr dataset layout
- the build pipeline from `harvest-stream` into Zarr
- the current trainer contract
- the current inference/output contract
- what is implemented now versus intentionally deferred

This document is the technical source of truth. Operator-facing steps live in
`data-harvester/README.md`.

## Why V16 Exists

V15 and earlier terrain lanes depended on large numbers of per-tile NPZ files.
That created four practical problems:

- too many tiny files
- inconsistent feature presence across builds
- heavy redundant per-tile metadata
- awkward patch/repair behavior once a corpus had already been built

V16 replaces that with one consolidated Zarr store per build and a harvest
stream that writes directly into the final dataset surface.

## Dataset Contract

### Store Layout

```text
wow-viewer/output/datasets/v16/
  <build>.zarr/
    zarr.json
    index.parquet
    placements.parquet
    signal_validation.json
    _resume_state.json
    height_257/
    normal_xyz/
    normal_mask/
    alpha_256/
    holes_16/
    liquid_mask/
    liquid_height/
    object_mask/
    object_precise_mask/
    object_instance_mask/
    mcnk_flags_16/
    mddf_mask/
    modf_mask/
    object_filtered_mask/
    minimap_rgb/
    shadow_mask/
    mcly_texture_ids/
    mcly_layer_mask/
```

### Key Design Rules

1. One Zarr store per build.
2. Flat arrays indexed by `tile_id`.
3. Missing signals stay zero-filled and are tracked with `has_*` flags in `index.parquet`.
4. No intermediate NPZ files are written during the V16 build path.
5. Light chunk compression stays on by default for new stores.

### Arrays

| Array | Shape per tile | dtype | Meaning |
|---|---|---|---|
| `height_257` | `257x257` | `float32` | world-space terrain height |
| `normal_xyz` | `257x257x3` | `float32` | terrain normals |
| `normal_mask` | `257x257` | `bool` | valid-normal coverage |
| `alpha_256` | `256x256x4` | `float32` | MCAL/MCLY blend weights |
| `holes_16` | `16x16` | `bool` | terrain hole mask |
| `liquid_mask` | `256x256` | `float32` | unified liquid presence |
| `liquid_height` | `256x256` | `float32` | unified liquid height |
| `object_mask` | `257x257` | `bool` | merged object footprint |
| `object_precise_mask` | `257x257` | `float32` | soft object footprint |
| `object_instance_mask` | `257x257` | `int32` | instance-id raster |
| `mcnk_flags_16` | `16x16` | `int32` | MCNK liquid/classification flags |
| `mddf_mask` | `257x257` | `float32` | doodad-only mask |
| `modf_mask` | `257x257` | `float32` | WMO-only footprint mask; archive harvest prefers geometry-derived WMO footprints when asset bytes are available |
| `object_filtered_mask` | `257x257` | `float32` | terrain-loss gating mask; includes WMO footprints and filtered MDDF footprints |
| `minimap_rgb` | `256x256x3` | `uint8` | baked minimap RGB |
| `shadow_mask` | `256x256` | `float32` | archived shadow signal |
| `mcly_texture_ids` | `16x16x4` | `int32` | texture-layer ids |
| `mcly_layer_mask` | `16x16x4` | `float32` | layer visibility mask |

### Derived Rules

- `normal_mask` is derived from nonzero normal vectors.
- `liquid_mask` and `liquid_height` are unified Python-side outputs built from richer raw liquid signals.
- terrain loss weighting uses `1.0 - object_filtered_mask`, not raw `object_mask`.
- archive-backed ADT harvest now prefers geometry-derived WMO footprints for `modf_mask` / `object_filtered_mask`; bounds fallback remains for unresolved WMO assets.
- MdxViewer dataset finalize now writes renderer-truth `object_visibility_mask` and `no_object_minimap` artifacts from the validation capture families; `0.x` builds prefer direct `objectsonly` silhouettes so early underground object bleed-through is preserved, while later builds prefer `primary` vs `noobjects` visibility diffs so terrain occlusion wins.
- Those renderer-truth artifacts are currently `V16.2` candidate signals, not part of the finalized base V16 Zarr contract yet.
- Current real capture proof for that renderer-truth lane is bounded to `0_5_3_3368` and `3_3_5_12340` on the `Azeroth_30_48` tile family until the remaining builds are explicitly validated.
- alpha QA should use the painted-layer view, not raw channel `0`.

## V16.2 Transition Note

The next dataset-contract lane for richer object guidance is `V16.2`, tracked
under `wow-viewer/specs/011-v16-2-patched-signal-expansion/`.

Current transition rule:

- keep finalized V16 base stores intact
- stage renderer-truth and richer precise-mask signals into sidecar stores first
- consume those sidecars through loader metadata overlay semantics
- only consider merge-back into canonical base stores after broader cross-build
  renderer-truth validation exists

## Index Contract

### Core Columns

| Column | dtype | Meaning |
|---|---|---|
| `tile_id` | `int64` | row into every fixed-shape array |
| `build` | `string` | build key |
| `map` | `string` | map name |
| `tile_x` | `int32` | tile X |
| `tile_y` | `int32` | tile Y |
| `height_mean` | `float32` | per-tile mean height |
| `height_std` | `float32` | per-tile height std |
| `n_mddf` | `int32` | doodad placement count |
| `n_modf` | `int32` | WMO placement count |

### Presence / Provenance Flags

The index also carries:

- `has_<array>` for each fixed-shape signal
- `has_liquid_source_mcnk`
- `has_liquid_source_mh2o`
- `has_liquid_source_mclq`
- `has_liquid_source_unified`
- `has_liquid_source_wl`

## Build Contract

### Build Pipeline

```text
staged client -> harvest-stream -> Python decoder -> Zarr arrays + parquet index
```

### Harvester / Builder Boundary

- `WowViewer.Tool.Harvest harvest-stream` emits length-prefixed `ARRY` tile blobs over stdout.
- diagnostics go to stderr only
- the Python builder consumes the stream directly and writes the Zarr store
- `ENDS` terminates the stream

### Required Inputs

- staged client root under `output/tmp/wowarchive-clients/<build>/World of Warcraft/`
- built C# harvester
- `uv` environment under `data-harvester/`

### Important Build Behaviors

- stores are written to `<build>.zarr.partial/` first and promoted only on success
- `_resume_state.json` tracks resumable progress
- `repair-index` fixes coordinate/index damage without rebuilding arrays
- `patch-liquids` rewrites only liquid arrays and liquid flags
- `patch-objects` rewrites only object-mask arrays and object flags
- `validate-signals` checks corpus signal coverage

### Compression

Default new-store compression:

- codec: `lz4`
- level: `1`
- shuffle: `shuffle`

The goal is light chunk compression without the old NPZ/zip overhead.

## Training Contract

### Current Implemented Path

| Surface | File | Status |
|---|---|---|
| dataset loader | `src/harvester/v16_dataset.py` | implemented |
| current terrain model | `src/harvester/v15_model.py` | implemented |
| trainer | `scripts/train_v16.py` | implemented |
| trainer-readiness validator | `scripts/validate_v16_training_ready.py` | implemented |
| inference bridge | `scripts/infer_v16.py` | implemented |

### Current Supervised Targets

| Target | Source tensor | Current loss |
|---|---|---|
| height | `height` | weighted L1 |
| normals | `normals`, `normal_mask` | cosine |
| alpha | `alpha` | weighted L1 |
| holes | `holes` | weighted L1 |
| liquid mask | `liquid` | weighted L1 |
| MCLY | `mcly_ids`, `mcly_mask` | masked cross-entropy |

### Present But Not Yet Supervised

| Signal | Current status |
|---|---|
| `liquid_height` | intentionally deferred |
| `object_instance_mask` | readable, not in loss |
| `object_precise_mask` | stored, not loaded by default trainer |
| `mcnk_flags_16` | provenance / QA signal only |
| `mddf_mask` | QA / future experiments |
| `modf_mask` | QA / future experiments |
| `shadow_mask` | archived auxiliary signal |

### Normalization

- height: per-tile z-score using `height_mean` and `height_std`
- normals: raw unit-vector target in `[-1, 1]`
- alpha: raw `[0, 1]`
- liquid mask: raw `[0, 1]`
- minimap RGB: `uint8 -> [0, 1]` at load time

### Current Trainer Behaviors

- `train-max-tiles` defines the persistent curated train pool
- `train-epoch-tiles` rotates a fresh per-epoch subset from that pool
- per-epoch train subsets are build-balanced when possible
- `curation-quality-profile basic` is the current default and drops obviously low-signal junk tiles
- `train_quality_audit.json` and `val_quality_audit.json` record quality-gate evidence
- regular qualitative snapshots write to `validation/epoch_XXXX/`
- every new best `val_h` writes a fresh random review set to `validation/best_epoch_XXXX/`

### Validation Snapshot Rules

- alpha snapshots use `alpha_gt_painted_max.png` / `alpha_pred_painted_max.png`
- best-epoch snapshots must not reuse the exact same tile selection as interval snapshots
- the fixed validation split remains stable for metrics; only the qualitative sample changes

## Quality / Audit Surfaces

### Required Validation Before Training

1. `validate-signals`
2. `inspect_v16_dataset.py --write-overview`
3. `validate_v16_training_ready.py`

### Alpha / Minimap Alignment Audit

`audit_v16_alpha_minimap_alignment.py` exists because some harvested tiles show
alpha GT that does not visually correspond to baked minimap structure.

Current audit truth:

- the issue is real, not purely subjective
- sampled corpus result showed a healthy center but a bad tail
- `edge_f1_p10 = 0.0` means some sampled alpha-bearing tiles had effectively zero minimap/alpha edge agreement

That audit should be used when validation panels suggest mismatched supervision.

## Inference Contract

### Input Store

- path: `wow-viewer/output/datasets/v16/<build>.zarr/`
- authority: `index.parquet`, `placements.parquet`, fixed-shape arrays

### Output Store

- target path: `wow-viewer/output/datasets/v16_inference/<run>/<build>.pred.zarr/`
- row order must match the input store exactly
- `tile_id` values must remain identical
- output stores are final artifacts, not append-in-place scratchpads

### Required Prediction Arrays

| Array | Shape per tile | dtype |
|---|---|---|
| `height_pred_257` | `257x257` | `float32` |
| `normal_pred_xyz` | `257x257x3` | `float32` |
| `alpha_pred_256` | `256x256x4` | `float32` |
| `holes_pred_16` | `16x16` | `float32` |
| `liquid_pred_mask_256` | `256x256` | `float32` |
| `mcly_pred_logits_16x16x4x16` | `16x16x4x16` | `float32` |

### Current Practical Outputs

`infer_v16.py` currently emits:

- `<build>.pred.zarr`
- per-tile `inference_summary.json`
- `predicted_height_257.npy`
- `predicted_liquid_mask_256.npy`

## Reconstruction Contract

The downstream patch/export direction remains:

- `terrain-patch-adt`
- `convert-lk-to-alpha`
- `convert-alpha-to-lk`

Two modes matter:

1. patch existing terrain using prediction outputs
2. synthesize new terrain-domain outputs from model predictions

Those workflows should consume paired input/output dataset stores rather than
ad hoc loose arrays whenever possible.

## Explicit Naming Note

- the current V16 terrain model implementation still lives in `src/harvester/v15_model.py`
- there is no separate live `v16_model.py` implementation today
- that filename mismatch is cosmetic until the implementation itself changes

## Current Corpus Status

Finalized stores currently exist for:

- `0_5_3_3368`
- `0_5_5_3494`
- `0_7_0_3694`
- `3_0_1_8303`
- `3_3_5_12340`
- `4_0_0_11927`

Current status:

- all six current `signal_validation.json` files pass
- all six current stores have visual QA artifacts
- `0_7_0_3694` still carries the expected allowed warning for zero `has_holes_16`

## Deliberately Deferred

These are intentionally not owned by the current terrain trainer:

- liquid-height fidelity as a first-class model target
- object segmentation as a separate model
- asset attribution / placement recovery
- PM4 cross-reference / CK24 mapping

Those belong in later multi-model work, not in the current single terrain-model contract.
