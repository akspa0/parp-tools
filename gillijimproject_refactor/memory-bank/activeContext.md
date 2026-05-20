# ACTIVE CONTEXT — wow-viewer

## Branch
- `v0.5.0-dev`

## Primary Live Lane
- V16 terrain dataset + training is the current execution path.
- Canonical flow:
  - `WowViewer.Tool.Harvest harvest-stream --stream-profile v16`
  - `build_v16_dataset.py build`
  - `validate_v16_training_ready.py`
  - `train_v16.py`
- `wow-viewer` is the implementation owner. `gillijimproject_refactor` is reference/continuity/validation only.

## Current V16 Corpus Truth
- Finalized stores exist for:
  - `0_5_3_3368`
  - `0_5_5_3494`
  - `0_7_0_3694`
  - `3_0_1_8303`
  - `3_3_5_12340`
  - `4_0_0_11927`
- All six current `signal_validation.json` files pass.
- Human-eye QA artifacts exist for all six under:
  - `wow-viewer/output/datasets/v16/inspection/`
- Only standing allowed warning in the current corpus:
  - `0_7_0_3694` has `has_holes_16 = 0`

## Current Trainer Contract
- Dataset loader: `wow-viewer/data-harvester/src/harvester/v16_dataset.py`
- Current terrain model host: `wow-viewer/data-harvester/src/harvester/v15_model.py`
- Current supervised terrain heads:
  - height
  - normals
  - alpha
  - holes
  - liquid mask
  - MCLY logits
- `liquid_height` stays in the dataset contract but is deferred from the current terrain trainer/inference path.
- Terrain loss weighting uses `object_filtered_mask`.
- `object_instance_mask` is readable but not yet used by the terrain trainer.

## Harvest / Dataset Truth
- Stream format is lean `ARRY`, not legacy `NPZB`.
- Archive-backed ADT families now route through the in-memory byte path.
- Default dataset compression is Blosc `lz4` / level `1` / `shuffle`.
- `repair-index` is the fast fix for coordinate-only damage.
- `patch-liquids` can rewrite only liquid arrays + liquid provenance flags in-place.
- `inspect_v16_dataset.py` is the human-eye QA surface.

## Critical Recent Fixes
- Mixed Cataclysm archive tiles can carry inline root `MCLY` / `MCAL` without `_tex0`.
  - `AdtTensorPackBuilder.ReadTextureDataFromBytes(...)` now falls back to inline root texture parsing when `_tex0` bytes are absent.
  - Focused proof on staged `4_0_0_11927 / AhnQiraj / (27,46)` restored `mcly_texture_ids`, `mcly_layer_mask`, and `mcal_alpha_pack_256`.
- Alpha placeholder `map=memory` metadata was fixed at the harvest / repair-index seam.
- Liquid derivation now prefers explicit `mh2o_presence_mask` / `mclq_presence_mask`; WL* remains last-resort fallback.

## Known Nuance
- WL* liquid coverage still does not always fill the whole chunk footprint that the raw data spans.
- This is currently treated as a downstream loader / trainer semantic issue, not a harvest-corruption issue.
- The corpus is now considered consistent enough for training work.

## Inference Direction
- Keep the paired contract:
  - input: `wow-viewer/output/datasets/v16/<build>.zarr`
  - output: `wow-viewer/output/datasets/v16_inference/<run>/<build>.pred.zarr`
- Current `infer_v16.py` emits:
  - `<build>.pred.zarr`
  - per-tile `inference_summary.json`
  - `predicted_height_257.npy`
  - `predicted_liquid_mask_256.npy`
- Downstream patch/export path remains:
  - `terrain-patch-adt`
  - `convert-lk-to-alpha`
  - `convert-alpha-to-lk`

## Focused Proof Pointers
- Trainer-readiness proof:
  - `wow-viewer/output/datasets/v16/validation/3_3_5_12340.training_readiness.json`
- Visual QA root:
  - `wow-viewer/output/datasets/v16/inspection/`
- Current per-build summaries:
  - `<build>.summary.json`
  - `<build>.samples.json`
  - `<build>.validation_audit_overview.png`

## Next Likely Slice
- Start the first real V16 training run.
- If WL* chunk-fill behavior matters to loss semantics, handle it in the loader/trainer, not by reopening harvest.
