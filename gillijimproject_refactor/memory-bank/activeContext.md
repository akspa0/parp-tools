# ACTIVE CONTEXT — wow-viewer

## Branch
- `v0.5.0-dev`

## Primary Live Lane
- V16 terrain dataset + training is the current execution path.
- Canonical short docs were rewritten and should now be the first read for this lane:
  - `wow-viewer/README.md`
  - `wow-viewer/data-harvester/README.md`
  - `wow-viewer/docs/architecture/v16-terrain-model-spec-2026-05-16.md`
- Canonical flow:
  - `WowViewer.Tool.Harvest harvest-stream --stream-profile v16`
  - `build_v16_dataset.py build`
  - `validate_v16_training_ready.py`
  - `train_v16.py`
- Current real-run trainer shape:
  - train pool: `--train-max-tiles 4000`
  - epoch budget: `--train-epoch-tiles 1350`
  - val budget: `--val-max-tiles 150`
  - batch size: `72`
  - GPU throttle: `--gpu-duty-cycle 100`
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
- Validation snapshot alpha QA now uses a painted-layer composite (`max(ch1..3)` with fallback) instead of raw `alpha[...,0]`, because channel `0` is commonly the implicit base layer and was producing false-black GT panels.
- `train-max-tiles` is now the persistent run-level train pool, while `train-epoch-tiles` can rotate a fresh per-epoch subset from that pool.
- CUDA-oriented loader defaults are less conservative now: `--num-workers=-1` auto-resolves a worker count and `persistent_workers` defaults on when workers are active.
- Trainer curation now has a basic quality gate by default: it drops obviously low-signal flat tiles and writes `train_quality_audit.json` / `val_quality_audit.json`.
- Every new best `val_h` epoch now writes a fresh random validation snapshot set under `validation/best_epoch_XXXX/`, separate from the normal interval snapshots.

## Harvest / Dataset Truth
- Stream format is lean `ARRY`, not legacy `NPZB`.
- Archive-backed ADT families now route through the in-memory byte path.
- Default dataset compression is Blosc `lz4` / level `1` / `shuffle`.
- `repair-index` is the fast fix for coordinate-only damage.
- `patch-liquids` can rewrite only liquid arrays + liquid provenance flags in-place.
- `inspect_v16_dataset.py` is the human-eye QA surface.
- Operator routing is intentionally simpler now:
  - root README = repo + workflow orientation
  - data-harvester README = commands + outputs
  - V16 spec = contract and boundaries

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
  - `wow-viewer/output/datasets/v16/validation/all-builds.training_readiness.json`
- Visual QA root:
  - `wow-viewer/output/datasets/v16/inspection/`
- Current per-build summaries:
  - `<build>.summary.json`
  - `<build>.samples.json`
  - `<build>.validation_audit_overview.png`
- Multi-build training smoke run:
  - `wow-viewer/models/v16/runs/smoke_v16_full_corpus_post_fix/`
  - 1 epoch on CPU completed cleanly against curated tiles from the finalized six-build corpus
- Alpha-validation snapshot fix proof:
  - `wow-viewer/models/v16/runs/smoke_alpha_validation_fix/validation/epoch_0001/tile_00/alpha_gt_painted_max.png`
  - `alpha_gt_painted_max.png` now carries nonzero GT intensity; the prior false-black symptom was a channel-selection issue, not a corpus-alpha loss issue
- Epoch-rotation proof:
  - `wow-viewer/models/v16/runs/smoke_epoch_rotation/evidence/train_epoch_orders.jsonl`
  - epoch `1` selected positions `[7,4,2,0]`; epoch `2` selected `[5,6,4,2]`, proving fresh epoch subsets from a larger curated pool
- Current production-oriented launch contract:
  - run name: `v16_full_corpus_epoch_rotation`
  - command uses `train-max-tiles 4000`, `train-epoch-tiles 1350`, `val-max-tiles 150`, `batch-size 72`, `gpu-duty-cycle 100`
- Alpha/minimap alignment audit:
  - `wow-viewer/output/datasets/v16/validation/alpha_minimap_alignment/alpha_minimap_alignment.summary.json`
  - sampled corpus result: `edge_f1_mean≈0.54`, `median≈0.64`, but `p10=0.0`, confirming a real zero-match bad tail
- Quality-curation proof:
  - `wow-viewer/models/v16/runs/smoke_quality_curation/evidence/train_quality_audit.json`
  - basic gate dropped `196` obviously low-signal flat train tiles from the `3_3_5_12340` smoke candidate pool (`4621 -> 4425`)

## Next Likely Slice
- Monitor `v16_full_corpus_epoch_rotation`, inspect `train_quality_audit.json`, and tighten mismatch rejection if the alpha/minimap bad tail still leaks into validation.
- If WL* chunk-fill behavior matters to loss semantics, handle it in the loader/trainer, not by reopening harvest.
- PM4 follow-up now has a library-owned `MSHD.Field04` region-id seam feeding `MdxViewer` overlay coloring/debug/export; broader PM4 object-mapping work can build on that without reintroducing viewer-owned decode logic.
