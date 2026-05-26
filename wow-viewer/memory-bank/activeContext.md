# Active Context — V16.x Terrain Training Pipeline

## Branch
- `v0.5.0-dev`

## Current Focus
Terrain normal/height prediction from minimap images. Chain of V16.1.x model iterations:
- **V16.1.1**: Normal model (minimap 3ch → normals)
- **V16.1.2**: Abandoned — refiner approach (random refiner used as distillation target, never trained)
- **V16.1.3**: Height-channel normal model (cat(minimap, height) 4ch → normals) — plateaued at epoch 123
- **V16.1.4**: Combined normal+height model (4ch → normals + height, shared backbone) — just implemented, not yet trained
- **V18 object-roof lane**: new Spec Kit draft for object-family roof curation plus learned minimap object sieve (`specs/025-object-roof-mask-library-and-minimap-sieve/`)
  - includes MdxViewer improvements for one-at-a-time object asset capture with pose metadata
  - stores per-asset object visual outputs in a separate Zarr datastore for roof exemplars and object-family recognition
  - separate object-identification model is intended to live in the Python `uv` stack and use the Hugging Face transformers library as the first host
  - SAM2 is the first promptable mask-generation host; SAM3 is allowed later if the Hugging Face token has approved gated access

## What Exists (Completed)
- All model classes in `wow-viewer/data-harvester/src/harvester/v16_1_models.py`:
  `V161NormalModel`, `V161HeightModel`, `V161NormalHeightModel`, `V161NormalHeightCombinedModel`,
  `V161NormalRefiner`, `V161HolesModel`, `V161LiquidModel`, `V161TexcompModel`
- Training loop in `train_v16_1_common.py` with autotune, curation, hard-region weighting
- `_combined_loss` + `combined` task registered in TASKS (V16.1.4)
- Working export script: `export_terrain_obj.py`
- V16 dataset: 5134 tiles across 6 builds in Zarr stores
- Spec 017 (V16.1.4): spec.md, plan.md, tasks.md (Phase 1 done)

## What's Next (Immediate)
- **Spec 020**: Fix renderer culling (renderer doesn't see objects) → tile-level capture → batch capture → V16.2 object mask generation
- V16.1.4 combined model training (waiting on smoke test/launch)
- V16.2 model architecture (V16.1.x lessons + proper object-aware loss weighting)

## MotherShip Direction
The long-range target is a universal game engine (`game-engine`) with a plugin architecture. WoW data support lives in `GameEngine.Plugin.WoW`. The current `wow-viewer` work feeds into this. The repo structure is placeholder but the direction is real: `theMothership/` with `game-engine/` core + plugins + viewer + tools. See `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` for the engine-modernization context.

## Known Issues
- Renderer culls all objects — coordinate system bug in `ComputeTilePlanarMin/Max` (spec 013 diagnosed, not fixed)
- MdxViewer loads whole map for single-tile capture (performance bottleneck for V16.2 dataset gen)
- Object mask weighting in training is ineffective (needs V16.2 precise masks)
- V16.1.2 refiner is dead code (never trained, random distillation)

## Relevant Files
- `wow-viewer/data-harvester/src/harvester/v16_1_models.py` — all model classes
- `wow-viewer/data-harvester/scripts/train_v16_1_common.py` — training loop, all loss functions
- `wow-viewer/data-harvester/scripts/train_v16_1_combined.py` — V16.1.4 entrypoint
- `wow-viewer/data-harvester/scripts/export_terrain_obj.py` — OBJ export
- `wow-viewer/data-harvester/src/harvester/v16_1_dataset.py` — dataset loader
- `wow-viewer/specs/017-v16-1-4-combined-normal-height-model/` — V16.1.4 spec
- `wow-viewer/specs/020-renderer-culling-and-tile-capture/` — renderer fix spec (new)
- `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` — MotherShip context
