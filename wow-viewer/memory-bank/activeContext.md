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
- **V18 dataset canonical contract**: Spec Kit draft expanded in `specs/001-v18-dataset-spec/` 🧭
  - now frames V18 as the direct versioned successor to the V16 dataset creation flow
  - promotes decoded metadata plus currently patched-on V16 signal families into the main V18 build contract
  - defines finalized dataset status, mandatory decoded metadata parity, merge fallback coverage, and additive raw-blob sidecar boundaries
  - initial implementation slice now exists in `data-harvester/scripts/build_v18_dataset.py`
  - landed so far: V18 output root, finalization report writing, optional renderer-truth capture promotion during `build`, and upstreamed object-roof arrays/provenance in the shared harvest/tensor-pack contract ✅
  - renderer-truth promotion is now explicitly gated as experimental until object-loading/capture proof is refreshed ⚠️
  - dry-run readiness passes on staged `0_5_3_3368 / Azeroth_30_48` and `3_3_5_12340 / Azeroth_30_48`
  - non-dry-run `gpu-viewer-style` capture on staged `3_3_5_12340 / Azeroth_30_48` completed 4/4 variants but still produced flat/uniform renders and an all-black object-visibility artifact
  - current conclusion: command-path proof exists, but real object-rendering proof is still blocked 🧪
  - proof-owner correction: bounded `gillijimproject_refactor/src/MdxViewer` validation capture remains the only credible full-layer terrain + object-inclusive proof lane until wow-viewer visual parity is actually demonstrated 📌
  - explicit scope guard: the parser → decoded → dataset direct-pipeline redesign is deferred to a future V20 dataset effort, not V18

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
- bounded real-data proof for `build_v18_dataset.py`
- fix the real object-rendering/capture lane before widening any renderer-truth promotion claims
- simplify `patch_v18_object_roof_masks.py` usage now that the shared C# contract emits roof arrays and provenance directly
- use bounded `MdxViewer` compatibility proof for visual validation before calling roof/object data trustworthy 🧭
- **Spec 020**: Fix renderer culling (renderer doesn't see objects) → tile-level capture → batch capture → V16.2 object mask generation
- V16.1.4 combined model training (waiting on smoke test/launch)
- V16.2 model architecture (V16.1.x lessons + proper object-aware loss weighting)

## MotherShip Direction
The long-range target is a universal game engine (`game-engine`) with a plugin architecture. WoW data support lives in `GameEngine.Plugin.WoW`. The current `wow-viewer` work feeds into this. The repo structure is placeholder but the direction is real: `theMothership/` with `game-engine/` core + plugins + viewer + tools. See `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` for the engine-modernization context.

## Known Issues
- Renderer culls all objects — coordinate system bug in `ComputeTilePlanarMin/Max` (spec 013 diagnosed, not fixed)
- MdxViewer loads whole map for single-tile capture (performance bottleneck for V16.2 dataset gen)
- Object mask weighting in training is ineffective (needs V16.2 precise masks)
- Bounded wow-viewer non-dry-run validation capture can finish 4/4 variants while still producing flat/uniform renders and all-black object visibility output; treat artifact emission as insufficient proof of real object rendering
- V16.1.2 refiner is dead code (never trained, random distillation)
- Current context-drift guard: if route, proof owner, or scope changes, restate them explicitly before continuing work ⚠️

## Relevant Files
- `wow-viewer/data-harvester/src/harvester/v16_1_models.py` — all model classes
- `wow-viewer/data-harvester/scripts/train_v16_1_common.py` — training loop, all loss functions
- `wow-viewer/data-harvester/scripts/train_v16_1_combined.py` — V16.1.4 entrypoint
- `wow-viewer/data-harvester/scripts/export_terrain_obj.py` — OBJ export
- `wow-viewer/data-harvester/src/harvester/v16_1_dataset.py` — dataset loader
- `wow-viewer/specs/017-v16-1-4-combined-normal-height-model/` — V16.1.4 spec
- `wow-viewer/specs/020-renderer-culling-and-tile-capture/` — renderer fix spec (new)
- `wow-viewer/docs/architecture/wow-engine-modernization-plan-2026-05-14.md` — MotherShip context
