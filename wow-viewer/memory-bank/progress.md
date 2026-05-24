# Progress — V16.x Terrain Training Pipeline

## Completed
- 2026-05-23: V16.1.2 refiner added (later abandoned — gradient can't flow through detached graph)
- 2026-05-23: V16.1.3 height-channel normal model (4ch input) added and trained (plateaued epoch 123)
- 2026-05-24: V16.1.4 `V161NormalHeightCombinedModel` added to v16_1_models.py
- 2026-05-24: `_combined_loss` + `_preview_combined` + `combined` task registered in TASKS
- 2026-05-24: CLI flags `--normal-weight`, `--height-weight` added
- 2026-05-24: Spec 020 written (renderer culling fix → tile-level capture → V16.2 masks)
- 2026-05-24: Memory bank updated with MotherShip direction

## In Progress
- Renderer culling fix needed before V16.2 dataset generation can proceed
- V16.1.4 combined model implemented but not yet trained

## Next Up
- Fix coordinate bug in `ComputeTilePlanarMin/Max` so single-tile capture works
- Tile-level loading (not full-map)
- Batch capture (Noggit-red composite pattern)
- V16.2 precise object mask generation
- V16.2 model with height-channel, combined heads, proper object weighting

## MotherShip Direction
Long-range: `theMothership/game-engine/` — universal game engine with WoW plugin. See `wow-engine-modernization-plan-2026-05-14.md`.
