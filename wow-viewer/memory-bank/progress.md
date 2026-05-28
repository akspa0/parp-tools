# Progress — V16.x Terrain Training Pipeline

## Completed
- 2026-05-27: Expanded Spec 001 (`specs/001-v18-dataset-spec/`) into a fuller V18 dataset canonical contract and then revised it to reflect the simpler direction: V18 is the direct versioned successor to the V16 dataset creation flow, with decoded metadata and currently patched-on signals promoted into the main build contract
- 2026-05-27: Added `specs/001-v18-dataset-spec/plan.md` and `tasks.md` manually on `v0.5.0-dev` after bypassing Speckit branch-gated scripts
- 2026-05-27: Created `data-harvester/scripts/build_v18_dataset.py` as the copy-forward V18 builder, changed it to write under `output/datasets/v18`, added canonical V18 artifact/finalization helpers, and integrated optional renderer-truth promotion into the `build` command
- 2026-05-27: Upstreamed object-roof support into the shared harvest/tensor-pack contract by adding roof arrays plus roof-mask-source metadata to the C# pack and serializer surfaces, and taught `build_v18_dataset.py` to treat those arrays as canonical streamed V18 signals
- 2026-05-27: Added an explicit `--experimental-renderer-truth-promotion` gate so capture-derived V18 signals are not treated as proven canonical outputs before refreshed object-loading/capture validation
- 2026-05-27: Locked scope so the parser → decoded → dataset direct-pipeline redesign is deferred to a future V20 dataset effort instead of being folded into V18
- 2026-05-27: Bounded `build_v18_dataset.py build --limit 1` proof succeeded on staged `3_3_5_12340 / Azeroth`; emitted `finalization.json`, `signal_validation.json`, and `decoded_metadata_validation.json` with pass status
- 2026-05-27: Validation-capture dry-run readiness passed on staged `0_5_3_3368 / Azeroth_30_48` and `3_3_5_12340 / Azeroth_30_48`, but non-dry-run `gpu-viewer-style` capture on staged `3_3_5_12340 / Azeroth_30_48` still produced flat/uniform renders and an all-black object-visibility artifact
- 2026-05-27: Corrected the proof-owner boundary for roof/object visual evidence — `wow-viewer` command-path success is not enough; bounded `MdxViewer` compatibility proof remains the credible visual lane until parity is demonstrated 🧭
- 2026-05-23: V16.1.2 refiner added (later abandoned — gradient can't flow through detached graph)
- 2026-05-23: V16.1.3 height-channel normal model (4ch input) added and trained (plateaued epoch 123)
- 2026-05-24: V16.1.4 `V161NormalHeightCombinedModel` added to v16_1_models.py
- 2026-05-24: `_combined_loss` + `_preview_combined` + `combined` task registered in TASKS
- 2026-05-24: CLI flags `--normal-weight`, `--height-weight` added
- 2026-05-24: Spec 020 written (renderer culling fix → tile-level capture → V16.2 masks)
- 2026-05-24: Memory bank updated with MotherShip direction

## In Progress
- Bounded V18 dataset-builder validation still needed on staged client roots
- Real object-rendering proof for wow-viewer validation capture is still not closed despite command-path success
- Renderer culling fix needed before V16.2 dataset generation can proceed
- V16.1.4 combined model implemented but not yet trained
- Context-drift prevention: keep route, proof owner, and scope stated explicitly when the lane changes ⚠️

## Next Up
- Run a bounded `build_v18_dataset.py build` proof and inspect `finalization.json`, `signal_validation.json`, and `decoded_metadata_validation.json`
- Refresh real object-loading and capture proof on the bounded staged anchors before widening renderer-truth promotion claims
- Route the next capture fix through the existing renderer/culling specs instead of pretending the emitted flat artifacts are acceptable proof
- Simplify or retire the Python-only `patch_v18_object_roof_masks.py` workflow where the shared C# roof arrays now cover the same contract
- Use a bounded `MdxViewer` proof artifact before calling full ADT MCLY/object-inclusive data trustworthy 🧪
- Fix coordinate bug in `ComputeTilePlanarMin/Max` so single-tile capture works
- Tile-level loading (not full-map)
- Batch capture (Noggit-red composite pattern)
- V16.2 precise object mask generation
- V16.2 model with height-channel, combined heads, proper object weighting
- V18 object-roof curation + minimap sieve spec now drafted; next step is to split it into bounded implementation slices
- V18 object-roof lane now explicitly includes MdxViewer per-asset capture improvements and a separate object-visual Zarr store
- V18 object-roof lane now also explicitly calls for a Python `uv` + transformers object-identification model that feeds the main V18 model
- V18 object-roof lane now treats SAM2 as the initial promptable mask host and SAM3 as a gated follow-on if the Hugging Face token unlocks it

## MotherShip Direction
Long-range: `theMothership/game-engine/` — universal game engine with WoW plugin. See `wow-engine-modernization-plan-2026-05-14.md`.
