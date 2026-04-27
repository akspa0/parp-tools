# Active Context

This file is intentionally compressed. Keep only the current route, the latest validated state, and the open boundaries here. Put deep history in plans, docs, or git history.

## Current Priorities

- `wow-viewer` is the canonical target for new shared I/O, runtime, PM4, dataset, and v10 terrain-AI ownership.
- `MdxViewer` is legacy or compatibility-only unless the task explicitly targets the old viewer or terrain archaeology.
- For world-viewer work, keep following the Apr 25 hard reset: port working `MdxViewer` world ownership into `wow-viewer` libraries first, then keep `WowViewer.App` thin.

## v10 Terrain AI Status

- Current position: Wave 1 is complete. Wave 2 pattern-mining infrastructure is complete. Stage 2 refinement is now underway.
- Wave 1 validated outputs in `wow-viewer` shared libraries:
  - `TerrainTileTensorPack`
  - `AdtTensorPackBuilder`
  - `NpzTileSerializer`
  - `AdtMclqReader`
  - `AdtMtxfReader`
  - `AdtMcrfReader`
  - `WlFile` and `WlFileReader`
- Latest committed Wave 2 slice is `f125fa5` (`feat: Add extraction of object-anchored 3D brush patterns and related functionality`).
- Current local continuation moved beyond that commit:
  - `wowviewer-converter extract-v10-tensors` remains the canonical Wave 1 NPZ extraction surface, now accepts `--minimap-root`, and writes matching `*_placements.json` sidecars when placement data exists
  - `wowviewer-converter dataset-build-v10-stage1` now owns the first bounded bulk Stage 1 corpus build over root ADTs plus a loose minimap root and writes a manifest that downstream trainers can consume directly
  - `wowviewer-converter mine-v10-brushes` now owns the anchor-aware miner natively in `WowViewer.Tool.Converter`
  - `wowviewer-converter mine-v10-mcly` now owns the native Wave 2 MCLY combination dictionary command, consumes texture-name metadata when present, and writes texture-path keyed `mclay_dictionary.json` plus `mcly_dictionary.json`
  - `wowviewer-converter label-v10-mcly` now owns reusable supervised MCLY label-manifest materialization from Stage 1 shards plus the mined MCLY dictionary
  - `wowviewer-converter mine-v10-mcal-compositions` now owns native chunk-level MCAL composition vocabulary mining and writes `mcal_composition_dictionary.json` plus `mcal_composition_dictionary.npz`
  - `wowviewer-converter mine-v10-mcal-brushes` now owns native non-object-anchored MCAL brush-stroke vocabulary mining and writes `mcal_brush_dictionary.json` plus `mcal_brush_dictionary.npz`
  - `wowviewer-converter mine-v10-height-profiles` now owns native height archetype clustering and writes `height_profile_dictionary.json` plus `height_profile_dictionary.npz`
  - `wowviewer-converter mine-v10-prefab-cells` now owns native rectangular prefab-cell clone detection, finds repeating chunk sets via strict SHA256 fingerprints, and writes `prefab_cell_dictionary.json` plus `prefab_cell_dictionary.npz`
  - `wow-viewer/scripts/curate_v10_training_shards.py` now curates mixed native v10 and legacy v9 NPZ shards into a balanced v10 training manifest
  - `wow-viewer/scripts/train_v10_stage2_terrain_synth.py` now consumes native PM4 masks and legacy v9 cache aliases for hole, precise-object, PM4, and liquid signals
  - `wow-viewer/scripts/train_v10_stage1_minimap2height.py` now exists as the first bounded Stage 1 trainer for `minimap_rgb_256 -> height_17`
  - `wow-viewer/scripts/train_v10_minimap_to_mclay.py` now exists as the first bounded Wave 2 classifier trainer for `minimap_rgb_256 -> retained MCLY palette label`
  - `wow-viewer/scripts/train_v10_minimap_to_mclay_grid.py` now exists as the first bounded Wave 2 chunk-grid classifier trainer for `minimap_rgb_256 -> 16x16 retained MCLY palette labels`, and can consume the reusable native MCLY label manifest directly
  - the native miner supports `objects`, `terrain`, and `hybrid` anchor modes
  - the older Python reference miner has been retired to `gillijimproject_refactor/src/WoWMapConverter/scripts/v10/archived/mine_mcal_brush_patterns.py`; the canonical path is `wowviewer-converter mine-v10-brushes`
- Current proof level:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with warnings only
  - native widened hybrid proof passed at `output/build-validation/v10-wave2-wider-corpus/native-cli-hybrid-proof`
  - enriched native MCLY dictionary proof passed at `output/build-validation/v10-wave2-mcly-dictionary/mclay_dictionary.json` after regenerating the 64-shard Stage 1 corpus with `mcly_texture_names`
  - native MCLY label-manifest proof passed at `output/build-validation/v10-wave2-mcly-labels/v10_mcly_label_manifest.json`
  - native MCAL composition proof passed at `output/build-validation/v10-wave2-mcal-compositions/mcal_composition_dictionary.json` over the current Stage 1 corpus
  - native MCAL brush dictionary proof passed at `output/build-validation/v10-wave2-mcal-brushes/mcal_brush_dictionary.json` over the current Stage 1 corpus
  - native height profile proof passed at `output/build-validation/v10-wave2-height-profiles/height_profile_dictionary.json` over all `64` Stage 1 shards
  - native prefab-cell clone detection proof passed at `output/build-validation/v10-wave2-prefab-cells-large/` with large-cell retained cells from `11` readable tiles:
    - `8x8`: `2` retained cells
    - `12x12`: `1` retained cell
    - `16x16`: `0` retained cells
  - per-layer alpha coverage signatures now captured in relaxed hash and JSON output (schema v2)
  - bounded Stage 1 corpus build passed at `output/build-validation/v10-stage1-development-corpus` with `64` written shards and a matching `v10_stage1_manifest.json`
  - bounded Stage 1 CUDA smoke passed at `output/ml-training/v10_stage1_gpu_smoke` using `gillijimproject_refactor/.venv-train` on the local RTX 4070 Ti SUPER
  - v10 mixed-corpus curation passed at `output/ml-training/v10_curated/v10_v9all_plus_native_dev_balanced_manifest.json` with `1,262` selected shards across `22` dataset buckets from the all-version v9 direct cache plus native v10 development shards
  - v10 Stage 2 CUDA run 1 passed at `output/ml-training/v10_stage2_balanced_cuda_run1/checkpoints/best.pt` for `3` epochs over the `1,262` selected shards after repairing `gillijimproject_refactor/.venv-train`
  - bounded minimap-to-MCLY CPU smoke passed at `output/ml-training/v10_minimap_to_mclay_smoke/minimap_to_mclay_classifier.pt` using the workspace `.venv` over the current `64` Stage 1 shards and `mclay_dictionary.json`
  - bounded minimap-to-MCLY chunk-grid CPU smoke passed at `output/ml-training/v10_minimap_to_mclay_grid_smoke/minimap_to_mclay_grid_classifier.pt`, with `1,973` retained chunk labels across `35` active palette classes
  - bounded manifest-driven minimap-to-MCLY chunk-grid CPU smoke passed at `output/ml-training/v10_minimap_to_mclay_grid_manifest_smoke/minimap_to_mclay_grid_classifier.pt`

## Wave 2 Status

- What is landed:
  - Wave 1 NPZ extraction is available through `wowviewer-converter extract-v10-tensors`
  - minimap-backed Stage 1 corpus materialization is available through `wowviewer-converter dataset-build-v10-stage1`
  - placement-derived `ObjectMask257` and `ObjectPreciseMask257` are populated from real ADT placements
  - anchor-aware MCAL brush mining exists as a concrete native `wowviewer-converter mine-v10-brushes` command
  - MCLY texture-layer combination mining exists as a concrete native `wowviewer-converter mine-v10-mcly` command with texture-name keyed palettes and conservative biome tags
  - retained MCLY label-grid materialization exists as a concrete native `wowviewer-converter label-v10-mcly` command
  - MCAL chunk-level composition mining exists as a concrete native `wowviewer-converter mine-v10-mcal-compositions` command with averaged 64x64x4 composition centroids
  - non-object-anchored MCAL brush-stroke vocabulary mining exists as a concrete native `wowviewer-converter mine-v10-mcal-brushes` command with per-layer 64x64 stamps and coarse shape-family labels
  - height profile clustering exists as a concrete native `wowviewer-converter mine-v10-height-profiles` command with normalized and absolute height centroids
  - rectangular prefab-cell clone detection exists as a concrete native `wowviewer-converter mine-v10-prefab-cells` command with strict fingerprint grouping and averaged cell centroid output
  - terrain-only prefab structure can now be mined from alpha plus terrain mesh shape, even with no nearby objects
  - the first bounded Stage 1 trainer exists in `wow-viewer/scripts/train_v10_stage1_minimap2height.py`
  - the first bounded minimap-to-MCLY palette classifier trainer exists in `wow-viewer/scripts/train_v10_minimap_to_mclay.py`
  - the first bounded minimap-to-MCLY 16x16 palette-grid classifier trainer exists in `wow-viewer/scripts/train_v10_minimap_to_mclay_grid.py`
- Validated bounded artifacts:
  - widened Wave 2 corpus root: `output/build-validation/v10-wave2-wider-corpus/corpus`
  - terrain-only proof: `output/build-validation/v10-wave2-wider-corpus/terrain-proof/brush_dictionary.json`
  - hybrid proof: `output/build-validation/v10-wave2-wider-corpus/hybrid-proof/brush_dictionary.json`
  - native CLI proof: `output/build-validation/v10-wave2-wider-corpus/native-cli-hybrid-proof/brush_dictionary.json`
  - MCLY dictionary proof: `output/build-validation/v10-wave2-mcly-dictionary/mclay_dictionary.json`
  - MCLY label-manifest proof: `output/build-validation/v10-wave2-mcly-labels/v10_mcly_label_manifest.json`
  - MCAL composition proof: `output/build-validation/v10-wave2-mcal-compositions/mcal_composition_dictionary.json`
  - MCAL brush dictionary proof: `output/build-validation/v10-wave2-mcal-brushes/mcal_brush_dictionary.json`
  - height profile proof: `output/build-validation/v10-wave2-height-profiles/height_profile_dictionary.json`
  - prefab-cell clone detection proof: `output/build-validation/v10-wave2-prefab-cells/prefab_cell_dictionary.json`
  - Stage 1 development corpus: `output/build-validation/v10-stage1-development-corpus/v10_stage1_manifest.json`
  - Stage 1 CUDA smoke output: `output/ml-training/v10_stage1_gpu_smoke`
  - minimap-to-MCLY classifier smoke output: `output/ml-training/v10_minimap_to_mclay_smoke`
  - minimap-to-MCLY grid classifier smoke output: `output/ml-training/v10_minimap_to_mclay_grid_smoke`
  - manifest-driven minimap-to-MCLY grid classifier smoke output: `output/ml-training/v10_minimap_to_mclay_grid_manifest_smoke`
- What is still open:
  - broader-corpus validation for the native MCAL brush dictionary beyond the bounded development Stage 1 corpus
  - broader-corpus minimap-to-MCLY classifier training beyond the `11` currently labelable development shards
  - native v10 regeneration for every v9-era client root; current broad training uses the legacy v9 NPZ cache for all-version coverage and native v10 shards only for the development-map richer-signal slice
  - richer biome tagging beyond current texture-name token heuristics
  - Stage 2 refinement and longer-running Stage 1 training orchestration

## Open v10 Boundaries

- `ObjectMask257` and `ObjectPreciseMask257` are still placement-derived proxy masks, not true rendered silhouettes.
- `Pm4PathMask` and `Pm4BuildingFootprintMask` remain empty pending PM4 integration.
- The current Stage 1 trainer is only a bounded baseline; Stage 2 refinement, broader non-development corpora, and production-grade experiment management remain open.
- `dataset-build-v10-stage1` currently skipped `34` development tiles because `AdtTensorPackBuilder` did not recognize them as usable root ADTs.
- The current enriched MCLY dictionary proof read `mcly_texture_ids` plus texture-name metadata from `11` of the `64` existing Stage 1 shards; `53` shards were explicitly skipped as `missing_mcly_texture_ids`.

## wow-viewer Viewer Boundary

- Recent app-shell progress is real, but still transitional.
- `WowViewerWorldSceneHost` now owns more bootstrap and host-state seams, including bootstrap-only world session state before a full runtime frame exists.
- That does not change the main boundary: `WowViewerWorldRuntimeBridge` plus `WorldGpuPreviewRenderer` are still temporary bridge code, not the target world architecture.

## Read-First Reminder

- For active migration state: this file plus `memory-bank/progress.md`
- For fixed data roots: `memory-bank/data-paths.md`
- For detailed v10 intent: `plans/v10_full_terrain_ai_master_plan_2026-04-26.md`
- For world-viewer direction: `plans/wow_viewer_mdxviewer_cutaway_reset_plan_2026-04-24.md`
