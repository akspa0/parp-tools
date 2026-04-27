# Active Context

This file is intentionally compressed. Keep only the current route, the latest validated state, and the open boundaries here. Put deep history in plans, docs, or git history.

## Current Priorities

- `wow-viewer` is the canonical target for new shared I/O, runtime, PM4, dataset, and v10 terrain-AI ownership.
- `MdxViewer` is legacy or compatibility-only unless the task explicitly targets the old viewer or terrain archaeology.
- For world-viewer work, keep following the Apr 25 hard reset: port working `MdxViewer` world ownership into `wow-viewer` libraries first, then keep `WowViewer.App` thin.

## v10 Terrain AI Status

- Current position: Wave 1 is complete and Wave 2 is underway.
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
  - `wowviewer-converter extract-v10-tensors` remains the canonical Wave 1 NPZ extraction surface and now writes matching `*_placements.json` sidecars when placement data exists
  - `wowviewer-converter mine-v10-brushes` now owns the anchor-aware miner natively in `WowViewer.Tool.Converter`
  - the native miner supports `objects`, `terrain`, and `hybrid` anchor modes
  - the older Python miner under `gillijimproject_refactor/src/WoWMapConverter/scripts/v10/mine_mcal_brush_patterns.py` remains a reference surface, not the canonical command path
- Current proof level:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with warnings only
  - native widened hybrid proof passed at `output/build-validation/v10-wave2-wider-corpus/native-cli-hybrid-proof`

## Wave 2 Status

- What is landed:
  - Wave 1 NPZ extraction is available through `wowviewer-converter extract-v10-tensors`
  - placement-derived `ObjectMask257` and `ObjectPreciseMask257` are populated from real ADT placements
  - anchor-aware MCAL brush mining exists as a concrete native `wowviewer-converter mine-v10-brushes` command
  - terrain-only prefab structure can now be mined from alpha plus terrain mesh shape, even with no nearby objects
- Validated bounded artifacts:
  - widened corpus root: `output/build-validation/v10-wave2-wider-corpus/corpus`
  - terrain-only proof: `output/build-validation/v10-wave2-wider-corpus/terrain-proof/brush_dictionary.json`
  - hybrid proof: `output/build-validation/v10-wave2-wider-corpus/hybrid-proof/brush_dictionary.json`
  - native CLI proof: `output/build-validation/v10-wave2-wider-corpus/native-cli-hybrid-proof/brush_dictionary.json`
- What is still open:
  - MCLY combination mining
  - broader non-object-anchored MCAL composition vocabulary work
  - whether to retain or retire the older Python reference miner

## Open v10 Boundaries

- `ObjectMask257` and `ObjectPreciseMask257` are still placement-derived proxy masks, not true rendered silhouettes.
- `Pm4PathMask` and `Pm4BuildingFootprintMask` remain empty pending PM4 integration.
- No v10 trainer or model code is in place yet.

## wow-viewer Viewer Boundary

- Recent app-shell progress is real, but still transitional.
- `WowViewerWorldSceneHost` now owns more bootstrap and host-state seams, including bootstrap-only world session state before a full runtime frame exists.
- That does not change the main boundary: `WowViewerWorldRuntimeBridge` plus `WorldGpuPreviewRenderer` are still temporary bridge code, not the target world architecture.

## Read-First Reminder

- For active migration state: this file plus `memory-bank/progress.md`
- For fixed data roots: `memory-bank/data-paths.md`
- For detailed v10 intent: `plans/v10_full_terrain_ai_master_plan_2026-04-26.md`
- For world-viewer direction: `plans/wow_viewer_mdxviewer_cutaway_reset_plan_2026-04-24.md`
