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
- That commit moved the lane past pure Wave 1 extraction:
  - added `gillijimproject_refactor/src/WoWMapConverter/scripts/v10/mine_mcal_brush_patterns.py` for object-anchored 3D brush-pattern mining from Wave 1 NPZ outputs plus placement catalogs
  - extended `AdtTensorPackBuilder` so `ObjectMask257` and `ObjectPreciseMask257` are now populated from `MDDF` and `MODF` placements via `AdtPlacementReader`
  - kept the `wowviewer-converter extract-v10-tensors` entrypoint available as the canonical NPZ extraction surface
  - cleared the earlier converter build blocker in `TerrainPatchAdtCommand`
- Current build proof level:
  - `dotnet build i:/parp/parp-tools/wow-viewer/tools/converter/WowViewer.Tool.Converter/WowViewer.Tool.Converter.csproj -c Debug` passed.
  - `dotnet build i:/parp/parp-tools/wow-viewer/WowViewer.slnx -c Debug` passed with warnings only.

## Wave 2 Status

- Wave 2 is in progress, but only its first committed slice is landed.
- What is landed:
  - object-anchored MCAL brush mining exists as a concrete script surface
  - Wave 1 NPZ extraction is available through `wowviewer-converter extract-v10-tensors`
  - object placement supervision is no longer empty because `ObjectMask257` and `ObjectPreciseMask257` now derive from real ADT placements
- What is still missing:
  - no committed `wow-viewer` command yet runs the new object-anchored mining end-to-end
  - the new mining script still expects external per-tile placement JSON under `--placement-dir`; no committed placement-JSON export handoff was found in this slice
  - no recorded real-data dictionary artifact or bounded validation run exists yet for `object_anchored_brush_dictionary.npz`
  - MCLY combination mining and non-object-anchored MCAL brush or composition dictionary work are still open
- Practical handoff:
  - Wave 2 left off after landing the first object-anchored mining slice and placement-derived object masks
  - the next slice is to wire or export the placement handoff and run the first bounded real-data dictionary build

## Open v10 Boundaries

- `ObjectMask257` and `ObjectPreciseMask257` are now populated, but only as placement-derived proxy masks: circles for model placements and bound or rectangle projection for WMOs. They are not yet true rendered silhouettes.
- `Pm4PathMask` and `Pm4BuildingFootprintMask` remain empty pending PM4 integration.
- No v10 trainer or model code is in place yet.

## wow-viewer Viewer Boundary

- Recent app-shell progress is real, but still transitional.
- `WowViewerWorldSceneHost` now owns more bootstrap and host-state seams, including bootstrap-only world session state before a full runtime frame exists.
- That does not change the main boundary: `WowViewerWorldRuntimeBridge` plus `WorldGpuPreviewRenderer` are still temporary bridge code, not the target world architecture.
- Keep the canonical extraction order anchored to:
  - `MdxViewer.Rendering.Camera`
  - `MdxViewer.Terrain.WorldScene`
  - `MdxViewer.Terrain.WorldAssetManager`
  - `MdxViewer.Terrain.StandardTerrainAdapter`
  - `MdxViewer.Terrain.TerrainTileMeshBuilder`
  - `MdxViewer.Terrain.TerrainRenderer`
  - `MdxViewer.MinimapHelpers`

## Read-First Reminder

- For active migration state: this file plus `memory-bank/progress.md`
- For fixed data roots: `memory-bank/data-paths.md`
- For detailed v10 intent: `plans/v10_full_terrain_ai_master_plan_2026-04-26.md`
- For world-viewer direction: `plans/wow_viewer_mdxviewer_cutaway_reset_plan_2026-04-24.md`
