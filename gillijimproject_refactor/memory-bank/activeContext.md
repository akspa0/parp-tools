# ACTIVE CONTEXT — V14 Branch (V11 Reset)

## BRANCH
`v0.4.9-strict-guards` forked from `971fff2` on 2026-05-06.

## wow-viewer Library Completeness / Harvest Status — Resynced 2026-05-07

Phase A is complete. The current truth at HEAD is no longer just "Alpha WDT validation done"; the current `wow-viewer` harvest/tensor-pack lane is working across staged clients from Alpha `0.5.x` through Cataclysm `4.0.0`, with Alpha-specific fixes, placement export, and object-mask generation all landed.

### Landed In The Recent May 6-7 Commits
- `NativeMpqService` is the active MPQ-backed reader for `WowViewer.Tool.Harvest extract-unified`.
- `ITerrainAdapter`, `TerrainChunkData`, `TerrainLayer`, `TileLoadResult`, and `AlphaTerrainAdapter` landed in `wow-viewer`.
- `AlphaWdtReader` now correctly skips the embedded `MCLY` chunk header and extracts `MCVT`, `MCNR`, `MCLY`, `MCAL`, `MCSH`, and `MCLQ` for the harvest path.
- `AlphaTensorPackBuilder` now emits the missing Alpha signal metadata and also generates `object_mask_257`, `object_precise_mask_257`, and `shadow_residual_mask_256`.
- `extract-unified --export-placements` now writes Alpha placement catalogs from `AlphaTileData.ToPlacementCatalog()`.
- Broad staged NPZ/tensor-pack support is proven on `0_5_3_3368`, `0_5_5_3494`, `0_7_0_3694`, `3_0_1_8303`, `3_3_5_12340`, and `4_0_0_11927`.

### Important Details Worth Preserving
- The Alpha `MCLY` subchunk includes an 8-byte chunk header before the layer-entry array. Not skipping that header corrupts texture ids, layer flags, and MCAL offsets.
- Tile-level flat arrays use natural row-major storage: `IndexX` is the chunk column and `IndexY` is the chunk row. World-space swizzle happens later in consumer code, not in the tensor-pack storage layout.
- Shadow residual is now defined as MCSH occupancy not explained by the precise object mask. It is a derived diagnostic signal, not a native client payload.

## WHAT WORKS
- `extract-unified` for Alpha monolithic WDT tiles on staged `0.5.3` and `0.5.5`
- `AdtTensorPackBuilder` / harvest tensor-pack generation on staged `0.7.0`, `3.0.1`, `3.3.5`, and `4.0.0`
- Alpha placement export through `--export-placements`
- Alpha and retail object footprint mask generation in the current tensor-pack contract
- Metadata JSON with current `AvailableSignals` coverage for the active harvest path

## WHAT IS STILL OPEN
- Explicit Alpha `0.6.0` split-ADT validation through `AdtProfile060070Baseline`
- Broader deep-reader/library closure beyond the current harvest/tensor-pack path
- Full multibuild corpus extraction and real training runs beyond the bounded staged-client proofs

## WHAT BROKE / DO NOT ROUTE BACK TO
- `--client-root` mode for the older pre-harvest dataset-build path
- `build_v10_2_dataset.py` and `train_v10_2_terrain_synth.py` as active architecture owners

## KEY FILES — wow-viewer Library
- Domain types: `wow-viewer/src/core/WowViewer.Core/Maps/`
- IO readers: `wow-viewer/src/core/WowViewer.Core.IO/Maps/`
  - `AlphaWdtReader.cs` — Alpha WDT parser for the harvest path
  - `AlphaTensorPackBuilder.cs` — AlphaTileData → TerrainTileTensorPack bridge
  - `AlphaTerrainAdapter.cs` — AlphaTileData → TerrainChunkData bridge
  - `AdtTensorPackBuilder.cs` — split-ADT tensor-pack builder across later builds
  - `NpzTileSerializer.cs` — TerrainTileTensorPack → NPZ serialization
- Native MPQ reader: `wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs`
- Harvest tool: `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs`
- Library completeness plan: `wow-viewer/docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md`
- Format spec: `gillijimproject_refactor/docs/ADT_WDT_Format_Specification.md`

## NEXT
1. Explicitly validate `0.6.0` split ADT through `AdtProfile060070Baseline`.
2. Decide whether the next slice is deeper format ownership (`WDT`/`WDL`/converters) or broader training-corpus extraction on the now-working harvest path.
3. Keep the library completeness plan and README aligned with the current harvest/tensor-pack truth so future chats do not route back to stale "Phase B pending" assumptions.
