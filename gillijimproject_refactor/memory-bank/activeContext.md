# ACTIVE CONTEXT — V14 Branch (V11 Reset)

## BRANCH
`v0.4.9-strict-guards` forked from `971fff2` on 2026-05-06.

## wow-viewer Library Completeness — Phase A DONE

Phase A (terrain type system) is complete. All domain types and adapters are wired:

- `ITerrainAdapter` interface in `WowViewer.Core.Maps`
- `TerrainChunkData`, `TerrainLayer`, `LiquidChunkData` in `WowViewer.Core.Maps`
- `MddfPlacement`, `ModfPlacement` structs in `WowViewer.Core.Maps`
- `TileLoadResult` in `WowViewer.Core.Maps`
- `AlphaTerrainAdapter` in `WowViewer.Core.IO.Maps` — implements `ITerrainAdapter`, bridges `AlphaWdtReader` → per-chunk `TerrainChunkData`
- `AlphaTileData.ToTileLoadResult()` — converts flat 257×257 arrays → per-chunk `TerrainChunkData[]`
- `TerrainTileTensorPack.ToTileLoadResult()` — slices LK flat arrays → per-chunk `TerrainChunkData[]`
- `NativeMpqService` ported as gold-standard MPQ reader (pure C#, no StormLib)
- StormLib completely removed from wow-viewer
- Harvest tool `extract-unified` command added

**Next**: Phase B — validate AlphaWdtReader against known-good tiles, add MCNR/MCSH extraction, MCLQ upscaling.

## V11 TRAINER (`train_v11.py`)
- **Backbone:** ConvNeXt V2 Tiny (28.6M) from `timm`. LayerNorm, batch-size agnostic.
- **Total:** 35.5M params, fits batch 32 in 8GB, batch 64+ in 17GB.
- **Inputs:** 26 channels — minimap, MCAL alpha, normals, MCCV (3x dropout), coarse height, liquid, objects, PM4, hole, luma, gradient, range.
- **Outputs:** height_17/65/257 + MCAL alpha (4ch) + MCLY class + hole binary.
- **Loss:** Uncertainty-weighted sigmas per task. Automatic balancing.
- **Extras:** EMA, cosine+warmup, gradient clip, signal dropout, LRU cache (2GB cap).

## WHAT WORKS
- `dataset-build-v10-stage1 --input-dir <dir> --minimap-root <dir>` — filesystem mode, no archives
- `dataset-build-cache --input <curated> --output-dir <dir>` — v9 pipeline, now with MCAL/MCLY
- `train_v11.py <shards> --epochs N` — full training with all signals
- `infer_v11.py <checkpoint> <shards>` — predict heights + MCAL + MCLY + holes, export OBJ
- `WowViewer.Tool.Harvest extract-unified` — MPQ-backed WDT/ADT → NPZ shard pipeline

## WHAT BROKE (archive path, DONT USE)
- `--client-root` mode for old dataset-build (pre-harvest tool)
- `build_v10_2_dataset.py`, `train_v10_2_terrain_synth.py` — dead code
- Shadow masks — never exist on minimap tiles, removed from channel list

## KEY FILES — wow-viewer Library
- Domain types: `wow-viewer/src/core/WowViewer.Core/Maps/`
- IO readers: `wow-viewer/src/core/WowViewer.Core.IO/Maps/`
- AlphaTerrainAdapter: `wow-viewer/src/core/WowViewer.Core.IO/Maps/AlphaTerrainAdapter.cs`
- NativeMpqService: `wow-viewer/src/core/WowViewer.Core.IO/Files/NativeMpqService.cs`
- Harvest tool: `wow-viewer/tools/harvest/WowViewer.Tool.Harvest/Program.cs`
- Library completeness plan: `wow-viewer/docs/architecture/wow-viewer-library-completeness-plan-2026-05-06.md`

## NEXT
1. Phase B: validate AlphaWdtReader, add MCNR/MCSH extraction, MCLQ upscaling
2. Phase B: wire AlphaTileData.ToPlacementCatalog into harvest output
3. Phase B: test Alpha 0.6.0 split ADT through AdtTensorPackBuilder
4. Extract training shards via harvest tool on staged clients
