# Alpha Regression Checklist

## Read First

- `gillijimproject_refactor/memory-bank/activeContext.md`
- `gillijimproject_refactor/memory-bank/data-paths.md`
- `gillijimproject_refactor/src/MdxViewer/memory-bank/activeContext.md`
- `gillijimproject_refactor/src/MdxViewer/memory-bank/terrain_editing_plan_2026-02-14.md`

## Baseline Diff

- Compare current behavior and code against `343dadfa27df08d384614737b6c5921efe6409c8`.
- Prioritize diffs in:
  - `src/WoWMapConverter/WoWMapConverter.Core/Formats/LichKing/Mcal.cs`
  - `src/MdxViewer/Terrain/StandardTerrainAdapter.cs`
  - `src/MdxViewer/Terrain/TerrainTileMeshBuilder.cs`
  - `src/MdxViewer/Terrain/TerrainRenderer.cs`
  - `src/MdxViewer/Export/TerrainImageIo.cs`
  - `src/MdxViewer/ViewerApp.cs`

## Pipeline Map

- Decode: `Mcal.cs`
- Shared alternative decoder: `VLM/AlphaMapService.cs`
- Adapter extraction: `StandardTerrainAdapter.cs` and `AlphaTerrainAdapter.cs`
- Chunk storage: `TerrainChunkData.cs`
- GPU packing: `TerrainTileMeshBuilder.cs`
- Render and debug modes: `TerrainRenderer.cs`
- Import and export UI: `TerrainImageIo.cs`, `ViewerApp.cs`, `USERGUIDE.md`

## Real Data Paths

- `test_data/development/World/Maps/development`
- `test_data/WoWMuseum/335-dev/World/Maps/development`
- `test_data/minimaps/development`

## Minimum Validation

- Build `WoWMapConverter.Core.csproj` for parser changes.
- Build `MdxViewer.sln` for viewer changes.
- Check both Alpha-era data and LK 3.3.5 data if the change touches decode or blending rules.
- If import or export changed, verify the corresponding UI command still round-trips the expected layer layout.

## Recommended First-Party Tests

- `Mcal` decode tests:
  - 4-bit decode with edge fix
  - 4-bit decode without edge fix
  - big alpha direct copy
  - compressed alpha expansion
- `StandardTerrainAdapter` tests:
  - strict LK path with `UseAlpha`
  - split `*_tex0.adt` texture and alpha sourcing
  - fallback path for older data without LK flags
- `TerrainTileMeshBuilder` tests:
  - slice index mapping
  - RGBA packing of alpha plus shadow

## Reporting Rules

- Findings first.
- State clearly whether tests were added, only builds were run, or real-data validation was skipped.