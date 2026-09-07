# Receipt — Spec 232 Phase 1: cell-level alignment (T010, T011, T012, T013)

**Date**: 2026-09-07 · **Spec**: [232](../spec.md) FR-1

## Files changed

| File | Change |
|---|---|
| [PhaseComposition.cs](../../../src/core/WowViewer.Core/Maps/PhaseComposition.cs) | `PhaseLayerSettings.CellOffsetX/Y` (±15 cells) + `HasCellOffset` + Clone copies; **`ResolveChunkSource`** — chunk-granularity resolution: donor TILE from `ResolveTileSource` (the discrete map streaming/minimap already use), cell delta inverse-rotated into the donor frame (`InverseRotateCellDelta`), donor chunk re-indexed with cross-border pull from the adjacent donor tile, grid-confined to 0..63 |
| [AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs) | `HasCellOffset` branch → `BuildCellShiftedTile`: composes the target tile chunk-by-chunk through `ResolveChunkSource`, donor tiles pulled on demand (±1 ring), content rotation kinds applied per donor chunk, world positions recomputed for the target slot; placements are the union of every donor tile pulled |
| [StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs) | Same chunk-granularity path for split-ADT (local chunk types, in-place list refill, primary donor tile's texture table) |
| Both adapters | `TranslatePhasePlacements` extended with the cell world delta (`TileOffsetToWorldTranslation(CellOffset, ChunkSize/16)`); merge guard extended to `HasTileOffset \|\| HasCellOffset` |
| [ViewerApp_PhaseLayers.cs](../../../src/viewer/WoWViewer/ViewerApp_PhaseLayers.cs) | **Cell fine-tune UI**: per-axis cell inputs (−15..+15, 1 cell = 1 MCNK = 1/16 tile) + reset, with tooltips |
| [ResolveChunkSourceTests.cs](../../../tests/WowViewer.Core.Tests/Maps/ResolveChunkSourceTests.cs) | **New** — 7 tests locking the invariants |

## Design notes

1. **Chunk resolution derives from the discrete tile map** (`ResolveTileSource`), not a parallel
   continuous map. First attempt used a continuous chunk-space rotation, which disagreed with the
   discrete tile map at tile boundaries (caught by
   `RotationOnly_AgreesWithTheDiscreteTileMap` before it could ship); the final design composes
   the discrete donor tile + the inverse-rotated cell delta, so streaming admission, minimap, and
   chunk content can never disagree.
2. **Cross-border pull**: a cell shift that moves a composed chunk's source past the donor tile
   edge reads the adjacent donor tile (chunk ±16 in global space), grid-confined.
3. **Placements**: uniform world delta (cell shift is a translation for absolute-positioned
   objects), applied with the existing tile-offset translation.

## Verification

- `dotnet build wow-viewer/WowViewer.slnx -c Debug` — **0 Errors**.
- `dotnet test --filter 'FullyQualifiedName~Maps'` — **164/164 passed**, including the 7 new
  `ResolveChunkSourceTests`: zero-transform identity, within-tile re-index, cross-border pull,
  tile+cell composition, rotation agreement with the discrete tile map, grid confinement, and
  out-of-grid donor rejection.
- **Operator visual gate (SC-1)**: nudge the DeadminesInstance roadway into alignment with the
  Moonbrook buildings using the cell inputs — open.
