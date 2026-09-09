# Receipt — Spec 232 T066: tile-rigid cell fine-tune

**Date**: 2026-09-09 · **Spec**: [232](../spec.md) · **Task**: T066 (operator directive)

## Defect

Operator reports: "cell finetuning shifts the cells around instead of just moving the TILE to the
cells, so it's shifting a single chunk and the tile, instead" and (with heightmap enabled, rotated
layer, cell offset −5/12) "applying the heightmap still does this horrific shit" — screenshot
showed checkerboard missing chunks, white untextured plates, and terrain at wrong elevations.

Root cause: both adapters' `BuildCellShiftedTile` resolved a **supply tile per composed chunk**
(`ResolveCellShiftedChunk` → `ResolveTileSource(supplyTile)` → per-tile load + transform). Under
rotation/cell offset that per-chunk resolution composed inconsistently (rotation inverse applied
at tile granularity only), pulled border chunks from unrelated donor tiles, and mixed texture
tables — producing the scrambled tiles. It also made the cell shift move individual chunks
relative to each other instead of moving the tile as one object.

## Fix

Both `BuildCellShiftedTile` implementations ([AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs),
[StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs))
are now **tile-rigid**:

- The layer's own donor tile for the target is resolved ONCE through the shared tile map
  (`ResolveTileSource`, rotation/mirror included).
- The donor content (full-tile transformed where applicable) slides by the cell offset inside the
  target tile: composed chunk (cx, cy) ← donor chunk (cx − CellOffsetX, cy − CellOffsetY);
  content pushed past a tile edge is dropped.
- Placements ride the same rigid move (pose transform + tile-offset + cell-delta translation);
  the per-target-tile rect filter is gone (exactly one donor tile feeds each target).
- `PhaseCompositionPolicy.ResolveCellShiftedChunk` is no longer used by the adapters (left in the
  policy for reference).

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**. |
| `dotnet test ... --filter "FullyQualifiedName~Maps"` | 0 | **177/177 passed**. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| Cell fine-tune moves the whole tile (heights, texturing, placements together) by whole cells | Code-level tile-rigid composition; visual witness operator-owned. | Open, operator-owned |
| Rotated layer + cell offset + heightmap composes without checkerboard/white plates | Requires the real configured map; visual witness operator-owned. | Open, operator-owned |
