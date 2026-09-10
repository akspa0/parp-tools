# Receipt — Spec 232 T066: layer-rigid cell fine-tune

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
relative to each other instead of moving the layer as one object.

## Fix (amended 2026-09-09 after the tile-rigid pass)

First pass was TILE-rigid (one donor tile per target, edge content dropped) — the operator then
verified alignment was right but "we're missing stuff in between": content slid out of a target
tile was dropped instead of landing in the neighbor target tile. Final form is **LAYER-rigid**:

Both `BuildCellShiftedTile` implementations ([AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs),
[StandardTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/StandardTerrainAdapter.cs))
now collect content from the **3×3 target-tile neighborhood**: each contributor resolves its own
donor tile through the shared tile map (`ResolveTileSource`, rotation/mirror included), and its
chunk (sx, sy) lands at target slot (sx + CellOffsetX + 16·i, sy + CellOffsetY + 16·j). With
|CellOffset| ≤ 15 the three per-axis contributor ranges are disjoint — no slot conflicts, no
re-picking from unrelated tiles, nothing dropped between tiles. Placements ride their OWN donor
tile only (they are world-positioned and render regardless of the owning tile). Texture name
tables come from the center (primary) donor tile — same-map donor tables normally match.
`PhaseCompositionPolicy.ResolveCellShiftedChunk` is no longer used by the adapters (left in the
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
