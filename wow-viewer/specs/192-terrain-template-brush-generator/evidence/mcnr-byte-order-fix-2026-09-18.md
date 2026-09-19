# Spec 192 Evidence — Generated-Map MCNR Byte-Order Fix (minimap shadow side)

Date: 2026-09-18

## Symptom

Operator report: synthesized minimaps generated from **new map data** (New Map Creator output)
show the terrain shadow on the wrong side — shadows on the **north-west** flank instead of the
south-east. Suspected normal flip.

## Root cause

`TemplatedTerrainGenerator` wrote MCNR normal components in `(X, Y, Z)` byte order, but the disk
MCNR component order is signed **`(X, Z, Y)`**:

- [`BlankAdtFactory.CreateUpNormals`](../../../src/core/WowViewer.Core.IO/Maps/BlankAdtFactory.cs:307)
  documents it explicitly: *"Disk MCNR component order is signed X, Z, Y. Flat terrain points up."*
  and writes `byte[1] = 127`.
- [`AlphaTerrainAdapter.DecodeNormal`](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs:1881)
  reads `data[0]=X, data[1]=Z, data[2]=Y`.
- [`WorldTerrainTileBuilder.TryReadMcnrNormals`](../../../src/core/WowViewer.Core.Runtime/World/Terrain/WorldTerrainTileBuilder.cs:298)
  reads the same order.

Both generator writers were wrong:

- `RecalculateChunkNormals` wrote `nx→[0], ny→[1], nz→[2]`.
- `GenerateFlatNormals` wrote `0, 0, 127` (up in the Z slot, i.e. `(X, Y, Z)`).

So on disk the up component landed in the horizontal **Y** axis and the Y slope landed in **Z**.
When read back with the correct `(X, Z, Y)` order, every generated slope shaded on the wrong side —
exactly the reported north-west/south-east flip. The compositor's own sun-direction tests
(`TerrainMinimapSunDirectionTests`) pass for correctly-ordered normals, confirming the compositor
convention is right and the generated data was the defect.

## Fix

[`TemplatedTerrainGenerator.cs`](../../../src/core/WowViewer.Core.IO/Terrain/TemplatedTerrainGenerator.cs):

- `RecalculateChunkNormals`: write `nx→[0], nz→[1], ny→[2]`.
- `GenerateFlatNormals`: write `0, 127, 0` (up in the Z slot).

## Verification

| Command | Exit | Output |
|---|---:|---|
| `dotnet test wow-viewer/tests/WowViewer.Core.Tests/WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~TemplatedTerrainGenerator"` | 0 | `Passed! - Failed: 0, Passed: 7, Skipped: 0, Total: 7` |

New regression test `GenerateMap_WritesMcnrInDiskXzyOrderSoTerrainNormalsPointUp` decodes every
generated chunk's MCNR with the disk `(X, Z, Y)` order and requires the up component to dominate
(`Z > 0.5`) for all 145 vertices. It failed at vertex 81 before the `GenerateFlatNormals` half of the
fix and passes after.

## Proof boundary

This is a source + unit-test fix. **No runtime visual proof is claimed** — the operator should
re-generate a map and re-export its synthesized minimap to confirm the shadow now falls south-east.

## Related operator items (not addressed here)

Filed from the same report, still open:
1. Liquids (ocean/other layers) render with grid lines/omissions in synthesized minimaps.
2. Option to **not** apply DXT1 compression to synthesized-minimap outputs (post-0.5.3 versions
   differ too much).
3. Option to **include MCCV** vertex colors in synthesized minimaps.