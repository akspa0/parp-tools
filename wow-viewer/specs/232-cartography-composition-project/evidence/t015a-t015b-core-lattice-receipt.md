# Receipt — Spec 232 T015a–T015b: full-tile lattice transform

**Date**: 2026-09-08 · **Spec**: [232](../spec.md) · **Tasks**: T015a, T015b

## Files changed

| File | Change |
|---|---|
| [AlphaTileData.cs](../../../src/core/WowViewer.Core/Maps/AlphaTileData.cs) | Added pure `RotateQuarterTurn(int, bool, bool)`. It moves full-tile 257×257 height/normal/MCCV/MCLV/liquid planes, 1024/256 shadow planes, alpha planes, 16×16 channel/area/hole/flags metadata, liquid chunk coordinates and local 9×9/8×8 grids, then leaves the input unchanged for `ToTileLoadResult` slicing. |
| [AlphaTileDataTransformTests.cs](../../../tests/WowViewer.Core.Tests/Maps/AlphaTileDataTransformTests.cs) | Added synthetic full-lattice direction/slot test and normal-plus-area metadata test. |
| [plan.md](../plan.md), [research.md](../research.md), [data-model.md](../data-model.md), [contracts/composition-boundary.md](../contracts/composition-boundary.md), [quickstart.md](../quickstart.md) | Restored the missing Spec Kit plan/design artifacts and operator-gate instructions. |
| [tasks.md](../tasks.md) | Checked only receipted T015a–T015b. |

`AlphaLiquidChunk` carries the authored Alpha MCLQ 9×9 height and 8×8 tile-flag grids. Its
separate 4×4 `TileGrid` is not present in the Core model: the viewer's legacy adapter currently
constructs that field as a zero-filled derived array. T015c will remove that per-MCNK adapter path;
there is therefore no authored Core `TileGrid` value to rotate in T015a.

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test tests\\WowViewer.Core.Tests\\WowViewer.Core.Tests.csproj -c Debug --filter "FullyQualifiedName~AlphaTileDataTransformTests\|FullyQualifiedName~TileContentTransformTests\|FullyQualifiedName~ResolveCellShiftedChunkTests\|FullyQualifiedName~PhaseLayerProjectFileTests"` | 0 | **31/31 passed**. |
| `dotnet build WowViewer.slnx -c Debug` | 0 | **0 errors** (the solution still reports existing warnings, including NU1903 advisories). |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| T015a: one full-tile index transform rotates channels and mirrors after rotation | `RotateQuarterTurn` applies the same `TileTransformKind` sequence to each full plane; `TransformYX`/`TransformXY` retain the parser's documented axis conventions. Liquid chunks and local square grids use the same sequence. | Pass (source + focused tests) |
| T015b: target MCNK direction agrees with `TransformChunkSlot` before rendering | `RotateQuarterTurn_SlicesTargetChunkFromTheSlotNamedByThePolicy` rotates a synthetic 257×257 lattice, slices it, and compares target (0,0) with the 90°-transformed source chunk that `TransformChunkSlot(0,15,Rotate90CW)` maps there. | Pass (31/31 focused tests) |
| T015c: live Alpha adapter consumes tile-level data | Not implemented in this receipt. The adapter still calls the old per-MCNK transform. | Open |
| T015d: rotated DeadminesInstance is seam-free with a cell nudge | Requires operator screenshot on configured real data. | Open, operator-owned |
