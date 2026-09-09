# Receipt — Spec 232 T015c: Alpha full-tile composition route

**Date**: 2026-09-08 · **Spec**: [232](../spec.md) · **Task**: T015c

## Files changed

| File | Change |
|---|---|
| [AlphaTerrainAdapter.cs](../../../src/viewer/WoWViewer/Terrain/AlphaTerrainAdapter.cs) | Adds `GetTileData(tileX, tileY)`, which contains the legacy-to-canonical Alpha MAIN coordinate swap. In both direct and cell-shifted transformed phase routes, `LoadTransformedFullTile` applies `RotateQuarterTurn` before `ToTileLoadResult` slices/re-homes the target. A core-read failure skips only the transformed contribution with one diagnostic; it does not reinstate the known seam-producing per-MCNK fallback. |
| [AlphaTileData.cs](../../../src/core/WowViewer.Core/Maps/AlphaTileData.cs) | Keeps MCNK flags when slicing a full tile and uses the viewer-consistent Alpha map origin when re-homing chunk world positions. |
| [AlphaTileDataTransformTests.cs](../../../tests/WowViewer.Core.Tests/Maps/AlphaTileDataTransformTests.cs) | Extends the synthetic contract with liquid local-grid, MCNK-flag, and target-world-position assertions. |
| [tasks.md](../tasks.md), [activeContext.md](../../../memory-bank/activeContext.md), [progress.md](../../../memory-bank/progress.md) | Checks only T015c and records the remaining visual gate. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~AlphaTileDataTransformTests\|FullyQualifiedName~TileContentTransformTests\|FullyQualifiedName~ResolveCellShiftedChunkTests\|FullyQualifiedName~PhaseLayerProjectFileTests"` | 0 | **33/33 passed**. |
| `dotnet build WowViewer.slnx -c Debug` | 0 | **0 errors**; 325 warnings reported, including NU1903 advisories. |
| `git diff --check` | 0 | No whitespace errors. Git emitted unrelated user-config access and CRLF notices. |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| T015c exposes unsliced Alpha tile data | `AlphaTerrainAdapter.GetTileData` calls `AlphaWdtReader.TryReadTile` and returns `AlphaTileData`; the single coordinate swap preserves the adapter's existing legacy MAIN convention. | Pass (source + cross-platform viewer build) |
| T015c replaces direct transformed per-MCNK rotation | `LoadTileWithPlacements` calls `LoadTransformedFullTile` for a non-empty transform list. That method calls `RotateQuarterTurn` then `ToTileLoadResult(targetTileX, targetTileY)`; Alpha adapter has no transformed-path call to `TransformChunksForTarget`. | Pass (source + 33/33 focused tests) |
| T015c keeps cell fine-tune source policy | `BuildCellShiftedTile` still obtains each supplying tile/chunk through `ResolveCellShiftedChunk` and `ResolveTileSource`; transformed suppliers use the same full-tile route before the selected chunk is copied into the final cell slot. | Pass (source + existing `ResolveCellShiftedChunkTests`) |
| T015c preserves relevant converted terrain payloads | The bridge converts sliced terrain, normal, layers/alpha, shadow, hole, area, MCNK-flag, and 9x9/8x8 liquid data into the viewer shape. The added synthetic test checks liquid-grid and flag slot mapping plus target world re-homing. | Pass (33/33 focused tests) |
| T015d: rotated DeadminesInstance is seam-free and accepts a cell nudge | Requires a real configured map and operator screenshot; no compilation or unit test establishes visual continuity. | Open, operator-owned |
