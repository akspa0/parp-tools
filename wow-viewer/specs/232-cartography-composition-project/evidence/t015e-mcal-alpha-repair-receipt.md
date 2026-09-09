# Receipt — Spec 232 T015e: MCAL alpha repair for the full-tile composition route

**Date**: 2026-09-09 · **Spec**: [232](../spec.md) · **Task**: T015e (regression repair under the T015d gate)

## Defect

Operator report 2026-09-08: "something broke the MCLY layers on overlapped maps. It was working in
the last build." Screenshot: the rotated DeadminesInstance layer over Azeroth rendered flat,
single-texture patches with no layer blending.

Root cause: [AlphaWdtReader.TryReadTileInternal](../../../src/core/WowViewer.Core.IO/Maps/AlphaWdtReader.cs)
constructs `AlphaTileData` with the **256×256 downsampled** alpha pack (`alphaPack256`) as
`McalAlphaPack`, but [AlphaTileData.ToTileLoadResult](../../../src/core/WowViewer.Core/Maps/AlphaTileData.cs)
slices per-chunk alpha with a **64-px-per-chunk stride** (`SliceChunkAlpha(..., 64)`,
`srcY = cy*64 + y`). Against a 256-wide pack every chunk with cx or cy ≥ 4 reads out of bounds and
decodes **all-zero alpha**; chunks 0–3 read the wrong scale/region. `FillAlphaShadowSlice`
([TerrainTileMeshBuilder.cs](../../../src/viewer/WoWViewer/Terrain/TerrainTileMeshBuilder.cs)) then
uploads those zero maps, layers 1–3 render transparent, and each composed chunk collapses to its
base texture — the flat patches.

The T015 synthetic tests passed because `CreateTile` passes `mcalAlphaPack: null`, and the
pre-T015 route (`ExtractChunkData` → `ExtractAlphaMaps` from raw MCNK payloads) never called
`ToTileLoadResult` on real data. T015c made it the live transformed-path source, exposing the
latent stride mismatch.

## Files changed

| File | Change |
|---|---|
| [AlphaTileData.cs](../../../src/core/WowViewer.Core/Maps/AlphaTileData.cs) | Adds `McalAlphaPackFull` (1024×1024×4, optional ctor param). `RotateQuarterTurn` rotates it with the same `TransformYX` index map. `ToTileLoadResult` now slices chunk alpha through `SliceChunkAlphaForChunk`: full pack at 64 px/chunk when present, else nearest-neighbor 4× upsample of the 256 pack (16 px/chunk), else null (unchanged for alpha-less synthetic tiles). `McalAlphaPack` semantics (256², dataset `mcal_alpha_pack` contract) unchanged. |
| [AlphaWdtReader.cs](../../../src/core/WowViewer.Core.IO/Maps/AlphaWdtReader.cs) | Passes `mcalAlphaPackFull: hasAlpha ? alphaPack : null` (the full-resolution plane it already decoded) to the tile constructor. |
| [AlphaTileDataTransformTests.cs](../../../tests/WowViewer.Core.Tests/Maps/AlphaTileDataTransformTests.cs) | `CreateTile` gains mask/alpha-pack params. New tests: full-pack chunk slicing at (5,7) layer 2; 256-pack fallback upsampling at the correct chunk slot; rotation equivalence of layer alpha against the proven per-chunk `TransformChunk` alpha transform. |

## Verification

| Command | Exit | Real output |
|---|---:|---|
| `dotnet test tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~AlphaTileDataTransformTests"` | 0 | **7/7 passed** (was 4/4; 3 new regression tests). |
| `dotnet test tests\WowViewer.Core.Tests\WowViewer.Core.Tests.csproj -c Debug --no-restore --filter "FullyQualifiedName~Maps"` | 0 | **174/174 passed**. |
| `dotnet build WowViewer.slnx -c Debug --no-restore` | 0 | **0 errors**, 306 warnings (pre-existing). |

## Criterion → evidence

| Criterion | Evidence | Status |
|---|---|---|
| Chunks with cx/cy ≥ 4 decode non-zero, correctly-located alpha | `ToTileLoadResult_SlicesAlphaFromTheFullResolutionPackAtEveryChunk`: chunk (5,7) layer 2 reads pack value 0.75 → 191 at the authored pixel. | Pass (unit) |
| Tiles without the full plane still get correctly-located alpha | `ToTileLoadResult_UpsamplesThePacked256PlaneWhenFullResolutionIsAbsent`: 256-pack value 0.5 → 127 at the upsampled slot. | Pass (unit) |
| Rotation moves layer alpha with the same slot map as heights | `RotateQuarterTurn_MovesLayerAlphaWithTheSameSlotMapAsHeights`: full-tile rotated alpha equals the proven per-chunk `TransformChunk` alpha transform. | Pass (unit) |
| Rotated DeadminesInstance renders seam-free with blended texture layers | Requires the real configured client and an operator screenshot. | Open, operator-owned (T015d) |

## Notes

- `TerrainTileTensorPack.ToTileLoadResult` has the same latent 256-pack/64-stride mismatch but no
  live caller today; left untouched and flagged here as a follow-up, not silently fixed.
- `AlphaToLkConverter` adapts to the pack edge via `GetLength(0)/16` — unaffected.
