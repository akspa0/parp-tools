# MCLQ — ADT Terrain Liquid (Build 0.9.1.3810, macOS)

## Summary
`MCLQ` is present and actively consumed in build `0.9.1.3810`. The client allocates per-chunk liquid objects and builds liquid vertices from `MCLQ` payload data. No `MH2O` usage was found in this binary.

## Parent Chunk
`MCNK`

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.9.1.3810 | Variable | Confirmed via decompiler in `CMapChunk::CreatePtrs`, `CMapChunk::CreateLiquids`, and `CChunkLiquid::CreateVertices` |

## Structure — MCLQ subchunk payload inside MCNK (0.9.1.3810)

The loader treats `MCLQ` as **4 consecutive liquid blocks** (one slot per liquid type), each with a fixed stride.

### Per-liquid block layout (inferred)
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x000 | float | minLiquidZ (inferred) | Copied to `CChunkLiquid + 0x10`; used by `GetAaBox()` as liquid AABB min Z |
| 0x004 | float | maxLiquidZ (inferred) | Copied to `CChunkLiquid + 0x14`; used by `GetAaBox()` as liquid AABB max Z |
| 0x008 | bytes | vertexRecords | 0x280 bytes processed as 80 records (8 bytes each) |
| 0x288 | uint32 | reserved0 (inferred) | Not read by observed 0.9.1 runtime paths |
| 0x28C | uint32 | reserved1 (inferred) | Not read by observed 0.9.1 runtime paths |
| 0x290 | bytes | tileFlags8x8 | Pointer stored at `CChunkLiquid + 0x1c`; queried as 8×8 tile flags in map liquid queries |
| 0x2D0 | uint32 | flowModeOrCount (inferred) | Copied to `CChunkLiquid + 0x20`; controls flow vector logic (`0`, `1`, `2`) |
| 0x2D4 | bytes | flowData | `ConvertArrayToBinary<SWFlowv>(..., 2)` |
| 0x324 | — | nextLiquidBlock | Next liquid type block starts here |

Per-liquid block stride: **0x324 bytes** (`0xC9 dwords`).

### Total MCLQ data consumed per MCNK
`4 * 0x324 = 0xC90` bytes of structured liquid block data.

## Runtime interpretation notes
- `CMapChunk::CreateLiquids(int)` iterates `uVar17 = 0..3`, gated by `MCNK.flags` bits `0x4, 0x8, 0x10, 0x20`.
- For enabled liquid slots, it allocates/initializes `CChunkLiquid`, wires pointers into the `MCLQ` block, and calls `CChunkLiquid::CreateVertices()`.
- `CChunkLiquid` preallocates **0x51 (81) C3Vector vertices**, matching a 9×9 vertex domain.
- `CChunkLiquid::CreateVertices()` indexes source records with `index * 8 + 4` for height reads, indicating height is read from offset `+4` inside each 8-byte source record for the path used here.
- Terrain liquid query code reads tile classification from `CChunkLiquid+0x1c` and uses per-cell low bits (`& 0xF`, `& 0x3`) and high-bit flags (`>> 7`).
- Flow vector application is controlled by `CChunkLiquid+0x20`: `0` (no flow), `1` (single flow entry), `2` (dual-flow blend path).

## Endianness behavior
This macOS build uses explicit byte-swapping (`ConvertToBinary`, `ConvertArrayToBinary`, manual swaps) when `Create(..., param_2)` indicates conversion is required. This is consistent with platform/binary endianness handling in this client line.

## Version Differences (0.9.1-specific conclusions)
- `MCLQ` is still the active terrain liquid format in 0.9.1.3810.
- `MH2O` was not observed in strings or parser paths in this binary.
- Liquid block composition differs from late-era formats and should not be assumed 1:1 with 3.3.5 references.

## Ghidra Notes
- `CMapChunk::CreatePtrs` uses `MCNK + header[0x60]` and asserts token `'MCLQ'`.
- `CMapChunk::CreateLiquids(int)` at `0x00297ecc`.
- `CChunkLiquid::CreateVertices()` at `0x002984ec`.
- `CChunkLiquid::CChunkLiquid()` at `0x00298450` confirms 81-vertex storage.
- `CMap::QueryLiquidStatus(...)` at `0x00022e00` reads tile flags from `CChunkLiquid+0x1c`.
- `FUN_00023884` (terrain liquid normal/flow helper) uses `CChunkLiquid+0x20/+0x24` flow fields.

## Confidence
- **High**: presence, pointer wiring, per-type block stride, 81-vertex generation path.
- **Medium-High**: semantic naming of flow/tile fields due direct consumer-path evidence.
- **Medium**: `0x288/0x28C` remain observed-as-reserved in this build path.
