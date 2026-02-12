# MCNK — ADT Terrain Chunk Header/Dispatch (Build 0.9.1.3810, macOS)

## Summary
`MCNK` parsing in 0.9.1.3810 is strict and offset-driven. The client validates required subchunks by FourCC and uses header offsets to bind pointers for terrain, textures, references, sound emitters, shadows, and liquids.

## Parent Chunk
ADT root (`MCIN`/tile chunk table resolves to `MCNK` blocks)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.9.1.3810 | Fixed header + variable subchunks | Confirmed via `CMapChunk::CreatePtrs` and `CMapChunk::Create` |

## Structure — MCNK header fields used by this build

The following offsets are **confirmed as consumed** in code:

| Header Offset | Type | Name (inferred) | Usage |
|---------------|------|------------------|-------|
| 0x0C | uint32 | layerCount | Used by `ConvertArrayToBinary<SMLayer>` and layer loop in `Create` |
| 0x10 | uint32 | doodadRefCount | Used by `CreateRefs` call |
| 0x14 | uint32 | ofsMCVT | `MCVT` token check + pointer bind |
| 0x18 | uint32 | ofsMCNR | `MCNR` token check + pointer bind |
| 0x1C | uint32 | ofsMCLY | `MCLY` token check + pointer bind |
| 0x20 | uint32 | ofsMCRF | `MCRF` token check + pointer bind |
| 0x24 | uint32 | ofsMCAL | `MCAL` token check + pointer bind |
| 0x2C | uint32 | ofsMCSH | `MCSH` token check + pointer bind |
| 0x34 | uint32 | areaId (likely) | Copied to chunk state at `this+0x174`; consistent with classic MCNK header layouts |
| 0x38 | uint32 | mapObjRefCount | Added with doodad count for `MCRF` conversion/counting |
| 0x5C | uint32 | soundEmitterCount | Drives `CreateSoundEmitters()` loop count over `MCSE` entries |
| 0x58 | uint32 | ofsMCSE | `MCSE` token check + pointer bind |
| 0x60 | uint32 | ofsMCLQ | `MCLQ` token check + pointer bind |
| 0x68 | float | positionX? | Copied to chunk state |
| 0x6C | float | positionY? | Copied to chunk state |
| 0x70 | float | positionZ? | Copied to chunk state |

## Observed required subchunk order checks (from header offsets)
The parser enforces presence of:
- `MCVT`, `MCNR`, `MCLY`, `MCRF`, `MCSH`, `MCAL`, `MCLQ`, `MCSE`

All are validated with hard assertions (`iffChunk->token == 'XXXX'`).

## Runtime behavior linked to MCNK
- `CreateVertices()` is called after pointer binding.
- `CreateLiquids()` consumes `MCLQ` gated by `MCNK` flags bits.
- `CreateSoundEmitters()` consumes `MCSE`.
- `CreateRefs()` consumes `MCRF` with doodad/mapobj counts from header.
- `CreateLayer()` iterates by `layerCount` using `MCLY` + `MCAL`.

## Count Fields (validated)
- `MCNK+0x10` (`doodadRefCount`) and `MCNK+0x38` (`mapObjRefCount`) are both used as **counts**.
- Their sum controls conversion/iteration over `MCRF` entries, and they are split by type when creating references.
- This supports the broader pattern that previously "mystery" ADT fields in older builds can be count fields rather than opaque flags.

## Version Differences
- `MCLQ` is fully integrated in this build (pre-`MH2O` path).
- This layout already includes late-vanilla-like fields (`MCSE`, shadow/layer/ref machinery), but offsets should still be treated as version-specific.

## Ghidra Notes
- `CMapChunk::CreatePtrs(int)` observed around `0x00297438` onward.
- `CMapChunk::Create(unsigned char*, int)` at `0x00296d2c`.
- Assertions reference `MapChunk.cpp` line numbers around `0x3f1..0x432` for token checks.

## Confidence
- **High**: subchunk offsets/tokens and call graph integration.
- **Medium**: `0x34` interpreted as `areaId` from usage + historical layout consistency (no direct symbol label in this build).
