# WMO Group Chunks — MOGP, MOTV, MLIQ (Build 0.9.1.3810, macOS)

## Summary
Build `0.9.1.3810` has a strict WMO group parser path with a fixed required chunk sequence in `CMapObjGroup`, plus flag-gated optional chunks. `MOTV` appears as a single required chunk in this parser path. `MLIQ` is parsed as an optional `MOGP` feature.

## Parent Chunk
WMO Group file body (`MOGP` container)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.9.1.3810 | Variable | Confirmed via `CMapObjGroup::Create`, `CreateDataPointers`, `CreateOptionalDataPointers` |

## Required group sequence (0.9.1.3810)
After `MVER` and `MOGP`, the parser expects this strict order:
1. `MOPY`
2. `MOVI`
3. `MOVT`
4. `MONR`
5. `MOTV`
6. `MOBA`
7. Optional chunk region (flag-gated)

Any mismatch triggers `_SErrDisplayError` assertions.

## MOGP header observations
`CMapObjGroup::Create()` reads and endian-converts `MOGP` header fields, then stores:
- Group flags (`this+0x0C`) used to gate optional subchunks
- Bounding/transform-ish fields copied from header words/halfs
- Name/description offsets and metadata copied into group state

Header interpretation is mature but not fully symbolized in this pass; unresolved fields are kept as inferred.

## MOTV findings (0.9.1)
- Exactly one `MOTV` chunk is consumed in `CreateDataPointers`.
- UV count computed as `chunkSize >> 3` (8 bytes per UV entry).
- Data is endian-converted as 32-bit values in-place.
- No second `MOTV` parse path was observed in this function.

## MLIQ findings (WMO group liquid)
`MLIQ` is optional and gated by `MOGP.flags & 0x1000`.

### Parsed `MLIQ` header fields (inferred)
| Offset | Type | Name (inferred) | Notes |
|--------|------|------------------|-------|
| 0x08 | int32 | vertCountX | Loaded into `this+0xF0` |
| 0x0C | int32 | vertCountY | Loaded into `this+0xF4` |
| 0x10 | int32 | tileCountX | Loaded into `this+0xF8` |
| 0x14 | int32 | tileCountY | Loaded into `this+0xFC` |
| 0x18 | vec3f | basePos | Loaded into `this+0x100` |
| 0x24 | uint16 | liquidFlags/type | Loaded into `this+0x10C` |
| 0x26 | bytes | vertexArray | Pointer saved at `this+0x110` |
| ... | bytes | tileArray | Pointer saved at `this+0x114` |

### MLIQ conversion behavior
- Per-vertex conversion mode branches on a flag bit (`(*(byte*)tileArray & 4)` in this decompile).
- One mode swaps only 32-bit height-like fields; the other swaps `uint16, uint16, uint32` per vertex record.
- This indicates at least two MLIQ per-vertex layouts in this build path.

## Optional MOGP chunk flags observed in parser
From `CreateOptionalDataPointers`:
- `0x0001` -> `MOBN` + `MOBR`
- `0x0004` -> `MOCV`
- `0x0200` -> `MOLR`
- `0x0400` -> `MPBV` + `MPBP` + `MPBI` + `MPBG`
- `0x0800` -> `MODR`
- `0x1000` -> `MLIQ`
- `0x20000` -> `MORI` + `MORB`

## Ghidra Notes
- `CMapObjGroup::Create()` at `0x002a1380`
- `CMapObjGroup::CreateDataPointers(unsigned char*)` at `0x002a1790`
- `CMapObjGroup::CreateOptionalDataPointers(unsigned char*)` at `0x002a1c4c`
- Assertions reference `MapObjRead.cpp` line region ~`0x1d9` through `0x2c6`.

## Confidence
- **High**: required chunk order, optional chunk gating bits, single-`MOTV` parse path, `MLIQ` presence/gating.
- **Medium**: semantic labels for several `MOGP`/`MLIQ` fields pending direct hex correlation.
