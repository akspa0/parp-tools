# XLDM — MDX/MDL File Header (Little-Endian 'MDLX')

## Summary
0.5.3 model loader checks the file magic against `'XLDM'`, the little-endian uint32 encoding of `MDLX`.

## Parent Chunk
Root-level model file header

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | 4 bytes magic + header | Assertion string `*((ULONG *) (fileData)) == 'XLDM'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | uint32 | magic | `MDLX` stored/read as `'XLDM'` in little-endian comparisons |
| 0x04 | ??? | headerBody | Subsequent section table/stream consumed by section-specific loaders |

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x00834364`)
- **Parser pattern**: header magic check precedes section iteration
- **Key observations**: confirms classic MDLX container semantics in this alpha build.

## Confidence
- **High** for magic check, **Low-Medium** for downstream layout.

## References
- internal string evidence at `0x00834364`
