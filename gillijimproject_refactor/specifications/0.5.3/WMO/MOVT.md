# MOVT — WMO Vertices

## Summary
Vertex-position chunk validated in 0.5.3 WMO parser path.

## Parent Chunk
WMO group (`MOGP`) stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion string `pIffChunk->token=='MOVT'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | float[3] * N | positions | Group vertex array |

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x008A28F8`)
- **Parser pattern**: token guard then chunk decoder
- **Key observations**: MOVT is in the same assertion cluster as MOGP/MOTV/MOPY/MLIQ.

## Confidence
- **Medium**

## References
- internal string evidence at `0x008A28F8`
