# MLIQ — WMO Liquid Chunk

## Summary
WMO-local liquid geometry/height chunk confirmed by parser-side token check in 0.5.3.

## Parent Chunk
WMO group (`MOGP`) stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion string `pIffChunk->token == 'MLIQ'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | liquidHeader | Header + payload expected; full field map pending |

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x008A2930`)
- **Parser pattern**: token check prior to liquid decode path
- **Key observations**: MLIQ path is present independently of terrain `MCLQ` concerns.

## Confidence
- **Medium-Low**

## References
- internal string evidence at `0x008A2930`
