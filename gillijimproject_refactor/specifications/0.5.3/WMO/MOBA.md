# MOBA — WMO Batch/Material Batch Data

## Summary
WMO batch-related chunk referenced by 0.5.3 parser assertions.

## Parent Chunk
Likely within/adjacent to `MOGP` group parsing context

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion string `pIffChunk->token=='MOBA'` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | batchEntries | Expected batch stream; exact entry layout pending |

## Ghidra Notes
- **Function address**: `???` (assertion string at `0x008A286C`)
- **Parser pattern**: token-equality guard before chunk-specific parse
- **Key observations**: MOBA was already part of the 0.5.3 WMO group pipeline.

## Confidence
- **Medium-Low**

## References
- internal string evidence at `0x008A286C`
