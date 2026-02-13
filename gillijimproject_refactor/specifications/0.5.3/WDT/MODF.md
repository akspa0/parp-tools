# MODF — Embedded WMO Placement Data

## Summary
Placement chunk for world model instances validated in 0.5.3 monolithic map parsing.

## Parent Chunk
Monolithic WDT body (ADT-like object layer)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `mIffChunk->token == 'MODF'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | entries[] | placement entries referencing `MONM` string table |

## Ghidra Notes
- **Evidence**: `0x008A2318`
- **Confidence**: **Medium-Low**
