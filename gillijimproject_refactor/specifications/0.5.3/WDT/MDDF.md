# MDDF — Embedded Doodad Placement Data

## Summary
Placement chunk for doodad instances validated in the monolithic map parse path.

## Parent Chunk
Monolithic WDT body (ADT-like object layer)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `mIffChunk->token == 'MDDF'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | entries[] | placement entries referencing `MDNM` string table |

## Ghidra Notes
- **Evidence**: `0x008A2334`
- **Confidence**: **Medium-Low**
