# MCRF — Embedded Terrain Reference List

## Summary
Reference-list chunk validated under MCNK-level terrain parse in 0.5.3.

## Parent Chunk
`MCNK` (embedded in monolithic WDT flow)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `iffChunk->token=='MCRF'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | uint32[]? | refs | likely doodad/WMO reference indices |

## Ghidra Notes
- **Evidence**: `0x008A12DC`
- **Confidence**: **Medium-Low**
