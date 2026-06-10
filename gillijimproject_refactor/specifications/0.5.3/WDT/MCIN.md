# MCIN — Embedded Chunk Index Table

## Summary
Index-style terrain table token validated in 0.5.3 monolithic map parser flow.

## Parent Chunk
Monolithic WDT body (ADT-like subdomain)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `mIffChunk->token == 'MCIN'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | chunkIndex[] | per-chunk/tile offsets likely present |

## Ghidra Notes
- **Evidence**: `0x008A236C`
- **Confidence**: **Medium-Low**
