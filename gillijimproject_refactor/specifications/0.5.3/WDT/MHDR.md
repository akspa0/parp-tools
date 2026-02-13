# MHDR — Embedded Terrain Header (Within 0.5.3 WDT Parse Domain)

## Summary
Terrain-header token validated in the same parser region as other monolithic map chunks.

## Parent Chunk
Monolithic WDT body (ADT-like subdomain)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `mIffChunk->token=='MHDR'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | headerFields | likely chunk offset/section descriptors |

## Ghidra Notes
- **Evidence**: `0x008A2388`
- **Confidence**: **Medium-Low**
