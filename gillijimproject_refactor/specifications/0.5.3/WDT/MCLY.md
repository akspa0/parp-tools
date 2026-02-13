# MCLY — Embedded Terrain Layer Chunk

## Summary
Texture-layer chunk validated under embedded terrain parsing in 0.5.3.

## Parent Chunk
`MCNK` (embedded in monolithic WDT flow)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `iffChunk->token=='MCLY'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | layerEntries | likely texture layer descriptors |

## Ghidra Notes
- **Evidence**: `0x008A1254`
- **Confidence**: **Medium-Low**
