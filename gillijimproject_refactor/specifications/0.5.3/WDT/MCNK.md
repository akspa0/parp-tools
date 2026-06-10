# MCNK — Embedded Terrain Chunk

## Summary
Terrain-cell chunk token validated in the same parse domain as WDT root processing, supporting monolithic behavior in 0.5.3.

## Parent Chunk
Monolithic WDT body (ADT-like subdomain)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `iffChunk->token=='MCNK'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | mcnkHeader | exact offsets unresolved |
| 0x?? | ??? | subchunks | includes confirmed `MCLY` and `MCRF` in parser assertions |

## Ghidra Notes
- **Evidence**: `0x008A126C`
- **Confidence**: **Medium** for presence, **Low** for field map
