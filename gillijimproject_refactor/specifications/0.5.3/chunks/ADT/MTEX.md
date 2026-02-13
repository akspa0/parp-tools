# MTEX — Terrain Texture Name Table

## Summary
ADT texture-name table token in 0.5.3 embedded terrain parse.

## Parent Chunk
Embedded ADT-like region under WDT map load.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `mIffChunk->token == 'MTEX'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | char[] | textureStrings | Null-terminated terrain texture paths |

## Confidence
- Presence/role: **High**
