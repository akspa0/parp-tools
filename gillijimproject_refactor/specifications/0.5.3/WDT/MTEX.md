# MTEX — Embedded Terrain Texture Name Table

## Summary
Texture-name table chunk validated in 0.5.3 embedded terrain parse stage.

## Parent Chunk
Monolithic WDT body (ADT-like terrain layer)

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | Assertion `mIffChunk->token == 'MTEX'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | char[] | textureStrings | null-terminated texture paths |

## Ghidra Notes
- **Evidence**: `0x008A2350`
- **Confidence**: **Medium-Low**
