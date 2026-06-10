# MDLTEXTURESECTION — Texture Reference Section

## Summary
Section that stores texture references used by model materials/layers.

## Parent Chunk
MDLX section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLTEXTURESECTION@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | textureCount | texture reference count or table size |
| 0x?? | ??? | textureEntries | path/hash/flags unresolved |

## Ghidra Notes
- **Function address**: `???`
- **Key observations**: appears with neighboring `MDLTEXLAYER`/`MDLMATERIALSECTION` symbols, indicating linked parse flow.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x0083452C`
