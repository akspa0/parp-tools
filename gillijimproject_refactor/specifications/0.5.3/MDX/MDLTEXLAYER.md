# MDLTEXLAYER — Texture Layer Section/Entry

## Summary
Texture-layer structure used in material composition paths.

## Parent Chunk
Material-related model section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLTEXLAYER@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | layerFlags | likely blend/filter flags |
| 0x?? | ??? | textureIndex | likely index into texture section |

## Ghidra Notes
- **Function address**: `???`
- **Key observations**: `TexLayer`/`TexLayerShared` runtime symbols are present nearby, supporting layered material model.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x00834510`, shared-layer symbols near `0x008348B0`
