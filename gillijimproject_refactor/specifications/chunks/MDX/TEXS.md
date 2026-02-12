# TEXS — Texture References

## Summary
Texture reference records used by model materials.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | `MdxReadTextures` stage confirmed |
| 0.7.0.3694 | Inferred continuity |

## Structure — Build 0.7.0.3694 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | replaceableId | Replaceable texture ID |
| 0x04 | char[] | filename | Null-terminated texture path |
| ... | uint32 | flags | Texture flags (`???` exact semantics) |

## Confidence
- **Medium**
