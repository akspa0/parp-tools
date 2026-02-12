# TEXS — Texture References

## Summary
Texture reference records used by model materials.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed in `FUN_0044cec0` using token `0x53584554` (`TEXS`) |

## Structure — Build 0.7.0.3694 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | record[0x10C] * n | textureRecords | Loader enforces `sectionBytes % 0x10C == 0` |

`numTextures = sectionBytes / 0x10C`.

## Confidence
- **High (record size/count), Medium (inner field semantics)**
