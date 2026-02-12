# VERS — MDX Version

## Summary
Version chunk for MDX model files.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Global-properties read path includes version handling |
| 0.6.0.3592 | Loader still expects `MDLX` format |
| 0.7.0.3694 | Inferred MDX continuity |

## Structure — Build 0.7.0.3694 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | version | MDX version value |

## Confidence
- **Medium**
