# MDLGEOSETSECTION — Geometry Set Section

## Summary
Primary mesh/geoset section in 0.5.3 model loader.

## Parent Chunk
MDLX section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLGEOSETSECTION@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | geosetCount | number of geosets or descriptors |
| 0x?? | ??? | geosetRecords | vertices/indices linkage unresolved |

## Ghidra Notes
- **Function address**: `???`
- **Key observations**: adjacent symbols include `CGeoset`, `CGeosetShared`, `CPrimitive`, supporting a geoset-centric runtime model.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x008345D8`, runtime symbols near `0x0083493C` and `0x008349B8`
