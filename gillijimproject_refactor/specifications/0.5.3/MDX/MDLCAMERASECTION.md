# MDLCAMERASECTION — Embedded Camera Section

## Summary
Model camera section referenced by 0.5.3 loader type metadata.

## Parent Chunk
MDLX section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLCAMERASECTION@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | cameraCount | camera entry count |
| 0x?? | ??? | cameraEntries | camera transforms/settings unresolved |

## Ghidra Notes
- **Function address**: `???`
- **Key observations**: appears alongside event/attachment/emitter sections, indicating broad MDLX feature support in this build.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x00834798`
