# MDLBONESECTION — Bone/Skeleton Section

## Summary
Skeleton/bone section type used by the 0.5.3 model parser.

## Parent Chunk
MDLX section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLBONESECTION@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | boneCount | likely number of bones |
| 0x?? | ??? | boneRecords | transforms/parent links unresolved |

## Ghidra Notes
- **Function address**: `???`
- **Key observations**: section coexists with animation/keyframe section symbols, implying expected skeletal animation pipeline.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x008346F4`
