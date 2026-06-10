# MDLSEQUENCESSECTION — Animation Sequence Section

## Summary
Named section type used by 0.5.3 model loader for animation sequence metadata.

## Parent Chunk
MDLX section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLSEQUENCESSECTION@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | sequenceCount | likely animation sequence count/records |
| 0x?? | ??? | sequenceRecords | pending decompile confirmation |

## Ghidra Notes
- **Function address**: `???`
- **Parser pattern**: section object creation in model load pipeline
- **Key observations**: explicit strongly-typed section class indicates non-trivial section-specific parse logic.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x008344A4`
