# MDLMATERIALSECTION — Material Section

## Summary
Material definition section used by the 0.5.3 MDL/MDX loader.

## Parent Chunk
MDLX section stream

## Builds Analyzed
| Build | Size | Notes |
|-------|------|-------|
| 0.5.3.3368 | variable | RTTI/type string `. ?AUMDLMATERIALSECTION@@` present |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|--------|------|------|-------------|
| 0x00 | ??? | materialCount | likely count and array of material definitions |
| 0x?? | ??? | materialEntries | exact entry schema pending |

## Ghidra Notes
- **Function address**: `???`
- **Parser pattern**: section parsed into material-related runtime objects (`HMATERIAL`, shared material paths also present nearby)
- **Key observations**: strongly suggests dedicated material-section decoder in alpha client.

## Confidence
- **Low-Medium**

## References
- string evidence at `0x008344EC`, nearby material strings around `0x00834454`
