# MCNK — Terrain Chunk Container

## Summary
Primary terrain chunk token in 0.5.3 embedded parse flow.

## Parent Chunk
Embedded ADT-like region under WDT map load.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `iffChunk->token=='MCNK'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ??? | mcnkHeader | Header size/field map unresolved |
| 0x?? | subchunk[] | subchunks | Includes confirmed `MCLY` and `MCRF`; others may be constant-checked |

## Confidence
- Presence: **High**
- Header mapping: **Low-Medium**
