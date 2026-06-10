# MVER — WDT Version

## Summary
Version chunk for 0.5.3 WDT root parsing.

## Parent Chunk
Root-level WDT chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `iffChunk.token == 'MVER'` and `iffChunk.token=='MVER'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | version | WDT version value (numeric not yet extracted from decompile body) |

## Notes
- Appears in the same root assertion cluster as `MAIN`, `MPHD`, `MDNM`, `MONM`, `MARE`, `MAOF`.

## Confidence
- Presence and role: **High**
- Exact numeric version value: **Medium/Unknown**
