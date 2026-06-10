# MPHD — WDT Global Header

## Summary
Global WDT header chunk controlling map-level behavior and/or table offsets.

## Parent Chunk
Root-level WDT chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `iffChunk.token=='MPHD'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags? | Likely map-global flags (`???`) |
| 0x04 | uint32 | data0? | Additional global metadata (`???`) |
| 0x08 | uint32 | data1? | Additional global metadata (`???`) |

## Notes
- Parsed as a root mandatory stage in the same cluster as `MVER`/`MAIN`.
- In 0.5.3 monolithic behavior, MPHD likely gates terrain/object follow-on parse behavior.

## Confidence
- Chunk presence: **High**
- Field semantics: **Low-Medium**
