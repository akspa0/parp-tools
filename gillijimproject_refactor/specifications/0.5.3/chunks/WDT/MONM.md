# MONM — WMO Name Table

## Summary
String table containing WMO names for world model placement records.

## Parent Chunk
Root-level WDT chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `iffChunk.token=='MONM'` |

## Structure — Build 0.5.3.3368 (inferred)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | char[] | stringBlock | Null-terminated WMO paths |

## Confidence
- Chunk role: **High**
- Exact index/link behavior with `MODF`: **Medium**
