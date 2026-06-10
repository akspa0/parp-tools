# MODF — WMO Placements

## Summary
WMO placement token in 0.5.3 embedded object layer.

## Parent Chunk
Embedded ADT-like object region.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | Asserted as `mIffChunk->token == 'MODF'` |

## Structure — Build 0.5.3.3368
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | ???[] | entries | Placement entries referencing `MONM` string table |

## Confidence
- Presence: **High**
- Entry field map: **Low-Medium**
