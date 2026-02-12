# MODL — Model Header Block

## Summary
Model-level metadata including name/extents and base properties.

## Parent Chunk
Root-level MDX chunk stream.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.7.0.3694 | Confirmed via `FUN_0044ce10` and `FUN_00421a00` with token `0x4c444f4d` (`MODL`) |

## Structure — Build 0.7.0.3694
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x174 | uint8 | modelFlags | Loader uses low bits and bit2 (`FUN_0044ce10`) |

## Confidence
- Presence: **High**
- Partial field map: **Medium**
