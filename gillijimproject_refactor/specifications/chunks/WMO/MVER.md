# MVER — WMO Version

## Summary
Version chunk for WMO file.

## Parent Chunk
Root-level WMO chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | Explicit check for version `0x10` (v14 path) |
| 0.7.0.3694 | Inferred same unless parser proves otherwise |

## Structure — Build 0.7.0.3694 (inferred, high confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | token | `MVER` |
| 0x04 | uint32 | size | Usually 4 |
| 0x08 | uint32 | version | Expected `0x10` in v14 lineage |

## Confidence
- **High**
