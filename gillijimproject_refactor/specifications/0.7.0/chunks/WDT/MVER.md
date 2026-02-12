# MVER — Version Chunk

## Summary
Declares file format version for WDT.

## Parent Chunk
Root-level WDT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.6.0.3592 | Parser expects `MVER` first |
| 0.7.0.3694 | Inferred same pattern |

## Structure — Build 0.7.0.3694 (inferred, high confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | token | `MVER` |
| 0x04 | uint32 | size | Usually 4 |
| 0x08 | uint32 | version | Version number |

## Confidence
- **High**
