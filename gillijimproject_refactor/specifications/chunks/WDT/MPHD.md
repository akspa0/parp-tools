# MPHD — WDT Header Flags

## Summary
Global WDT header flags and reserved metadata.

## Parent Chunk
Root-level WDT chunk.

## Builds Analyzed
| Build | Notes |
|---|---|
| 0.5.3.3368 | 128-byte MPHD header documented |
| 0.6.0.3592 | MPHD parse path preserved in WDT loader |
| 0.7.0.3694 | Inferred continuity |

## Structure — Build 0.7.0.3694 (inferred, medium-high confidence)
| Offset | Type | Name | Description |
|---|---|---|---|
| 0x00 | uint32 | flags | WDT flags (`???` complete bit map for 0.7) |
| 0x04 | uint32[31] | reserved | Reserved/unknown fields |

Expected payload size: `0x80` (128) bytes.

## Confidence
- Layout size: **High**
- Flag semantics: **Medium**
